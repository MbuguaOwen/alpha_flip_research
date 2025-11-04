#!/usr/bin/env python
import os, sys
# Ensure repository root is on path when run from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, yaml, traceback, pandas as pd, numpy as np
from src.io import ensure_dirs, maybe_make_synthetic, load_ticks, load_bars_1m
from src.ticks_to_bars import ticks_to_1m
from src.regimes import build_macro_regime, find_flips
from src.features.micro_features import build_micro_features
from src.features.normalization import RollingRobustZ
from src.stats.event_study import run_event_study as evt
from src.stats.fdr import bh_fdr
from src.utils import ensure_datetime_index
from src.cli import ProgressBar, info, ok, warn, error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    # Write a small preregistration manifest for auditability
    try:
        prereg = {
            "macro_bar": cfg["regime"]["macro_bar"],
            "pre_minutes": cfg["event_study"]["pre_minutes"],
            "post_minutes": cfg["event_study"]["post_minutes"],
            "permutations": cfg["event_study"]["permutations"],
            "fdr_q": cfg["event_study"]["fdr_q"],
            "features_include": cfg.get("features", {}).get("include", []),
            "lags": cfg.get("features", {}).get("lags", cfg.get("event_study", {}).get("lags", [])),
        }
        os.makedirs(os.path.join(cfg["project"]["out_dir"], "event_study"), exist_ok=True)
        with open(os.path.join(cfg["project"]["out_dir"], "event_study", "preregistered.json"), "w") as f:
            import json
            json.dump(prereg, f, indent=2)
    except Exception:
        pass

    out_dir = os.path.join(cfg["project"]["out_dir"], "event_study")
    ensure_dirs([out_dir])

    steps_total = 6
    pb = ProgressBar(total=steps_total, prefix="EventStudy")
    try:
        # 0) Load or synthesize data
        info("Loading ticks and bars...")
        ticks = load_ticks(cfg["data"]["ticks_glob"], show_progress=True)
        if ticks is None:
            info("No ticks found. Generating synthetic sample...")
            ticks = maybe_make_synthetic()
        bars_1m = load_bars_1m(cfg["data"]["bars_1m_glob"], show_progress=True)
        if bars_1m is None:
            bars_1m = ticks_to_1m(ticks)
        pb.advance()

        # 1) Macro regime & flips
        info("Building macro regime & finding flips...")
        macro = build_macro_regime(bars_1m, cfg["regime"])
        flips = find_flips(macro)
        # Robust UP/DOWN split from regime itself
        try:
            R_raw = macro["trend_state"]
            if getattr(R_raw.dtype, "kind", "") not in "biufc":
                R = R_raw.map({"bull": +1, "bear": -1}).fillna(0).astype("int64")
            else:
                R = np.sign(pd.to_numeric(R_raw, errors="coerce")).fillna(0).astype("int64")
            flip_mask = R.ne(R.shift(1))
            flip_times = R.index[flip_mask]
            delta = (R - R.shift(1)).reindex(flip_times).fillna(0).astype("int64")
            flips_up_idx = pd.DatetimeIndex(flip_times[delta == 2])   # -1 -> +1
            flips_dn_idx = pd.DatetimeIndex(flip_times[delta == -2])  # +1 -> -1
        except Exception:
            flips_up_idx = pd.DatetimeIndex([])
            flips_dn_idx = pd.DatetimeIndex([])
        pb.advance()

        # 2) Micro features
        info("Computing micro features...")
        feats = build_micro_features(bars_1m, ticks, cfg["features"])
        # Add regime-aligned transform for imbalance (no lookahead)
        try:
            if "imbalance_1s" in feats.columns:
                R = macro["trend_state"].map({"bull": 1, "bear": -1}).fillna(0).astype(float)
                R_past = R.shift(1).reindex(feats.index, method="pad").fillna(0.0)
                feats["imbalance_1s_against_regime"] = - R_past * feats["imbalance_1s"]
        except Exception as _e:
            warn(f"Failed to compute imbalance_1s_against_regime: {_e}")
        pb.advance()

        # 3) Rolling robust normalization (causal)
        info("Normalizing features (rolling robust z)...")
        norm = RollingRobustZ(window_days=cfg["features"]["normalize"]["window_days"],
                              per_hour_of_day=cfg["features"]["normalize"]["per_hour_of_day"],
                              winsor_pct=cfg["features"]["normalize"]["winsor_pct"])
        feats_z = norm.transform(feats)
        # Optional: restrict to configured feature subset or frozen selected list
        feat_cfg = cfg.get("features", {})
        selected = feat_cfg.get("selected")
        inc = feat_cfg.get("include")
        if selected:
            sel_names = [str(it.get("name")) for it in selected if it and ("name" in it)]
            cols = [c for c in sel_names if c in feats_z.columns]
            if cols:
                feats_z = feats_z[cols]
            else:
                warn("Configured features.selected has no overlap with computed features; proceeding with available features.")
        elif inc:
            cols = [c for c in inc if c in feats_z.columns]
            if cols:
                feats_z = feats_z[cols]
            else:
                warn("Configured features.include has no overlap with computed features; proceeding with all features.")
        # Diagnostics: filter flips away from boundaries and count valid non-NaN per (feature, lag)
        try:
            pre = int(cfg["event_study"]["pre_minutes"])
            post = int(cfg["event_study"]["post_minutes"])
            # Lags precedence: features.selected -> features.lags -> event_study.lags
            if selected:
                lags = [int(it.get("lag_min")) for it in selected if (it is not None and ("lag_min" in it))]
            else:
                lags = list(cfg.get("features", {}).get("lags", cfg.get("event_study", {}).get("lags", [])))
            start_ok = bars_1m.index.min() + pd.Timedelta(minutes=pre + max(abs(int(l)) for l in lags) if lags else pre)
            end_ok = bars_1m.index.max() - pd.Timedelta(minutes=post)
            X = feats_z

            def _valid(idx, tag):
                idx = pd.DatetimeIndex(idx)
                valid = idx[(idx >= start_ok) & (idx <= end_ok)]
                info(f"[{tag}] flips total={len(idx)}, valid={len(valid)} | window=[{start_ok} .. {end_ok}] | pre+maxlag={pre + (max(abs(int(l)) for l in lags) if lags else 0)}")
                return valid

            # Apply to pooled/up/down
            flips = _valid(flips, "pooled")
            flips_up_idx = _valid(flips_up_idx, "up")
            flips_dn_idx = _valid(flips_dn_idx, "down")

            # Per-(feature, lag) valid counts on pooled flips
            rows = []
            feat_list = [c for c in X.columns if (not inc) or (c in inc)]
            for f in feat_list:
                for lag in lags:
                    t_idx = flips + pd.Timedelta(minutes=int(lag))
                    s = X.reindex(t_idx)[f]
                    n = int(s.notna().sum())
                    rows.append({"feature": f, "lag_min": int(lag), "valid_nonNaN": n})
            if rows:
                import os as _os
                dbg = pd.DataFrame(rows).sort_values(["feature", "lag_min"])  # type: ignore[name-defined]
                _os.makedirs(out_dir, exist_ok=True)
                dbg.to_csv(_os.path.join(out_dir, "valid_counts.csv"), index=False)
                info(f"[diag] wrote valid_counts.csv with {len(dbg)} rows")

            # === Quantiles by regime and pre-flip windows ===
            try:
                qs = [0.05, 0.25, 0.5, 0.75, 0.95]
                feature = "imbalance_1s"
                q_lags = [-30]  # extend if desired
                if feature in X.columns:
                    rows_q = []
                    R_past = R.shift(1).reindex(X.index, method="pad")
                    bull_idx = X.index[R_past == +1]
                    bear_idx = X.index[R_past == -1]
                    for tag, idx in [("bull_all", bull_idx), ("bear_all", bear_idx)]:
                        s = X.loc[idx, feature].dropna()
                        if len(s):
                            q = s.quantile(qs)
                            rows_q.append({
                                "scope": tag,
                                "type": "regime",
                                "lag_min": None,
                                **{f"q{int(p*100)}": float(q[p]) for p in qs},
                                "n": int(len(s)),
                            })

                    def preflip_quantiles(flips_idx, label):
                        for lag in q_lags:
                            t_idx = pd.DatetimeIndex(flips_idx) + pd.Timedelta(minutes=int(lag))
                            s = X.reindex(t_idx)[feature].dropna()
                            if len(s):
                                q = s.quantile(qs)
                                rows_q.append({
                                    "scope": label,
                                    "type": "preflip",
                                    "lag_min": int(lag),
                                    **{f"q{int(p*100)}": float(q[p]) for p in qs},
                                    "n": int(len(s)),
                                })

                    preflip_quantiles(flips_up_idx, "up_flips")
                    preflip_quantiles(flips_dn_idx, "down_flips")
                    preflip_quantiles(flips, "pooled_flips")

                    if rows_q:
                        qdf = pd.DataFrame(rows_q)
                        qdf.to_csv(_os.path.join(out_dir, "imbalance_quantiles.csv"), index=False)
                        info(f"[diag] wrote imbalance_quantiles.csv with {len(qdf)} rows")
                else:
                    warn("[diag] imbalance_1s not in normalized X; skipping imbalance_quantiles.csv")
            except Exception as _e:
                warn(f"[diag] imbalance_quantiles skipped: {_e}")
        except Exception as _e:
            warn(f"[diag] valid_counts skipped: {_e}")
        pb.advance()

        # 4) Event study around flips
        info("Running event study (permutations)...")
        pre_m = cfg.get("event_study", {}).get("pre_minutes", cfg["labels"]["lead_window_pre_min"])
        post_m = cfg.get("event_study", {}).get("post_minutes", cfg["labels"]["post_window_min"])
        n_perm = cfg.get("event_study", {}).get("permutations", 500)
        # Prefer frozen selected lags, then feature-level lags, then event_study.lags
        feat_cfg = cfg.get("features", {})
        selected = feat_cfg.get("selected")
        if selected:
            lags_cfg = sorted(set(int(it.get("lag_min")) for it in selected if (it and ("lag_min" in it))))
        else:
            lags_cfg = feat_cfg.get("lags", cfg.get("event_study", {}).get("lags"))
        min_ev = int(cfg.get("event_study", {}).get("min_events_per_test", cfg.get("event_study", {}).get("min_samples", 20)))
        res = evt(
            flips,
            feats_z,
            pre_minutes=int(pre_m),
            post_minutes=int(post_m),
            n_perm=int(n_perm),
            show_progress=True,
            lags=lags_cfg,
            min_events=min_ev,
        )
        # Directional subsets
        res_up = evt(
            flips_up_idx,
            feats_z,
            pre_minutes=int(pre_m),
            post_minutes=int(post_m),
            n_perm=int(n_perm),
            show_progress=True,
            lags=lags_cfg,
            min_events=min_ev,
        )
        res_dn = evt(
            flips_dn_idx,
            feats_z,
            pre_minutes=int(pre_m),
            post_minutes=int(post_m),
            n_perm=int(n_perm),
            show_progress=True,
            lags=lags_cfg,
            min_events=min_ev,
        )
        pb.advance()

        # 5) FDR control across features-lags
        info("Applying FDR and saving results...")
        out_csv = os.path.join(out_dir, "event_study_results.csv")
        out_csv_up = os.path.join(out_dir, "event_study_results_up.csv")
        out_csv_dn = os.path.join(out_dir, "event_study_results_down.csv")
        if res is None or len(res) == 0 or ("p_value" not in res.columns):
            warn("Event study produced no valid tests (insufficient samples for all lags). Writing empty results.")
            empty = pd.DataFrame(columns=["feature","lag_min","stat","p_value","q_value"])
            empty.to_csv(out_csv, index=False)
            empty.to_csv(out_csv_up, index=False)
            empty.to_csv(out_csv_dn, index=False)
        else:
            res["q_value"] = bh_fdr(res["p_value"].values, q=cfg["event_study"]["fdr_q"])
            res.to_csv(out_csv, index=False)
            # Directional q-values & saves
            if res_up is not None and len(res_up) > 0 and ("p_value" in res_up.columns):
                res_up["q_value"] = bh_fdr(res_up["p_value"].values, q=cfg["event_study"]["fdr_q"])
                res_up.to_csv(out_csv_up, index=False)
            else:
                pd.DataFrame(columns=["feature","lag_min","stat","p_value","q_value"]).to_csv(out_csv_up, index=False)
            if res_dn is not None and len(res_dn) > 0 and ("p_value" in res_dn.columns):
                res_dn["q_value"] = bh_fdr(res_dn["p_value"].values, q=cfg["event_study"]["fdr_q"])
                res_dn.to_csv(out_csv_dn, index=False)
            else:
                pd.DataFrame(columns=["feature","lag_min","stat","p_value","q_value"]).to_csv(out_csv_dn, index=False)
        pb.advance(); pb.finish()
        ok(f"Event study complete: {out_csv}")
    except Exception as e:
        try:
            pb.finish()
        except Exception:
            pass
        error(f"Event study failed: {e.__class__.__name__}: {e}")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
