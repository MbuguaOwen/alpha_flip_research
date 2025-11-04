#!/usr/bin/env python
import os, sys
# Ensure repository root is on path when run from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, yaml, traceback
import json
from joblib import dump
import pandas as pd
from src.io import ensure_dirs, maybe_make_synthetic, load_ticks, load_bars_1m
from src.ticks_to_bars import ticks_to_1m
from src.regimes import build_macro_regime, find_flips, make_flip_labels
from src.features.micro_features import build_micro_features
from src.features.normalization import RollingRobustZ
from src.stats.cpcv import cpcv_split_by_months
from src.models.hazard import train_hazard_logit
from src.stats.metrics import evaluate_hazard, save_metrics
from src.cli import ProgressBar, info, ok, warn, error
from src.gate import gate_timeseries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    out_dir = os.path.join(cfg["project"]["out_dir"], "hazard")
    ensure_dirs([out_dir])

    steps_total = 9
    pb = ProgressBar(total=steps_total, prefix="Hazard")
    try:
        # Load/synthesize
        info("Loading ticks and bars...")
        ticks = load_ticks(cfg["data"]["ticks_glob"], show_progress=True)
        if ticks is None:
            info("No ticks found. Generating synthetic sample...")
            ticks = maybe_make_synthetic()
        bars_1m = load_bars_1m(cfg["data"]["bars_1m_glob"], show_progress=True)
        if bars_1m is None:
            bars_1m = ticks_to_1m(ticks)
        pb.advance()

        # Macro regimes & flips
        info("Building macro regime & finding flips...")
        macro = build_macro_regime(bars_1m, cfg["regime"])
        flips = find_flips(macro)
        pb.advance()

        # Labels for hazard task
        info("Constructing labels for hazard horizon...")
        H = cfg["hazard"]["flip_horizon_min"]
        y, lead_time = make_flip_labels(macro, flips, horizon_min=H)
        pb.advance()

        # Micro features (causal) + regime-aligned transform (no lookahead) + normalization
        info("Computing micro features...")
        feats = build_micro_features(bars_1m, ticks, cfg["features"])
        # Add imbalance_1s_against_regime = - R.shift(1) * imbalance_1s
        try:
            if "imbalance_1s" in feats.columns:
                R = macro["trend_state"].map({"bull": 1, "bear": -1}).fillna(0).astype(float)
                R_past = R.shift(1).reindex(feats.index, method="pad").fillna(0.0)
                feats["imbalance_1s_against_regime"] = - R_past * feats["imbalance_1s"]
            else:
                warn("imbalance_1s not available; cannot compute imbalance_1s_against_regime.")
        except Exception as _e:
            warn(f"Failed to compute imbalance_1s_against_regime: {_e}")
        pb.advance()

        info("Normalizing features (rolling robust z)...")
        norm = RollingRobustZ(window_days=cfg["features"]["normalize"]["window_days"],
                              per_hour_of_day=cfg["features"]["normalize"]["per_hour_of_day"],
                              winsor_pct=cfg["features"]["normalize"]["winsor_pct"])
        X = norm.transform(feats)
        feat_cfg = cfg.get("features", {})
        selected_cfg = feat_cfg.get("selected")
        include_cfg = feat_cfg.get("include")
        # keep only the columns you intend to model
        selected_names = [d["name"] for d in selected_cfg] if selected_cfg else include_cfg
        if selected_names:
            cols = [c for c in X.columns if c in selected_names]
            if cols:
                X = X[cols].copy()
            else:
                warn("Selected/include list has no overlap with computed features; proceeding with available features.")
        # single, authoritative validity mask (after selection)
        X = X.reindex(y.index)
        valid = X.notna().all(1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid].astype(int)
        pb.advance()

        # CPCV splits by month with embargo ~ H
        info("Creating CPCV splits...")
        splits = cpcv_split_by_months(X.index, n_blocks=cfg["cpcv"]["n_blocks"],
                                      embargo_minutes=H, max_combinations=cfg["cpcv"]["max_combinations"])
        # intersect with the cleaned X index and drop empty / single-class folds
        clean_splits = []
        for tr_idx, te_idx in splits:
            tr = X.index.intersection(tr_idx)
            te = X.index.intersection(te_idx)
            if len(tr) == 0 or len(te) == 0:
                continue
            if y.loc[tr].nunique() < 2:
                continue
            clean_splits.append((tr, te))
        splits = clean_splits
        assert len(splits) > 0, "No valid CPCV folds after cleaning."
        pb.advance()

        # Train hazard model (logit + isotonic if requested)
        info("Training hazard model...")
        model, cal_model = train_hazard_logit(X, y,
                                              class_weight=cfg["hazard"]["class_weight"],
                                              calibrate=cfg["hazard"]["calibrate"],
                                              splits=splits)
        pb.advance()

        # Evaluate out-of-fold
        info("Evaluating hazard model (OOF)...")
        metrics, yhat_series = evaluate_hazard(X, y, model, cal_model, splits, H,
                                               alert_threshold=cfg["hazard"]["alert_threshold"],
                                               min_sep_min=cfg["hazard"]["min_separation_min"])
        pb.advance()

        # --- SAVE: metrics + calibrated OOF probs + labels ---
        out_dir = os.path.join(cfg["project"]["out_dir"], "hazard")
        os.makedirs(out_dir, exist_ok=True)

        probs_fp   = os.path.join(out_dir, "hazard_probs.csv")
        metrics_fp = os.path.join(out_dir, "hazard_metrics.json")

        # align y to the prob index and coerce to int
        y_aligned = y.reindex(yhat_series.index).fillna(0).astype(int)

        df_out = pd.DataFrame({
            "ts": yhat_series.index,        # explicit timestamp column
            "p":  yhat_series.values,       # calibrated probabilities
            "y":  y_aligned.values          # labels (0/1)
        })
        df_out.to_csv(probs_fp, index=False)
        json.dump(metrics, open(metrics_fp, "w"))

        print(f"[ok] Hazard evaluation complete: {metrics_fp} and {probs_fp}")
        # Also compute and save sparse alert times for quick inspection (parity with live)
        gate = cfg.get("hazard", {})
        alerts = gate_timeseries(
            yhat_series,
            float(gate.get("alert_threshold", 0.5)),
            int(gate.get("confirm_k", 1)),
            int(gate.get("ema_span", 1)),
            int(gate.get("min_separation_min", 0)),
        )
        alerts_fp = os.path.join(out_dir, "hazard_alerts.csv")
        pd.DataFrame({"ts": alerts, "alert": 1}).to_csv(alerts_fp, index=False)
        print(f"[ok] saved alerts: {alerts_fp}")

        # --- ADD/KEEP at top of file ---
        # import os, json
        # from joblib import dump
        # import pandas as pd
        # from src.gate import gate_timeseries

        # --- ADD near the end, before the final print/return ---
        out_dir = os.path.join(cfg["project"]["out_dir"], "hazard")
        os.makedirs(out_dir, exist_ok=True)

        # 1) Save calibrated probabilities + labels (you already do this)
        probs_fp   = os.path.join(out_dir, "hazard_probs.csv")
        metrics_fp = os.path.join(out_dir, "hazard_metrics.json")
        (pd.DataFrame({"p": yhat_series}).join(y.rename("y"), how="left")
           .to_csv(probs_fp, index_label="ts"))
        json.dump(metrics, open(metrics_fp, "w"))

        # 2) Save the model, calibrator, and feature/normalization specs
        dump(model,      os.path.join(out_dir, "model.joblib"))
        if cal_model is not None:
            dump(cal_model, os.path.join(out_dir, "calibrator.joblib"))

        # Minimal feature spec so BT/live build the SAME inputs
        feat_spec = {
            "features_include": cfg["features"]["include"],
            "lags": cfg["features"]["lags"],
            "macro_bar": cfg["regime"]["macro_bar"],
            "selected_event_features": [{"name":"imbalance_1s","lag_min":-30}],
        }
        pd.Series(feat_spec).to_json(os.path.join(out_dir, "feature_spec.json"))

        # Normalization config so BT/live use same rolling robust-z settings
        norm_spec = cfg["features"]["normalize"]
        pd.Series(norm_spec).to_json(os.path.join(out_dir, "norm_config.json"))

        # 3) Save gate params (operating point) and emit alerts now for parity
        gate_cfg = cfg["hazard"]
        pd.Series({
            "alert_threshold": gate_cfg["alert_threshold"],
            "confirm_k": gate_cfg.get("confirm_k", 1),
            "ema_span": gate_cfg.get("ema_span", 1),
            "min_separation_min": gate_cfg.get("min_separation_min", 0),
            "notes": "frozen operating point for BT/live parity"
        }).to_json(os.path.join(out_dir, "operating_point.json"))

        A = gate_timeseries(
            yhat_series,
            gate_cfg["alert_threshold"],
            gate_cfg.get("confirm_k", 1),
            gate_cfg.get("ema_span", 1),
            gate_cfg.get("min_separation_min", 0),
        )
        pd.DataFrame({"ts": A, "alert": 1}).to_csv(os.path.join(out_dir, "hazard_alerts.csv"), index=False)

        print(f"[ok] Hazard evaluation complete: {metrics_fp} and {probs_fp}")
        print(f"[ok] saved alerts: {os.path.join(out_dir, 'hazard_alerts.csv')}")
        print(f"[ok] saved model: {os.path.join(out_dir, 'model.joblib')}")

        pb.advance(); pb.finish()
    except Exception as e:
        try:
            pb.finish()
        except Exception:
            pass
        error(f"Hazard run failed: {e.__class__.__name__}: {e}")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
