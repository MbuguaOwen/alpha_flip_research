#!/usr/bin/env python
import os, sys
# Ensure repository root is on path when run from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, yaml, traceback
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
    cfg = yaml.safe_load(open(args.config, "r"))

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
        pb.advance()

        # 2) Micro features
        info("Computing micro features...")
        feats = build_micro_features(bars_1m, ticks, cfg["features"])
        pb.advance()

        # 3) Rolling robust normalization (causal)
        info("Normalizing features (rolling robust z)...")
        norm = RollingRobustZ(window_days=cfg["features"]["normalize"]["window_days"],
                              per_hour_of_day=cfg["features"]["normalize"]["per_hour_of_day"],
                              winsor_pct=cfg["features"]["normalize"]["winsor_pct"])
        feats_z = norm.transform(feats)
        pb.advance()

        # 4) Event study around flips
        info("Running event study (permutations)...")
        res = evt(flips, feats_z,
                  pre_minutes=cfg["labels"]["lead_window_pre_min"],
                  post_minutes=cfg["labels"]["post_window_min"],
                  n_perm=cfg["event_study"]["permutations"],
                  show_progress=True)
        pb.advance()

        # 5) FDR control across features-lags
        info("Applying FDR and saving results...")
        res["q_value"] = bh_fdr(res["p_value"].values, q=cfg["event_study"]["fdr_q"])
        out_csv = os.path.join(out_dir, "event_study_results.csv")
        res.to_csv(out_csv, index=False)
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
