#!/usr/bin/env python
import os, sys
# Ensure repository root is on path when run from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, yaml, traceback
from src.io import ensure_dirs, maybe_make_synthetic, load_ticks, load_bars_1m
from src.ticks_to_bars import ticks_to_1m
from src.regimes import build_macro_regime, find_flips, make_flip_labels
from src.features.micro_features import build_micro_features
from src.features.normalization import RollingRobustZ
from src.stats.cpcv import cpcv_split_by_months
from src.models.hazard import train_hazard_logit
from src.stats.metrics import evaluate_hazard, save_metrics
from src.cli import ProgressBar, info, ok, warn, error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

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

        # Micro features (causal) + normalization
        info("Computing micro features...")
        feats = build_micro_features(bars_1m, ticks, cfg["features"])
        pb.advance()

        info("Normalizing features (rolling robust z)...")
        norm = RollingRobustZ(window_days=cfg["features"]["normalize"]["window_days"],
                              per_hour_of_day=cfg["features"]["normalize"]["per_hour_of_day"],
                              winsor_pct=cfg["features"]["normalize"]["winsor_pct"])
        X = norm.transform(feats)
        inc = cfg.get("features", {}).get("include")
        if inc:
            cols = [c for c in inc if c in X.columns]
            if cols:
                X = X[cols]
            else:
                warn("Configured features.include has no overlap with computed features; proceeding with all features.")
        X = X.reindex(y.index).dropna()
        y = y.reindex(X.index).fillna(0).astype(int)  # align
        pb.advance()

        # CPCV splits by month with embargo ~ H
        info("Creating CPCV splits...")
        splits = cpcv_split_by_months(X.index, n_blocks=cfg["cpcv"]["n_blocks"],
                                      embargo_minutes=H, max_combinations=cfg["cpcv"]["max_combinations"])
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

        info("Saving metrics and calibrated probabilities...")
        save_metrics(metrics, os.path.join(out_dir, "hazard_metrics.json"))
        yhat_series.to_csv(os.path.join(out_dir, "hazard_probs.csv"))
        pb.advance(); pb.finish()
        ok("Hazard evaluation complete: metrics & calibrated probs saved.")
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
