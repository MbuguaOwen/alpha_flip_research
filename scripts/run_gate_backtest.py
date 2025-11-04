#!/usr/bin/env python
import os, sys
# Ensure repository root is on path when run from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, yaml, traceback, pandas as pd
from src.io import ensure_dirs
from src.stats.metrics import gate_eval_from_alerts
from src.cli import ProgressBar, info, ok, warn, error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--entries_csv", help="CSV of your micro entries with timestamp index and 'side'/'price' columns")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    out_dir = os.path.join(cfg["project"]["out_dir"], "gate_backtest")
    ensure_dirs([out_dir])

    pb = ProgressBar(total=3, prefix="GateBacktest")
    try:
        # Load hazard alerts
        info("Loading hazard probabilities and forming alerts...")
        hazard_csv = os.path.join(cfg["project"]["out_dir"], "hazard", "hazard_probs.csv")
        if not os.path.exists(hazard_csv):
            raise FileNotFoundError("Run run_hazard.py first to produce hazard_probs.csv")
        yhat = pd.read_csv(hazard_csv, index_col=0, parse_dates=True).squeeze("columns")
        alerts = (yhat >= cfg["hazard"]["alert_threshold"]).astype(int)
        pb.advance()

        # Entries handling
        if not args.entries_csv or not os.path.exists(args.entries_csv):
            warn("No entries CSV supplied. Running a dummy gate evaluation (coverage only).")
            cov = alerts.rolling("1D").mean().mean()
            with open(os.path.join(out_dir, "dummy_gate_stats.txt"), "w") as f:
                f.write(f"Avg alert density per day (fraction of minutes alerted): {cov:.4f}\n")
            pb.advance(); pb.finish()
            ok("Dummy gate stats emitted.")
            return

        info("Evaluating entries under alert gating...")
        entries = pd.read_csv(args.entries_csv, index_col=0, parse_dates=True)
        report = gate_eval_from_alerts(entries, alerts)
        report.to_csv(os.path.join(out_dir, "gate_eval.csv"), index=False)
        pb.advance(); pb.finish()
        ok("Gate backtest report saved.")
    except Exception as e:
        try:
            pb.finish()
        except Exception:
            pass
        error(f"Gate backtest failed: {e.__class__.__name__}: {e}")
        traceback.print_exc()
        raise SystemExit(1)

if __name__ == "__main__":
    main()
