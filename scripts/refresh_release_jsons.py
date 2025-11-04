#!/usr/bin/env python
import os
import json
from glob import glob
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Ensure repository root modules are importable when run from scripts/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gate import gate_timeseries


def _contiguous_one_runs(y: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    y = y.sort_index().astype(int)
    grp_id = (y.ne(y.shift(1))).cumsum()
    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, grp in y.groupby(grp_id):
        if int(grp.iloc[0]) == 1:
            spans.append((grp.index[0], grp.index[-1]))
    return spans


def _infer_horizon_min(spans: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> int:
    if not spans:
        return 180  # sensible default
    # length in minutes approximated by number of 1-min samples in run
    lengths = []
    for a, b in spans:
        # +1 to count both endpoints if perfectly regular; fall back to timediff
        n = max(int(round((b - a).total_seconds() / 60.0)) + 1, 1)
        lengths.append(n)
    # choose the median to be robust to occasional gaps
    return int(np.median(lengths))


def recompute_metrics_for_release_dir(release_dir: str) -> Dict:
    probs_fp = os.path.join(release_dir, "hazard_probs.csv")
    op_fp = os.path.join(release_dir, "operating_point.json")
    out_fp = os.path.join(release_dir, "hazard_metrics.json")

    if not os.path.exists(probs_fp):
        raise FileNotFoundError(f"Missing hazard_probs.csv in {release_dir}")
    if not os.path.exists(op_fp):
        raise FileNotFoundError(f"Missing operating_point.json in {release_dir}")

    df = pd.read_csv(probs_fp, parse_dates=["ts"])  # expects columns: ts, p, y
    if not {"ts", "p", "y"}.issubset(df.columns):
        raise ValueError(f"{probs_fp} must contain columns ts,p,y; got {list(df.columns)}")
    df = df.sort_values("ts")
    p = pd.Series(df["p"].values, index=pd.to_datetime(df["ts"]))
    y = pd.Series(df["y"].values, index=pd.to_datetime(df["ts"]))

    # Brier score (manual to avoid dependencies)
    brier = float(np.mean((p.values - y.values) ** 2))

    # Operating point (thr, k, ema, sep)
    op = json.load(open(op_fp, "r", encoding="utf-8"))
    thr = float(op.get("thr", op.get("alert_threshold")))
    k = int(op.get("k", op.get("confirm_k", 1)))
    ema = int(op.get("ema", op.get("ema_span", 1)))
    sep = int(op.get("sep", op.get("min_separation_min", 0)))

    # Alerts using tuned gate
    alerts = gate_timeseries(p, thr=thr, k=k, ema_span=ema, min_sep_min=sep)

    # Flip windows and horizon
    spans = _contiguous_one_runs(y)
    H = _infer_horizon_min(spans)

    # Coverage and lead-time stats
    def in_any_win(t: pd.Timestamp) -> bool:
        for _, b in spans:
            if (b - pd.Timedelta(minutes=H)) <= t <= b:
                return True
        return False

    covered = 0
    leads: List[float] = []
    for (_, b) in spans:
        win_start = b - pd.Timedelta(minutes=H)
        hits_in_win = [t for t in alerts if (win_start <= t <= b)]
        if hits_in_win:
            covered += 1
            first_hit = min(hits_in_win)
            leads.append((b - first_hit).total_seconds() / 60.0)

    n_days = max((p.index[-1] - p.index[0]).total_seconds() / 86400.0, 1e-9)
    fa = sum(1 for t in alerts if not in_any_win(t))

    metrics = {
        "brier": brier,
        "flip_coverage": covered / max(len(spans), 1),
        "false_alarms_per_day": fa / n_days,
        "lead_time_avg_min": float(np.mean(leads)) if leads else 0.0,
        "horizon_min": int(H),
    }

    with open(out_fp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rel_root = os.path.join(root, "release")
    dirs = [d for d in glob(os.path.join(rel_root, "*")) if os.path.isdir(d)]
    if not dirs:
        print("[warn] No release/* directories found.")
        return

    for d in dirs:
        try:
            m = recompute_metrics_for_release_dir(d)
            name = os.path.basename(d)
            print(f"[ok] Updated {name}/hazard_metrics.json => "
                  f"brier={m['brier']:.4f}, coverage={m['flip_coverage']:.3f}, "
                  f"fa/day={m['false_alarms_per_day']:.2f}, lead_avg={m['lead_time_avg_min']:.1f}m, H={m['horizon_min']}")
        except Exception as e:
            print(f"[error] {d}: {e}")


if __name__ == "__main__":
    main()

