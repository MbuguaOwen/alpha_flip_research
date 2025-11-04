#!/usr/bin/env python
import os, json
import pandas as pd

# Ensure repository root is on path when run from scripts/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gate import gate_timeseries


def _load_operating_point(path: str):
    cfg = json.load(open(path, "r", encoding="utf-8"))
    # Support both compact (thr,k,ema,sep) and verbose keys
    thr = cfg.get("thr", cfg.get("alert_threshold"))
    k = cfg.get("k", cfg.get("confirm_k"))
    ema = cfg.get("ema", cfg.get("ema_span"))
    sep = cfg.get("sep", cfg.get("min_separation_min"))
    if thr is None or k is None or ema is None or sep is None:
        raise SystemExit("operating_point.json missing one of: thr/alert_threshold, k/confirm_k, ema/ema_span, sep/min_separation_min")
    return float(thr), int(k), int(ema), int(sep)


def main():
    PRE = 180
    op_path = os.path.join("outputs", "hazard", "operating_point.json")
    probs_path = os.path.join("outputs", "hazard", "hazard_probs.csv")
    out_dir = os.path.join("outputs", "hazard")
    os.makedirs(out_dir, exist_ok=True)

    thr, k, ema, sep = _load_operating_point(op_path)

    df = pd.read_csv(probs_path, parse_dates=["ts"]).set_index("ts").sort_index()
    if not {"p", "y"}.issubset(df.columns):
        raise SystemExit(f"hazard_probs.csv missing required columns p,y; got {list(df.columns)}")

    # Generate alerts at operating point
    A = gate_timeseries(df["p"], thr, k, ema, sep)

    # Build flip windows from y==1 runs
    g = (df["y"].ne(df["y"].shift(1))).cumsum()
    spans = [(grp.index[0], grp.index[-1]) for _, grp in df["y"].groupby(g) if int(grp.iloc[0]) == 1]
    wins = [(b - pd.Timedelta(minutes=PRE), b) for (_, b) in spans]

    def in_any_win(t):
        return any(u <= t <= v for (u, v) in wins)

    covered, leads = 0, []
    for (a, b) in spans:
        win = (b - pd.Timedelta(minutes=PRE), b)
        hits = [t for t in A if win[0] <= t <= win[1]]
        if hits:
            covered += 1
            leads.append((b - min(hits)).total_seconds() / 60.0)

    n_days = max((df.index[-1] - df.index[0]).total_seconds() / 86400.0, 1e-9)
    fa = sum(1 for t in A if not in_any_win(t))

    stats = {
        "episodes": len(spans),
        "covered": covered,
        "coverage": covered / max(len(spans), 1),
        "fa_per_day": fa / n_days,
        "lead_to_flip_min": (pd.Series(leads).describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_dict() if leads else None),
    }

    out_fp = os.path.join(out_dir, "hazard_eval.json")
    json.dump(stats, open(out_fp, "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

