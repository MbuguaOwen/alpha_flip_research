#!/usr/bin/env python
import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repository root is on path when run from scripts/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gate import gate_timeseries


def main():
    probs_fp = os.path.join("outputs", "hazard", "hazard_probs.csv")
    out_dir = os.path.join("outputs", "hazard")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "hazard_gate_debug.png")

    df = pd.read_csv(probs_fp, parse_dates=["ts"]).set_index("ts").sort_index()
    if not {"p", "y"}.issubset(df.columns):
        raise SystemExit(f"hazard_probs.csv missing columns p,y; got {list(df.columns)}")

    thr, k, ema, sep = 0.558, 2, 3, 60
    A = gate_timeseries(df["p"], thr, k, ema, sep)

    fig, ax = plt.subplots(figsize=(13, 4))
    df["p"].plot(ax=ax, lw=0.8, color="tab:blue")
    ax.axhline(thr, ls="--", color="tab:red", alpha=0.8, label=f"thr={thr}")

    # Shade y==1 spans
    g = (df["y"].ne(df["y"].shift(1))).cumsum()
    for _, grp in df["y"].groupby(g):
        if int(grp.iloc[0]) == 1:
            ax.axvspan(grp.index[0], grp.index[-1], color="tab:green", alpha=0.15)

    # Mark alerts
    if len(A):
        ax.scatter(A, df.loc[A, "p"], s=12, color="tab:orange", zorder=3, label="alerts")

    ax.set_title("Hazard p, threshold, y==1 spans, and alerts")
    ax.set_ylabel("p")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[ok] wrote {out_png}")


if __name__ == "__main__":
    main()

