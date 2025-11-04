#!/usr/bin/env python
import os, json, numpy as np, pandas as pd


def gate_timeseries(p, thr, k, ema_span, min_sep_min):
    s = p.ewm(span=ema_span, adjust=False).mean() if ema_span > 1 else p
    k_ok = (s >= thr).rolling(k, min_periods=k).sum().fillna(0).astype(int) == k if k > 1 else (s >= thr)
    idx = k_ok.index[k_ok]
    alerts, last, delta = [], None, pd.Timedelta(minutes=min_sep_min)
    for t in idx:
        if last is None or t - last >= delta:
            alerts.append(t); last = t
    return pd.DatetimeIndex(alerts)


def spans_from_y(y):
    g = (y.ne(y.shift(1))).cumsum()
    return [(grp.index[0], grp.index[-1]) for val, grp in y.groupby(g) if int(grp.iloc[0]) == 1]


if __name__ == "__main__":
    df = pd.read_csv("outputs/hazard/hazard_probs.csv", parse_dates=["ts"]).set_index("ts").sort_index()
    p, y = df["p"], df["y"]
    spans = spans_from_y(y)
    n_days = max((df.index[-1] - df.index[0]).total_seconds() / 86400, 1e-9)
    PRE = 180

    rows = []
    for thr in np.round(np.arange(0.540, 0.590, 0.002), 3):
        for k in (1, 2):
            for ema in (1, 3):
                for sep in (30, 60):
                    A = gate_timeseries(p, thr, k, ema, sep)
                    # flip windows [flip-PRE, flip]
                    wins = [(b - pd.Timedelta(minutes=PRE), b) for (a, b) in spans]
                    def in_any_win(t):
                        for u, v in wins:
                            if u <= t <= v:
                                return True
                        return False
                    tp = sum(in_any_win(t) for t in A)
                    fa = len(A) - tp
                    rows.append({
                        "thr": thr, "k": k, "ema": ema, "sep": sep,
                        "alerts": len(A),
                        "fa_per_day": fa / n_days,
                        "tp": tp,
                        "coverage": tp / max(len(spans), 1),
                    })
    out = pd.DataFrame(rows)
    os.makedirs("outputs/hazard", exist_ok=True)
    out.to_csv("outputs/hazard/hazard_sweep.csv", index=False)
    best = out[(out.fa_per_day <= 2.0) & (out.coverage > 0)].sort_values(
        ["coverage", "fa_per_day", "thr"], ascending=[False, True, True]
    ).head(1)
    if len(best):
        best.iloc[0].to_json("outputs/hazard/operating_point.json", indent=2)
    print("saved sweep → outputs/hazard/hazard_sweep.csv")
