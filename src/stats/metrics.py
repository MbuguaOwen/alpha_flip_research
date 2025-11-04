import pandas as pd, numpy as np, json
from sklearn.metrics import brier_score_loss

def evaluate_hazard(X, y, model, cal_model, splits, H, alert_threshold=0.35, min_sep_min=30):
    preds = pd.Series(index=X.index, dtype=float)
    for (tr, te) in splits:
        proba_raw = model.predict_proba(X.loc[te])[:, 1]
        # If a global calibrator was fit, you may choose to apply it here.
        # Keep as raw by default to avoid leakage; optional: uncomment to use cal_model.
        proba = proba_raw
        preds.loc[te] = proba
    preds = preds.sort_index()
    # Guard metrics against stray NaNs / inf and improve numeric stability
    preds = preds.replace([float("inf"), float("-inf")], float("nan")).clip(1e-6, 1 - 1e-6).dropna()
    y_eval = y.loc[preds.index]
    # basic calibration metric
    brier = brier_score_loss(y_eval, preds)
    # alerts
    alerts = (preds >= alert_threshold).astype(int)
    # throttle
    last_on = None
    sep = pd.Timedelta(minutes=min_sep_min)
    for t in alerts.index:
        if alerts.loc[t]==1:
            if last_on is None or (t - last_on) >= sep:
                last_on = t
            else:
                alerts.loc[t] = 0
    # coverage: fraction of flips with alert in pre-window (H)
    flips = y[(y==1) & (y.shift(1)==0)].index  # first positive minute of a window
    covered = 0
    for t in flips:
        pre = alerts.loc[t - pd.Timedelta(minutes=H):t]
        if pre.any():
            covered += 1
    coverage = covered / max(1, len(flips))
    # false alarms per day
    fa_per_day = alerts.resample("1D").sum().mean()
    lead_times = []
    for t in flips:
        pre = alerts.loc[t - pd.Timedelta(minutes=H):t]
        if pre.any():
            first = pre[pre==1].index.min()
            lead_times.append((t-first).total_seconds()/60.0)
    lead_avg = float(np.mean(lead_times)) if lead_times else 0.0
    metrics = {
        "brier": float(brier),
        "flip_coverage": float(coverage),
        "false_alarms_per_day": float(fa_per_day),
        "lead_time_avg_min": lead_avg,
        "horizon_min": int(H)
    }
    return metrics, preds

def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def gate_eval_from_alerts(entries_df: pd.DataFrame, alerts: pd.Series):
    """Filter entries by alerts ON and compute basic before/after counts (placeholder for your costed sim)."""
    entries_df = entries_df.copy()
    entries_df.index = pd.to_datetime(entries_df.index, utc=True)
    alerts = alerts.reindex(entries_df.index).fillna(0)
    before = len(entries_df)
    after = int((alerts==1).sum())
    rate = after / max(1, before)
    return pd.DataFrame([{"entries_before": before, "entries_after_alert_gate": after, "kept_fraction": rate}])
