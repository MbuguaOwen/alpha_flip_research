from pandas import Timedelta, DatetimeIndex
import pandas as pd


def gate_timeseries(p: pd.Series, thr: float, k: int, ema_span: int, min_sep_min: int) -> DatetimeIndex:
    """Generate sparse alert times from a per-minute probability series.

    - p: minute-indexed probabilities in [0, 1]
    - thr: on-threshold
    - k: require k consecutive minutes above threshold
    - ema_span: optional EMA smoothing span (<=1 disables smoothing)
    - min_sep_min: cooldown between successive alerts
    """
    p = p.sort_index()
    s = p.ewm(span=ema_span, adjust=False).mean() if ema_span and ema_span > 1 else p
    on = (
        (s >= thr).rolling(k, min_periods=k).sum().fillna(0).astype(int) == k
        if k and k > 1 else (s >= thr)
    )
    idx = on.index[on]
    out: list[pd.Timestamp] = []
    last = None
    cooldown = Timedelta(minutes=int(min_sep_min) if min_sep_min else 0)
    for t in idx:
        if last is None or t - last >= cooldown:
            out.append(t)
            last = t
    return DatetimeIndex(out)


def gate_with_series_threshold(
    p: pd.Series,
    thr_series: pd.Series,
    k: int,
    ema_span: int,
    min_sep_min: int,
) -> DatetimeIndex:
    """Gate using a per-minute threshold series (e.g., from vol buckets).

    - p: minute-indexed probabilities in [0, 1]
    - thr_series: minute-indexed thresholds; will be aligned (pad) to p index
    - k: require k consecutive minutes above threshold
    - ema_span: optional EMA smoothing span (<=1 disables smoothing)
    - min_sep_min: cooldown between successive alerts
    """
    p = p.sort_index()
    s = p.ewm(span=ema_span, adjust=False).mean() if ema_span and ema_span > 1 else p
    thr = thr_series.sort_index().reindex(s.index, method="pad")
    cond = s >= thr
    on = (
        cond.rolling(k, min_periods=k).sum().fillna(0).astype(int) == k
        if k and k > 1 else cond
    )
    idx = on.index[on]
    out: list[pd.Timestamp] = []
    last = None
    cooldown = Timedelta(minutes=int(min_sep_min) if min_sep_min else 0)
    for t in idx:
        if last is None or t - last >= cooldown:
            out.append(t)
            last = t
    return DatetimeIndex(out)
