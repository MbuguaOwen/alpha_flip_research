import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression

def _ols_slope_r2(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    n = int(mask.sum())
    if n < 2:
        return np.nan, np.nan
    x = np.arange(n).reshape(-1, 1)
    lr = LinearRegression().fit(x, y[mask])
    slope = lr.coef_[0]
    r2 = lr.score(x, y[mask])
    return slope, r2

def build_macro_regime(bars_1m: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build 4h macro bars → trend state (bull/bear/range) + vol bucket, with hysteresis."""
    macro = bars_1m.copy()
    macro["ret"] = np.log(macro["close"]).diff()
    # build macro bars
    bar = cfg["macro_bar"]
    agg = macro.resample(bar).agg({"open":"first","high":"max","low":"min","close":"last"})
    agg = agg.dropna(subset=["close"])  # ensure finite close
    # realized volatility proxy over macro bars; avoid implicit filling warnings
    close_ret = agg["close"].pct_change(fill_method=None)
    agg["rv"] = close_ret.rolling(cfg["vol_bucket"]["lookback_bars"]).apply(
        lambda s: np.sqrt(np.nansum((s.astype(float))**2)), raw=False
    )
    # trend via OLS slope on log-price
    look = cfg["detector"]["lookback_bars"]
    r2_min = cfg["detector"]["r2_min"]
    slopes, r2s = [], []
    logp = np.log(agg["close"]).replace([np.inf, -np.inf], np.nan)
    for i in range(len(agg)):
        if i < look:
            slopes.append(np.nan); r2s.append(np.nan); continue
        s, r2 = _ols_slope_r2(logp.iloc[i-look:i].values)
        slopes.append(s); r2s.append(r2)
    agg["slope"] = slopes
    agg["r2"] = r2s
    state = pd.Series("range", index=agg.index, dtype=object)
    state[(agg["slope"]>0)&(agg["r2"]>=r2_min)] = "bull"
    state[(agg["slope"]<0)&(agg["r2"]>=r2_min)] = "bear"
    # hysteresis: require persistence to flip
    h = cfg["detector"]["hysteresis_bars"]
    state_h = state.copy()
    last = state.iloc[0]
    cnt = 0
    for i, s in enumerate(state):
        if i==0: continue
        if s!= last:
            cnt += 1
            if cnt >= h:
                last = s
                cnt = 0
            else:
                state_h.iloc[i] = last
        else:
            cnt = 0
            state_h.iloc[i] = last
    agg["trend_state"] = state_h.values
    # vol buckets
    cuts = cfg["vol_bucket"]["cuts"]
    pct = agg["rv"].rank(pct=True)
    vb = pd.cut(pct, bins=[0.0]+cuts[1:-1]+[1.0], labels=["low","mid","high"], include_lowest=True)
    agg["vol_state"] = vb.astype(str)
    return agg

def find_flips(macro: pd.DataFrame) -> pd.DatetimeIndex:
    st = macro["trend_state"].astype(str)
    flips = st[st!=st.shift(1)].index[1:]
    return flips

def make_flip_labels(macro: pd.DataFrame, flips, horizon_min=180):
    """Binary label at 1m resolution: flip occurs within (t, t+H]."""
    idx = pd.date_range(macro.index.min(), macro.index.max(), freq="1min")
    y = pd.Series(0, index=idx, dtype=int)
    for t in flips:
        win = (y.index > t) & (y.index <= t + pd.Timedelta(minutes=horizon_min))
        y.loc[win] = 1
    # lead time (optional)
    lead = pd.Series(np.nan, index=y.index)
    for t in flips:
        mask = (lead.index <= t) & (lead.index > t - pd.Timedelta(minutes=horizon_min))
        lead.loc[mask] = (t - lead.index[mask]).total_seconds()/60.0
    return y, lead
