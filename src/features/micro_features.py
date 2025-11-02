import pandas as pd, numpy as np
from ..utils import resample_ohlcv

def _rolling_mad(x):
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med)) + 1e-12

def build_micro_features(bars_1m: pd.DataFrame, ticks: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return a 1-minute indexed DataFrame of micro features (causal)."""
    b = bars_1m.copy()
    b["ret"] = np.log(b["close"]).diff()
    p = b["close"]

    # Core features requested
    # ret_1m (causal)
    ret_1m = b["ret"].shift(1)

    # rv_1m realized variance from 1s log-returns within each minute
    rv_1m = None
    try:
        p1s = ticks["price"].groupby(pd.Grouper(freq="1s")).last().dropna()
        r1s = np.log(p1s).diff()
        rv_1m = r1s.pow(2).resample("1min").sum().shift(1)
    except Exception:
        rv_1m = b["ret"].pow(2).shift(1)

    # z_vol_1m: rolling z-score of volume at 1m
    vol = b.get("volume")
    z_vol_1m = None
    if vol is not None:
        w = int(cfg.get("params", {}).get("vol_z_win", 256))
        mu = vol.rolling(w, min_periods=max(16, w//4)).mean()
        sd = vol.rolling(w, min_periods=max(16, w//4)).std()
        z_vol_1m = ((vol - mu) / (sd + 1e-12)).shift(1)

    # trade_rate_1s: average trades per second over the minute
    try:
        trades_1s = ticks.groupby(pd.Grouper(freq="1s")).size()
        trade_rate_1s = trades_1s.resample("1min").mean().shift(1)
    except Exception:
        trade_rate_1s = None

    # imbalance_1s: per-second imbalance averaged over minute
    imbalance_1s = None
    if "is_buyer_maker" in ticks.columns:
        t = ticks.copy()
        sign = np.where(t["is_buyer_maker"] > 0, 1, -1)
        t["signed_qty"] = sign * t["qty"]
        sq = t["signed_qty"].groupby(pd.Grouper(freq="1s")).sum()
        vq = t["qty"].groupby(pd.Grouper(freq="1s")).sum()
        imb_1s = (sq / (vq + 1e-12)).clip(-1, 1)
        imbalance_1s = imb_1s.resample("1min").mean().shift(1)

    # Liquidity stress proxy (already present)
    vol_ret = b["ret"].rolling(64, min_periods=64).std()
    liq_stress = (b["ret"].abs() / (vol_ret**0.5 + 1e-12)).shift(1)

    # Legacy features (kept for completeness, still causal)
    # Compression: BB width pct & Donchian width pct
    w_bb = cfg["params"]["bb_win"]
    mu = p.rolling(w_bb, min_periods=w_bb).mean()
    sd = p.rolling(w_bb, min_periods=w_bb).std()
    bb_width = (mu + 2*sd) - (mu - 2*sd)
    bb_width_pct = (bb_width / p).shift(1)
    w_don = cfg["params"]["donchian_win"]
    don_h = b["high"].rolling(w_don, min_periods=w_don).max()
    don_l = b["low"].rolling(w_don, min_periods=w_don).min()
    don_width_pct = ((don_h - don_l)/p).shift(1)

    # Vol-of-vol
    rv = b["ret"].rolling(32, min_periods=32).std()
    vov = rv.rolling(16, min_periods=16).std().shift(1)

    # OFI / buy-maker share if available in ticks
    ofi = None; bm_share = None
    if "is_buyer_maker" in ticks.columns:
        t = ticks.copy()
        sign = np.where(t["is_buyer_maker"]>0, 1, -1)
        t["signed_qty"] = sign * t["qty"]
        ofi_1s = t["signed_qty"].groupby(pd.Grouper(freq="1s")).sum()
        ofi_1m = ofi_1s.resample("1min").sum().fillna(0.0)
        ofi = ofi_1m.ewm(span=cfg["params"]["ofi_win"], adjust=False).mean().shift(1)
        buys_1s = (t["is_buyer_maker"]>0).astype(int).groupby(pd.Grouper(freq="1s")).sum()
        trades_1s = t["is_buyer_maker"].groupby(pd.Grouper(freq="1s")).size()
        share_1m = (buys_1s.resample("1min").sum() / trades_1s.resample("1min").sum()).fillna(0.5)
        bm_share = share_1m.ewm(span=cfg["params"]["ofi_win"], adjust=False).mean().shift(1)

    # Skew/Kurt (rolling, causal)
    skew = b["ret"].rolling(cfg["params"]["skew_win"]).skew().shift(1)
    kurt = b["ret"].rolling(cfg["params"]["kurt_win"]).kurt().shift(1)

    # ACF approximations
    acf1 = b["ret"].rolling(128, min_periods=128).apply(lambda s: s.autocorr(1), raw=False).shift(1)
    acf5 = b["ret"].rolling(128, min_periods=128).apply(lambda s: s.autocorr(5), raw=False).shift(1)
    acf10= b["ret"].rolling(128, min_periods=128).apply(lambda s: s.autocorr(10), raw=False).shift(1)

    # Seasonality
    hod = b.index.hour + b.index.minute/60.0
    sin = np.sin(2*np.pi*hod/24.0); cos = np.cos(2*np.pi*hod/24.0)

    out = pd.DataFrame({
        # Core set
        "ret_1m": ret_1m,
        "rv_1m": rv_1m,
        "z_vol_1m": z_vol_1m,
        "trade_rate_1s": trade_rate_1s,
        "imbalance_1s": imbalance_1s,
        "liq_stress": liq_stress,
        # Legacy
        "bb_width_pct": bb_width_pct,
        "don_width_pct": don_width_pct,
        "vov": vov,
        "skew": skew,
        "kurt": kurt,
        "acf1": acf1,
        "acf5": acf5,
        "acf10": acf10,
        "season_sin": sin,
        "season_cos": cos
    }, index=b.index)

    if ofi is not None:
        out["ofi_ewm"] = ofi.reindex(out.index)
    if bm_share is not None:
        out["bm_share_ewm"] = bm_share.reindex(out.index)

    return out.dropna()
