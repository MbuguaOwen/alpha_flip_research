import pandas as pd, numpy as np

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
        out.index = idx
        return out.sort_index()
    # Try to find timestamp column
    for c in ["timestamp","time","datetime","date"]:
        if c in out.columns:
            ts = out[c]
            if np.issubdtype(ts.dtype, np.number):
                # Detect epoch unit by magnitude: s(1e9), ms(1e12-13), us(1e15-16), ns(1e18)
                try:
                    vmax = float(pd.Series(ts).dropna().max())
                except Exception:
                    vmax = float(ts.max())
                if vmax > 1e16:
                    unit = "ns"
                elif vmax > 1e13:
                    unit = "us"
                elif vmax > 1e10:
                    unit = "ms"
                else:
                    unit = "s"
                idx = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
            else:
                idx = pd.to_datetime(ts, utc=True, errors="coerce")
            out = out.drop(columns=[c])
            out.index = idx
            return out.sort_index()
    raise ValueError("No datetime index or timestamp column found.")

def resample_ohlcv(ticks_1s: pd.DataFrame, freq="1min"):
    # Expect second-level tick stream; aggregate
    price = ticks_1s["price"]
    o = price.resample(freq).first()
    h = price.resample(freq).max()
    l = price.resample(freq).min()
    c = price.resample(freq).last()
    v = ticks_1s["qty"].resample(freq).sum() if "qty" in ticks_1s.columns else None
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
    if v is not None:
        out["volume"] = v
    return out.dropna()
