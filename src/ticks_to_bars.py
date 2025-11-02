import pandas as pd, numpy as np
from .utils import ensure_datetime_index, resample_ohlcv

def ticks_to_1m(ticks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticks to 1-minute OHLCV. Assumes tick index is seconds-level UTC."""
    df = ticks.copy()
    df = df.groupby(pd.Grouper(freq="1s")).agg({"price":"last","qty":"sum","is_buyer_maker":"last"})
    df = df.dropna(subset=["price"])
    bars = resample_ohlcv(df, "1min")
    # add realized spread proxy if possible
    return bars
