import os, glob, pandas as pd, numpy as np
from .utils import ensure_datetime_index

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def load_ticks(globs, show_progress: bool = False):
    if not globs:
        return None
    paths = []
    for g in globs:
        paths.extend(glob.glob(g))
    if not paths:
        return None
    dfs = []
    pb = None
    if show_progress:
        try:
            from .cli import ProgressBar
            pb = ProgressBar(total=len(paths), prefix="Load ticks")
        except Exception:
            pb = None
    for i, p in enumerate(sorted(paths), start=1):
        df = pd.read_csv(p)
        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" not in df.columns or "price" not in df.columns or "qty" not in df.columns:
            if pb: pb.advance(1)
            continue
        df = ensure_datetime_index(df)
        cols = ["price", "qty"] + (["is_buyer_maker"] if "is_buyer_maker" in df.columns else [])
        dfs.append(df[cols])
        if pb:
            pb.update(i)
    if pb:
        pb.finish()
    if not dfs:
        return None
    out = pd.concat(dfs).sort_index()
    return out

def load_bars_1m(globs, show_progress: bool = False):
    if not globs:
        return None
    paths = []
    for g in globs:
        import glob as _glob
        paths.extend(_glob.glob(g))
    if not paths:
        return None
    dfs = []
    pb = None
    if show_progress:
        try:
            from .cli import ProgressBar
            pb = ProgressBar(total=len(paths), prefix="Load 1m bars")
        except Exception:
            pb = None
    for i, p in enumerate(sorted(paths), start=1):
        df = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp")
        dfs.append(df)
        if pb:
            pb.update(i)
    if pb:
        pb.finish()
    if not dfs:
        return None
    out = pd.concat(dfs).sort_index()
    return out

def maybe_make_synthetic(n_minutes=5000, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_minutes*60, freq="s")  # seconds
    # Geometric BM-ish price with regime flips
    r = rng.normal(0, 0.0005, size=len(idx))
    price = 100 + np.cumsum(r) + 0.2*np.sin(np.arange(len(idx))/10000)
    qty = rng.integers(1, 5, size=len(idx)).astype(float)
    is_buyer_maker = rng.integers(0, 2, size=len(idx))
    df = pd.DataFrame({"price": price, "qty": qty, "is_buyer_maker": is_buyer_maker}, index=idx)
    df.index.name = "timestamp"
    return df
