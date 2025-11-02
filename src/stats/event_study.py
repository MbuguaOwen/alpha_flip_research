import pandas as pd, numpy as np, sys
from .permutation import permutation_test_series


def _print_progress_bar(current: int, total: int, prefix: str = "Event study", bar_len: int = 30):
    if total <= 0:
        return
    frac = min(max(current / total, 0.0), 1.0)
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = int(frac * 100)
    sys.stdout.write(f"\r[{prefix}] |{bar}| {pct:3d}% ({current}/{total})")
    sys.stdout.flush()


def run_event_study(flips_index, features_df, pre_minutes=720, post_minutes=360, n_perm=500, show_progress: bool = True):
    """Align windows around flips and test pre-flip feature deviations with permutation tests.

    Parameters
    - flips_index: DatetimeIndex of regime flip times
    - features_df: DataFrame of features sampled at minute frequency
    - pre_minutes: int, minutes before flip to analyze (lags)
    - post_minutes: kept for API symmetry; unused here
    - n_perm: int, permutations per test
    - show_progress: bool, render a simple console progress bar
    """
    results = []

    total_iters = max(len(features_df.columns) * int(pre_minutes), 1)
    completed = 0
    if show_progress:
        _print_progress_bar(completed, total_iters)

    for col in features_df.columns:
        series = features_df[col]
        for lag in range(int(pre_minutes), 0, -1):
            # pre-flip time t = flip_time - lag
            values = []
            for t in flips_index:
                tt = t - pd.Timedelta(minutes=lag)
                if tt in series.index:
                    values.append(series.loc[tt])
            if len(values) >= 20:  # need sample size
                values = np.array(values, dtype=float)
                stat, p = permutation_test_series(values, n_perm=n_perm)
                results.append({"feature": col, "lag_min": -lag, "stat": float(np.nanmean(values)), "p_value": p})

            completed += 1
            # Update roughly at 1% increments to reduce console spam
            step = max(total_iters // 100, 1)
            if show_progress and (completed % step == 0 or completed == total_iters):
                _print_progress_bar(completed, total_iters)

    if show_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return pd.DataFrame(results)
