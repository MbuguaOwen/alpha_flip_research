import pandas as pd, numpy as np, itertools

def cpcv_split_by_months(idx, n_blocks=6, embargo_minutes=60, max_combinations=10):
    """Return list of (train_idx, test_idx) for CPCV over chronological month blocks with embargo."""
    ix = pd.DatetimeIndex(idx).sort_values()
    months = pd.PeriodIndex(ix, freq="M").asi8
    uniq = np.unique(months)
    # limit to first n_blocks unique months if too many
    if len(uniq) > n_blocks:
        uniq = uniq[-n_blocks:]
    blocks = [ix[months==u] for u in uniq]
    pairs = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            test = blocks[j]
            train_blocks = [k for k in range(len(blocks)) if k not in (i, j)]
            # Vectorized gather + concat to avoid append deprecation
            if len(train_blocks):
                parts = [blocks[k].view("i8") for k in train_blocks]
                joined = np.concatenate(parts) if len(parts) else np.array([], dtype="i8")
                train = pd.to_datetime(joined, utc=True)
            else:
                train = pd.DatetimeIndex([])
            # embargo around test
            emb = pd.Timedelta(minutes=embargo_minutes)
            test_start = test.min(); test_end = test.max()
            train = train[(train < test_start - emb) | (train > test_end + emb)]
            pairs.append((train, test))
    # cap combinations
    if len(pairs) > max_combinations:
        pairs = pairs[-max_combinations:]
    return pairs
