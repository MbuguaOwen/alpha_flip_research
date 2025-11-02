import numpy as np

def permutation_test_series(values: np.ndarray, n_perm=500, rng_seed=123):
    """Two-sided permutation test vs zero-mean null by sign-flipping."""
    rng = np.random.default_rng(rng_seed)
    values = values[np.isfinite(values)]
    obs = np.nanmean(values)
    if not np.isfinite(obs):
        return 0.0, 1.0
    sur = []
    for _ in range(n_perm):
        signs = rng.choice([-1,1], size=len(values))
        sur.append(np.mean(values * signs))
    sur = np.array(sur)
    p = (np.sum(np.abs(sur) >= np.abs(obs)) + 1) / (len(sur) + 1)
    return obs, float(p)
