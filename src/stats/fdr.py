import numpy as np, pandas as pd

def bh_fdr(pvals, q=0.10):
    p = np.array(pvals, dtype=float)
    n = len(p)
    idx = np.argsort(p)
    ranks = np.arange(1, n+1)
    thresh = q * ranks / n
    passed = p[idx] <= thresh
    # return q-values (Benjamini–Hochberg adjusted)
    qvals = np.empty(n, dtype=float)
    qvals[idx] = np.minimum.accumulate((p[idx][::-1] * n / ranks[::-1]))[::-1]
    return qvals
