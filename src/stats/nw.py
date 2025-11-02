import numpy as np

def newey_west_variance(u, lag=5):
    T = len(u)
    gamma0 = np.var(u, ddof=1)
    var = gamma0
    for L in range(1, lag+1):
        w = 1 - L/(lag+1)
        cov = np.cov(u[L:], u[:-L], ddof=1)[0,1]
        var += 2*w*cov
    return var
