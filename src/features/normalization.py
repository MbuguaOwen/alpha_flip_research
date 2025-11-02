import pandas as pd, numpy as np

class RollingRobustZ:
    """Median/MAD rolling z-score, optionally per hour-of-day; strictly causal; winsorize tails."""
    def __init__(self, window_days=5, per_hour_of_day=True, winsor_pct=0.01):
        self.window_days = window_days
        self.per_hod = per_hour_of_day
        self.winsor = winsor_pct

    def _winsor(self, s, p):
        lo, hi = s.quantile(p), s.quantile(1-p)
        return s.clip(lo, hi)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        if self.per_hod:
            # group by hour-of-day
            x["_hod"] = x.index.hour
            outs = []
            for h, g in x.groupby("_hod"):
                g2 = g.drop(columns=["_hod"])
                w = f"{self.window_days}D"
                med = g2.rolling(w, closed="left").median()
                mad = g2.rolling(w, closed="left").apply(lambda s: np.nanmedian(np.abs(s - np.nanmedian(s)))+1e-9, raw=False)
                z = (g2 - med) / mad
                z = z.apply(self._winsor, p=self.winsor)
                outs.append(z)
            out = pd.concat(outs).sort_index()
        else:
            w = f"{self.window_days}D"
            med = x.rolling(w, closed="left").median()
            mad = x.rolling(w, closed="left").apply(lambda s: np.nanmedian(np.abs(s - np.nanmedian(s)))+1e-9, raw=False)
            out = (x - med)/mad
            out = out.apply(self._winsor, p=self.winsor)
        return out.dropna()
