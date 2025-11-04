import pandas as pd
from src.gate import gate_timeseries


def test_gate_parity_fixture():
    df = pd.read_csv("tests/fixtures/hazard_probs_2025-06.csv", parse_dates=["ts"]).set_index("ts")
    p = df["p"]
    a_off = gate_timeseries(p, 0.558, 2, 3, 60)
    # emulate “live”: recompute gate over a streaming prefix
    seen = []
    last = None
    for t in p.index:
        a = gate_timeseries(p.loc[:t], 0.558, 2, 3, 60)
        if len(a) and (last is None or a[-1] != last):
            seen.append(a[-1]); last = a[-1]
    assert list(a_off) == seen
