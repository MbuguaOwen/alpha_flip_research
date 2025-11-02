# Alpha Flip Research Scaffold

This repo is a **scientific, leak-proof pipeline** for discovering and validating **micro precursors** to **macro regime flips**, starting with an **event study** (prove signals exist) and then a **hazard model** (predict flips without lookahead).

## Core Principles
- **Event Study First**: We only model signals that show pre-flip signatures under permutation tests with FDR control.
- **No Lookahead**: All features are causal. Normalization is **rolling, robust, and hour-of-day aware**.
- **CPCV**: Combinatorial purged CV with embargo ≥ flip horizon.
- **Order Flow Optional**: If your ticks include `is_buyer_maker`, we compute OFI and buy-maker share features.
- **Reproducible**: YAML configs, deterministic seeds, one change per commit.

## Data Expectations
- **Ticks CSV** (per symbol per day/month) OR a single big CSV. The scripts will glob using the config.
- Required columns (case-insensitive): 
  - `timestamp` (ms or ns epoch, or ISO string),
  - `price` (float),
  - `qty` (float or int),
  - `is_buyer_maker` (0/1) — optional but recommended.
- Timezone: assumed **UTC**.

If no real data is found, the pipeline will synthesize a small dataset to demonstrate end-to-end execution.

## Quickstart
```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1) Event study (prove signals exist)
python scripts/run_event_study.py --config configs/project.yaml

# 2) Hazard model for flip prediction (no lookahead)
python scripts/run_hazard.py --config configs/project.yaml

# Optional: Backtest gating logic around alerts (expects entries CSV if you have one)
python scripts/run_gate_backtest.py --config configs/project.yaml
```

## Outputs
- `outputs/event_study/` — per-feature pre-flip signatures, permutation p-values, FDR q-values, CSV + PNG plots.
- `outputs/hazard/` — calibrated flip probabilities, CPCV metrics (Brier, flip coverage, false alarms/day), diagnostics.
- `outputs/reports/` — markdown summaries and CSV scorecards.

## Config
See `configs/project.yaml` for paths, regime detector params, feature windows, flip horizon, CPCV settings, and thresholds.
