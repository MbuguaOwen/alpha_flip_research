#!/usr/bin/env bash
set -e
python scripts/run_event_study.py --config configs/project.yaml
python scripts/run_hazard.py --config configs/project.yaml
python scripts/run_gate_backtest.py --config configs/project.yaml
