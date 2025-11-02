#!/usr/bin/env python
import os, sys
# Ensure repository root is on path when run from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, json
import numpy as np, pandas as pd, yaml


def bh_qvalues(p):
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    q = np.empty(m, dtype=float)
    prev = 1.0
    # Benjamini–Hochberg step-up, monotone
    for r, idx in enumerate(order[::-1], start=1):
        k = m - r + 1
        val = p[idx] * m / k
        prev = min(prev, val)
        q[idx] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", default="outputs/event_study/event_study_results.csv")
    ap.add_argument("--out_dir", default="outputs/event_study")
    ap.add_argument("--features", nargs="+", default=["ret_1m","rv_1m","z_vol_1m","trade_rate_1s","imbalance_1s","liq_stress"])
    ap.add_argument("--lags", nargs="+", type=int, default=[-180,-120,-90,-60,-30])
    ap.add_argument("--q", type=float, default=0.10, help="FDR threshold within subset")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.results_csv)
    # Filter to pre-registered family
    sub = df[df["feature"].isin(args.features) & df["lag_min"].isin(args.lags)].copy()
    if sub.empty:
        raise SystemExit("No rows matched the pre-registered family. Check features/lags or results path.")

    sub["q_sub"] = bh_qvalues(sub["p_value"].values)
    sub = sub.sort_values(["q_sub","p_value","stat"], ascending=[True, True, False])

    # Save subset table
    sub_out = os.path.join(args.out_dir, "event_study_subset.csv")
    sub.to_csv(sub_out, index=False)

    # Winners @ q_sub<=q
    winners = sub[sub["q_sub"] <= args.q].copy()
    winners_out = os.path.join(args.out_dir, "event_study_winners.csv")
    winners.to_csv(winners_out, index=False)

    # Emit YAML whitelist for downstream
    selected = [{"name": str(r.feature), "lag_min": int(r.lag_min)} for r in winners.itertuples()]
    whitelist = {"features": {"selected": selected}}
    yaml_out = os.path.join(args.out_dir, "selected_features.yaml")
    with open(yaml_out, "w") as f:
        yaml.safe_dump(whitelist, f, sort_keys=False)

    # Save a tiny manifest for audit
    man = {
        "n_tests_subset": int(len(sub)),
        "fdr_q": float(args.q),
        "features": list(args.features),
        "lags": [int(x) for x in args.lags],
        "results_csv": args.results_csv,
        "subset_csv": sub_out,
        "winners_csv": winners_out,
        "whitelist_yaml": yaml_out,
    }
    with open(os.path.join(args.out_dir, "analyze_manifest.json"), "w") as f:
        json.dump(man, f, indent=2)

    # Summary to stdout
    print(f"[subset] n={len(sub)} tests; winners @ q≤{args.q}: {len(winners)}")
    print(sub[["feature","lag_min","stat","p_value","q_sub"]].head(25).to_string(index=False))
    if len(winners):
        print("\n[Winners]")
        print(winners[["feature","lag_min","stat","p_value","q_sub"]].to_string(index=False))
        print(f"\nWhitelist YAML: {yaml_out}")
    else:
        print("\nNo winners at the specified q. Consider more data or alternative event definitions with guardrails.")


if __name__ == "__main__":
    main()

