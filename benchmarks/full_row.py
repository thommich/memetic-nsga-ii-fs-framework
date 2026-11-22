"""Opt-in full-row benchmark; never run by the ordinary test suite."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import psutil
from memetic_fs import MemeticNSGA2Selector, OptimizationConfig, ResourceConfig

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_500_000)
    parser.add_argument("--features", type=int, default=250)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--path", help="Optional Parquet dataset with target/reference_month/applicant_id")
    parser.add_argument("--output", default="runs/benchmark")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    start = time.perf_counter()
    if args.path:
        frame = pd.read_parquet(args.path)
    else:
        rng = np.random.default_rng(42)
        # Fill in column blocks to avoid an additional float64 full matrix.
        values = np.empty((args.rows, args.features), dtype=np.float32)
        for column in range(args.features):
            values[:, column] = rng.standard_normal(args.rows, dtype=np.float32)
        frame = pd.DataFrame(values, columns=[f"f{i}" for i in range(args.features)], copy=False)
        probability = 1 / (1 + np.exp(-(values[:, 0] + values[:, min(1, args.features-1)])))
        frame["target"] = (rng.random(args.rows) < probability).astype(np.int8)
        months = pd.period_range("2024-01", periods=6, freq="M")
        codes = (np.arange(args.rows, dtype=np.int64) * 6 // args.rows).clip(max=5)
        frame["reference_month"] = np.asarray([int(str(m).replace("-", "")) for m in months])[codes]
        frame["applicant_id"] = np.arange(args.rows, dtype=np.int64)
    selector = MemeticNSGA2Selector(feature_range=(args.features, args.features),
        num_boost_round=args.rounds, use_rfe_initialization=False,
        optimization=OptimizationConfig(population_size=2, max_generations=1,
            reduced_generations=0, local_search_budget=0),
        resources=ResourceConfig(max_workers=1, threads_per_worker=args.threads),
        checkpoint_dir=args.output)
    selector.fit(frame, "target", id_col="applicant_id")
    result = {"rows": len(frame), "features": selector.n_features_in_, "rounds": args.rounds,
              "elapsed_seconds": time.perf_counter()-start,
              "parent_rss_bytes": psutil.Process().memory_info().rss,
              "reference_peak_private_bytes": selector.reference_["memory_peak_bytes"],
              "folds": selector.reference_["folds"], "versions": selector.evaluator_.versions}
    (Path(args.output) / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
