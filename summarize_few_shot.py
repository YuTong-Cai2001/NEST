"""Create a paper-ready paired summary of SPAD few-shot runs."""

import argparse
import csv
from pathlib import Path

import numpy as np


def load_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def population_stats(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="results/few_shot_final_runs.csv"
    )
    parser.add_argument(
        "--output", default="results/few_shot_mean_std.csv"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_rows(input_path)
    seeds = sorted({int(row["seed"]) for row in rows})

    summary = []
    for metric, zero_field, few_field, delta_field in (
        ("ROC-AUC", "zero_auc_query", "auc", "delta_auc"),
        ("AP", "zero_ap_query", "ap", "delta_ap"),
    ):
        zero_mean, zero_std = population_stats(
            [float(row[zero_field]) for row in rows]
        )
        few_mean, few_std = population_stats(
            [float(row[few_field]) for row in rows]
        )
        delta_mean, delta_std = population_stats(
            [float(row[delta_field]) for row in rows]
        )
        summary.append({
            "metric": metric,
            "zero_shot_query_mean": zero_mean,
            "zero_shot_query_std": zero_std,
            "few_shot_mean": few_mean,
            "few_shot_std": few_std,
            "mean_delta": delta_mean,
            "delta_std": delta_std,
            "num_seeds": len(seeds),
            "seeds": "|".join(str(seed) for seed in seeds),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Few-shot summary written to {output_path}")


if __name__ == "__main__":
    main()
