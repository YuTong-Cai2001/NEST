"""Validate the fixed-query and nested-support few-shot protocol outputs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', default='results/few_shot_curve_runs.csv'
    )
    args = parser.parse_args()

    with Path(args.input).open(
        'r', newline='', encoding='utf-8-sig'
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 9:
        raise AssertionError(f'Expected 9 rows, found {len(rows)}')

    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row['seed'])].append(row)
    if set(grouped) != {42, 43, 44}:
        raise AssertionError(f'Unexpected seeds: {sorted(grouped)}')

    for seed, seed_rows in grouped.items():
        seed_rows.sort(key=lambda row: int(row['shot']))
        if [int(row['shot']) for row in seed_rows] != [5, 10, 20]:
            raise AssertionError(f'Unexpected shots for seed {seed}')
        pool_hashes = {row['evaluation_pool_hash'] for row in seed_rows}
        query_sizes = {int(row['query_samples']) for row in seed_rows}
        query_positives = {int(row['query_positives']) for row in seed_rows}
        zero_auc = {float(row['zero_auc_query']) for row in seed_rows}
        zero_ap = {float(row['zero_ap_query']) for row in seed_rows}
        if len(pool_hashes) != 1 or query_sizes != {5012}:
            raise AssertionError(f'Fixed query mismatch for seed {seed}')
        if query_positives != {103} or len(zero_auc) != 1 or len(zero_ap) != 1:
            raise AssertionError(f'Fixed baseline mismatch for seed {seed}')

        artifacts = []
        for row in seed_rows:
            artifact_path = Path(row['checkpoint'])
            if not artifact_path.exists():
                raise FileNotFoundError(artifact_path)
            artifact = torch.load(
                artifact_path, map_location='cpu', weights_only=False
            )
            if len(artifact['support_node_ids']) != (
                int(row['num_abnormal']) + int(row['num_normal'])
            ):
                raise AssertionError(f'Support count mismatch: {artifact_path}')
            if not np.array_equal(
                artifact['evaluation_pool_ids'],
                artifacts[0]['evaluation_pool_ids'] if artifacts else
                artifact['evaluation_pool_ids'],
            ):
                raise AssertionError(f'Evaluation pool mismatch for seed {seed}')
            artifacts.append(artifact)

        support_sets = [set(item['support_node_ids'].tolist()) for item in artifacts]
        if not support_sets[0] < support_sets[1] < support_sets[2]:
            raise AssertionError(f'Support sets are not strictly nested for seed {seed}')

    print(
        'Validation passed: 9 runs; seeds 42/43/44; shots 5/10/20; '
        'fixed 5,012-node query with 103 positives; nested supports verified.'
    )


if __name__ == '__main__':
    main()
