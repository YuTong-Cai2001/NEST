"""Aggregate parameter-efficient SPAD adapter runs by shot count."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def stats(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    ddof = 1 if values.size > 1 else 0
    return float(values.mean()), float(values.std(ddof=ddof))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', default='results/few_shot_adapter_trial.csv'
    )
    parser.add_argument(
        '--output', default='results/few_shot_adapter_mean_std.csv'
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open('r', newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'No rows found in {input_path}')

    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row['shot'])].append(row)

    output_rows = []
    for shot in sorted(grouped):
        group = sorted(grouped[shot], key=lambda row: int(row['seed']))
        seeds = [int(row['seed']) for row in group]
        output = {
            'shot': shot,
            'num_abnormal': int(group[0]['num_abnormal']),
            'num_normal': int(group[0]['num_normal']),
            'num_seeds': len(seeds),
            'seeds': '|'.join(str(seed) for seed in seeds),
            'query_samples': int(group[0]['query_samples']),
            'query_positives': int(group[0]['query_positives']),
        }
        for prefix, source in (
            ('zero_auc', 'zero_auc_query'),
            ('adapter_auc', 'auc'),
            ('delta_auc', 'delta_auc'),
            ('zero_ap', 'zero_ap_query'),
            ('adapter_ap', 'ap'),
            ('delta_ap', 'delta_ap'),
        ):
            output[f'{prefix}_mean'], output[f'{prefix}_std'] = stats(
                group, source
            )
        output.update({
            'auc_improved_runs': sum(
                float(row['delta_auc']) > 0 for row in group
            ),
            'ap_improved_runs': sum(
                float(row['delta_ap']) > 0 for row in group
            ),
            'selected_epochs': '|'.join(
                f"seed{row['seed']}:{row['selected_epoch']}" for row in group
            ),
            'support_cv_gains': '|'.join(
                f"seed{row['seed']}:{float(row['support_cv_gain']):.6f}"
                for row in group
            ),
            'trainable_parameters': int(group[0]['trainable_parameters']),
            'evaluation_protocol': group[0]['evaluation_protocol'],
        })
        output_rows.append(output)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f'Few-shot adapter summary written to {output_path}')


if __name__ == '__main__':
    main()
