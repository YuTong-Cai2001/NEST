"""Aggregate fixed-query SPAD few-shot curve runs by shot count."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def stats(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', default='results/few_shot_curve_runs.csv'
    )
    parser.add_argument(
        '--output', default='results/few_shot_curve_mean_std.csv'
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
        group = grouped[shot]
        seeds = sorted({int(row['seed']) for row in group})
        zero_auc_mean, zero_auc_std = stats(group, 'zero_auc_query')
        few_auc_mean, few_auc_std = stats(group, 'auc')
        delta_auc_mean, delta_auc_std = stats(group, 'delta_auc')
        zero_ap_mean, zero_ap_std = stats(group, 'zero_ap_query')
        few_ap_mean, few_ap_std = stats(group, 'ap')
        delta_ap_mean, delta_ap_std = stats(group, 'delta_ap')
        output_rows.append({
            'shot': shot,
            'num_abnormal': int(group[0]['num_abnormal']),
            'num_normal': int(group[0]['num_normal']),
            'num_seeds': len(seeds),
            'seeds': '|'.join(str(seed) for seed in seeds),
            'query_samples': int(group[0]['query_samples']),
            'query_positives': int(group[0]['query_positives']),
            'zero_auc_mean': zero_auc_mean,
            'zero_auc_std': zero_auc_std,
            'few_auc_mean': few_auc_mean,
            'few_auc_std': few_auc_std,
            'delta_auc_mean': delta_auc_mean,
            'delta_auc_std': delta_auc_std,
            'auc_improved_runs': sum(float(row['delta_auc']) > 0 for row in group),
            'zero_ap_mean': zero_ap_mean,
            'zero_ap_std': zero_ap_std,
            'few_ap_mean': few_ap_mean,
            'few_ap_std': few_ap_std,
            'delta_ap_mean': delta_ap_mean,
            'delta_ap_std': delta_ap_std,
            'ap_improved_runs': sum(float(row['delta_ap']) > 0 for row in group),
            'selected_readouts': '|'.join(
                f"seed{row['seed']}:{row['selected_readout']}"
                for row in sorted(group, key=lambda item: int(item['seed']))
            ),
            'evaluation_protocol': group[0]['evaluation_protocol'],
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f'Few-shot curve summary written to {output_path}')


if __name__ == '__main__':
    main()
