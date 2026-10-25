"""Aggregate NEST fixed-query runs separately by shot count and support ratio."""

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
        '--input', default='results/reproduced/few_shot_grid_runs.csv'
    )
    parser.add_argument(
        '--output', default='results/reproduced/few_shot_grid_mean_std.csv'
    )
    parser.add_argument('--baseline_output', default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open('r', newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'No rows found in {input_path}')

    grouped = defaultdict(list)
    for row in rows:
        ratio = int(row.get('normal_multiplier') or
                    int(row['num_normal']) // int(row['num_abnormal']))
        grouped[(int(row['shot']), ratio)].append(row)

    output_rows = []
    for shot, ratio in sorted(grouped):
        group = grouped[(shot, ratio)]
        seeds = sorted({int(row['seed']) for row in group})
        if len(seeds) != len(group):
            raise ValueError(f'Duplicate seed for shot={shot}, ratio={ratio}')
        zero_auc_mean, zero_auc_std = stats(group, 'zero_auc_query')
        few_auc_mean, few_auc_std = stats(group, 'auc')
        delta_auc_mean, delta_auc_std = stats(group, 'delta_auc')
        zero_ap_mean, zero_ap_std = stats(group, 'zero_ap_query')
        few_ap_mean, few_ap_std = stats(group, 'ap')
        delta_ap_mean, delta_ap_std = stats(group, 'delta_ap')
        output_rows.append({
            'shot': shot,
            'normal_multiplier': ratio,
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

    baseline_by_seed = {}
    for row in rows:
        value = (float(row['zero_auc_query']), float(row['zero_ap_query']),
                 int(row['query_samples']), int(row['query_positives']),
                 row.get('evaluation_pool_hash', row.get('pool_hash', '')))
        seed = int(row['seed'])
        if seed in baseline_by_seed and baseline_by_seed[seed] != value:
            raise ValueError(f'Query or zero-shot reference changes within seed {seed}')
        baseline_by_seed[seed] = value
    baseline = {
        'setting': 'zero_shot_fixed_query', 'num_seeds': len(baseline_by_seed),
        'seeds': '|'.join(str(seed) for seed in sorted(baseline_by_seed)),
        'auc_mean': float(np.mean([v[0] for v in baseline_by_seed.values()])),
        'auc_std': float(np.std([v[0] for v in baseline_by_seed.values()], ddof=0)),
        'ap_mean': float(np.mean([v[1] for v in baseline_by_seed.values()])),
        'ap_std': float(np.std([v[1] for v in baseline_by_seed.values()], ddof=0)),
    }
    baseline_path = (Path(args.baseline_output) if args.baseline_output else
                     output_path.with_name(output_path.stem + '_zero_shot.csv'))
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline))
        writer.writeheader()
        writer.writerow(baseline)
    print(f'Separate zero-shot summary written to {baseline_path}')


if __name__ == '__main__':
    main()
