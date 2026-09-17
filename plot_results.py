#!/usr/bin/env python3
"""Generate paper-ready SPAD figures from reproducible CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(fig: plt.Figure, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def load_optional_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()


def plot_zero_shot_comparison(
    baselines: pd.DataFrame, zero_shot: pd.DataFrame, output_dir: Path
):
    comparison = baselines.copy()
    full = zero_shot[
        (zero_shot['prototype_mode'] == 'single')
        & (zero_shot['lambda_self'] > 0)
        & (zero_shot['lambda_proto'] > 0)
        & (zero_shot['lambda_reg'] > 0)
    ] if not zero_shot.empty else pd.DataFrame()

    # Prefer the final domain-invariant prototype-transport implementation
    # when legacy repaired runs coexist in the same CSV.
    if not full.empty and 'method_version' in full.columns:
        strengthened = full[full['method_version'].astype(str) == 'dipt']
        if strengthened.empty:
            strengthened = full[full['method_version'].astype(str) == 'di2']
        if not strengthened.empty:
            full = strengthened

    if not full.empty:
        comparison = comparison[
            ~comparison['method'].str.contains('SPAD', case=False, na=False)
        ]
        comparison = pd.concat([
            comparison,
            pd.DataFrame([{
                'method': 'SPAD',
                'auc': full['auc'].mean(),
                'ap': full['ap'].mean(),
                'source': 'repaired_runs',
            }]),
        ], ignore_index=True)

    if comparison.empty:
        return
    comparison = comparison.sort_values('auc', ascending=False).reset_index(drop=True)
    comparison.to_csv(
        output_dir / 'zero_shot_comparison_summary.csv',
        index=False,
        encoding='utf-8-sig',
    )

    x = np.arange(len(comparison))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(x, comparison['auc'], color='#4C78A8')
    ax.set_ylabel('ROC-AUC')
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison['method'], rotation=45, ha='right')
    for bar, value in zip(bars, comparison['auc']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=7,
        )
    ax.grid(axis='y', alpha=0.25)
    save_figure(fig, output_dir, 'zero_shot_auc_comparison')

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(comparison['auc'], comparison['ap'], s=55, color='#E45756')
    for _, row in comparison.iterrows():
        ax.annotate(
            row['method'],
            (row['auc'], row['ap']),
            xytext=(4, 4),
            textcoords='offset points',
            fontsize=7,
        )
    ax.set_xlabel('ROC-AUC')
    ax.set_ylabel('Average Precision')
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir, 'zero_shot_auc_ap_scatter')


def variant_label(row: pd.Series) -> str:
    if 'no_transport' in str(row.get('run_name', '')):
        return 'w/o transport residual'
    if 'no_local' in str(row.get('run_name', '')):
        return 'w/o local evolution'
    if row['prototype_mode'] == 'dual':
        return 'Dual prototype'
    if row['lambda_self'] == 0:
        return 'w/o temporal consistency'
    if row['lambda_proto'] == 0:
        return 'w/o prototype alignment'
    if row['lambda_reg'] == 0:
        return 'w/o representation reg.'
    return 'SPAD full'


def plot_ablation(zero_shot: pd.DataFrame, output_dir: Path):
    if zero_shot.empty:
        return
    data = zero_shot.copy()
    if 'method_version' in data.columns:
        strengthened = data[data['method_version'].astype(str) == 'dipt']
        if strengthened.empty:
            strengthened = data[data['method_version'].astype(str) == 'di2']
        if not strengthened.empty:
            data = strengthened
    data['variant'] = data.apply(variant_label, axis=1)
    summary = data.groupby('variant', as_index=False).agg(
        auc_mean=('auc', 'mean'),
        auc_std=('auc', 'std'),
        ap_mean=('ap', 'mean'),
        ap_std=('ap', 'std'),
        runs=('seed', 'count'),
    ).fillna(0)
    order = [
        'SPAD full',
        'w/o temporal consistency',
        'w/o prototype alignment',
        'w/o representation reg.',
        'w/o transport residual',
        'w/o local evolution',
        'Dual prototype',
    ]
    summary['order'] = summary['variant'].map({name: i for i, name in enumerate(order)})
    summary = summary.sort_values('order').drop(columns='order')
    summary.to_csv(
        output_dir / 'ablation_summary.csv', index=False, encoding='utf-8-sig'
    )

    x = np.arange(len(summary))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(
        x - width / 2, summary['auc_mean'], width,
        yerr=summary['auc_std'], capsize=3, label='ROC-AUC', color='#4C78A8'
    )
    ax.bar(
        x + width / 2, summary['ap_mean'], width,
        yerr=summary['ap_std'], capsize=3, label='AP', color='#F2CF5B'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary['variant'], rotation=20, ha='right')
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('Metric value')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    save_figure(fig, output_dir, 'spad_ablation')


def plot_few_shot(
    zero_shot: pd.DataFrame, few_shot: pd.DataFrame, output_dir: Path
):
    if few_shot.empty:
        return
    if 'method_version' in few_shot.columns:
        strengthened = few_shot[
            few_shot['method_version'].astype(str) == 'dipt'
        ]
        if strengthened.empty:
            strengthened = few_shot[
                few_shot['method_version'].astype(str) == 'di2'
            ]
        if not strengthened.empty:
            few_shot = strengthened
    if {'zero_auc_query', 'zero_ap_query'}.issubset(few_shot.columns):
        paired = few_shot[[
            'run_name', 'seed', 'zero_auc_query', 'zero_ap_query', 'auc', 'ap'
        ]].rename(columns={
            'zero_auc_query': 'zero_auc',
            'zero_ap_query': 'zero_ap',
            'auc': 'few_auc',
            'ap': 'few_ap',
        }).dropna(subset=['zero_auc', 'zero_ap'])
    else:
        if zero_shot.empty:
            return
        zero = zero_shot[['run_name', 'seed', 'auc', 'ap']].rename(
            columns={'auc': 'zero_auc', 'ap': 'zero_ap'}
        )
        few = few_shot[['run_name', 'seed', 'auc', 'ap']].rename(
            columns={'auc': 'few_auc', 'ap': 'few_ap'}
        )
        paired = zero.merge(few, on=['run_name', 'seed'], how='inner')
    if paired.empty:
        return
    paired.to_csv(
        output_dir / 'few_shot_paired_summary.csv',
        index=False,
        encoding='utf-8-sig',
    )

    x = np.arange(len(paired))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(x, paired['zero_auc'], 'o-', label='Zero-shot', color='#9D9D9D')
    axes[0].plot(x, paired['few_auc'], 'o-', label='Few-shot', color='#E45756')
    axes[0].set_title('ROC-AUC')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"seed {s}" for s in paired['seed']])
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, paired['zero_ap'], 'o-', label='Zero-shot', color='#9D9D9D')
    axes[1].plot(x, paired['few_ap'], 'o-', label='Few-shot', color='#E45756')
    axes[1].set_title('Average Precision')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"seed {s}" for s in paired['seed']])
    axes[1].grid(alpha=0.25)
    save_figure(fig, output_dir, 'few_shot_comparison')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zero_shot_csv', default='results/zero_shot_runs.csv')
    parser.add_argument('--few_shot_csv', default='results/few_shot_runs.csv')
    parser.add_argument(
        '--baseline_csv', default='results/paper_reported_baselines.csv'
    )
    parser.add_argument('--output_dir', default='figures/generated')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zero_shot = load_optional_csv(args.zero_shot_csv)
    few_shot = load_optional_csv(args.few_shot_csv)
    baselines = load_optional_csv(args.baseline_csv)
    plot_zero_shot_comparison(baselines, zero_shot, output_dir)
    plot_ablation(zero_shot, output_dir)
    plot_few_shot(zero_shot, few_shot, output_dir)
    print(f'Figures written to {output_dir.resolve()}')


if __name__ == '__main__':
    main()
