"""Run a fixed-query, nested-support few-shot curve for SPAD."""

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from few_shot_finetune import (
    FewShotFinetuner,
    FewShotSupportSet,
    upsert_csv_row,
)


def parse_shots(value: str) -> list[int]:
    shots = sorted({int(item.strip()) for item in value.split(',') if item.strip()})
    if not shots or shots[0] < 2:
        raise ValueError('shots must contain integers >= 2')
    return shots


def short_hash(node_ids: Iterable[int]) -> str:
    values = np.asarray(list(node_ids), dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def nested_support(
    pool_node_ids: np.ndarray,
    labels: np.ndarray,
    num_abnormal: int,
    num_normal: int,
) -> np.ndarray:
    pool_labels = np.asarray(labels)[pool_node_ids]
    abnormal = pool_node_ids[pool_labels == 1]
    normal = pool_node_ids[pool_labels == 0]
    if len(abnormal) < num_abnormal or len(normal) < num_normal:
        raise ValueError(
            f'Fixed support pool has {len(abnormal)} abnormal and {len(normal)} '
            f'normal nodes, but {num_abnormal}/{num_normal} were requested.'
        )
    return np.concatenate([
        abnormal[:num_abnormal],
        normal[:num_normal],
    ])


def main():
    parser = argparse.ArgumentParser(
        description='SPAD fixed-query nested-support few-shot curve'
    )
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--shots', default='5,10,20')
    parser.add_argument('--normal_multiplier', type=int, default=5)
    parser.add_argument('--max_support_abnormal', type=int, default=20)
    parser.add_argument('--max_support_normal', type=int, default=100)
    parser.add_argument(
        '--min_adaptation_abnormal', type=int, default=10,
        help=(
            'Minimum abnormal support count required to fit a correction; '
            'smaller settings retain the zero-shot ranking'
        ),
    )
    parser.add_argument('--start_year', type=int, default=2018)
    parser.add_argument('--end_year', type=int, default=2022)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--min_cv_gain', type=float, default=0.05)
    parser.add_argument(
        '--results_csv', default='results/few_shot_curve_runs.csv'
    )
    parser.add_argument(
        '--figraph_path', default='data/FiGraph-main/data/'
    )
    args = parser.parse_args()

    shots = parse_shots(args.shots)
    if max(shots) > args.max_support_abnormal:
        raise ValueError('Largest shot exceeds max_support_abnormal')
    if max(shots) * args.normal_multiplier > args.max_support_normal:
        raise ValueError('Largest normal support exceeds max_support_normal')

    config = {
        'model_path': args.model_path,
        'start_year': args.start_year,
        'end_year': args.end_year,
        'num_abnormal': max(shots),
        'num_normal': max(shots) * args.normal_multiplier,
        'adaptation_mode': 'score_fusion',
        'cv_folds': args.cv_folds,
        'min_cv_gain': args.min_cv_gain,
        'seed': args.seed,
        'results_csv': args.results_csv,
        'figraph_path': args.figraph_path,
    }
    finetuner = FewShotFinetuner(config)

    print('=' * 68)
    print(' SPAD Fixed-Query Few-Shot Curve')
    print(f'  checkpoint: {args.model_path}')
    print(f'  seed: {args.seed}')
    print(f'  shots: {shots}')
    print(
        f'  fixed support pool: {args.max_support_abnormal} abnormal + '
        f'{args.max_support_normal} normal'
    )
    print('=' * 68)

    model = finetuner.load_pretrained_model()
    for parameter in model.parameters():
        parameter.requires_grad = False
    test_data, labels = finetuner.load_test_data()
    if test_data['feature_dim'] != finetuner.model_config['feature_dim']:
        raise ValueError(
            f"Aligned feature dimension {test_data['feature_dim']} does not "
            f"match checkpoint dimension {finetuner.model_config['feature_dim']}"
        )

    pool_builder = FewShotSupportSet(
        num_abnormal=args.max_support_abnormal,
        num_normal=args.max_support_normal,
        seed=args.seed,
    )
    evaluation_pool_ids, pool_strategy = pool_builder.build_support_set(
        test_data, labels, last_time_idx=-1
    )
    pool_hash = short_hash(evaluation_pool_ids)

    score_features, feature_names, representations = (
        finetuner.extract_score_features(model, test_data)
    )
    zero_results = finetuner.evaluate_score_array(
        score_features[:, 0], labels, evaluation_pool_ids,
        evaluation_name='Few-Shot-Curve-ZeroShot-FixedQuery',
    )

    checkpoint = finetuner.checkpoint
    run_name = Path(args.model_path).resolve().parents[1].name
    for shot in shots:
        num_normal = shot * args.normal_multiplier
        support_node_ids = nested_support(
            evaluation_pool_ids, labels, shot, num_normal
        )
        if shot < args.min_adaptation_abnormal:
            adapted_scores = score_features[:, 0].copy()
            adaptation = {
                'name': 'zero_shot_recency',
                'indices': [0],
                'representation': None,
                'C': None,
                'alpha': 0.0,
                'cv_metric': None,
                'base_cv_metric': None,
                'cv_gain': 0.0,
                'coef': [1.0],
                'intercept': 0.0,
                'selected_features': [feature_names[0]],
                'fallback_reason': (
                    f'num_abnormal<{args.min_adaptation_abnormal}'
                ),
            }
        else:
            adapted_scores, adaptation = finetuner.fit_score_fusion(
                score_features, feature_names, representations,
                labels, support_node_ids,
            )
        results = finetuner.evaluate_score_array(
            adapted_scores, labels, evaluation_pool_ids,
            evaluation_name=(
                f'Few-Shot-Curve-{shot}A-{num_normal}N-FixedQuery'
            ),
        )

        source_path = Path(args.model_path)
        artifact_path = source_path.with_name(
            f'{source_path.stem}_fewshot_curve_{shot}a_{num_normal}n.pth'
        )
        torch.save({
            'source_checkpoint': str(source_path),
            'support_node_ids': support_node_ids,
            'evaluation_pool_ids': evaluation_pool_ids,
            'support_pool_strategy': pool_strategy,
            'support_pool_hash': pool_hash,
            'support_hash': short_hash(support_node_ids),
            'adaptation_mode': 'score_fusion',
            'adaptation': adaptation,
            'score_feature_names': feature_names,
            'zero_shot_query_results': zero_results,
            'results': results,
        }, artifact_path)

        row = {
            'run_name': run_name,
            'stage': 'few_shot_curve',
            'method_version': checkpoint.get('config', {}).get(
                'method_version', 'dipt'
            ),
            'adaptation_mode': 'score_fusion',
            'seed': args.seed,
            'shot': shot,
            'num_abnormal': shot,
            'num_normal': num_normal,
            'normal_multiplier': args.normal_multiplier,
            'max_support_abnormal': args.max_support_abnormal,
            'max_support_normal': args.max_support_normal,
            'cv_folds': args.cv_folds,
            'min_cv_gain': args.min_cv_gain,
            'min_adaptation_abnormal': args.min_adaptation_abnormal,
            'adaptation_eligible': shot >= args.min_adaptation_abnormal,
            'start_year': args.start_year,
            'end_year': args.end_year,
            'zero_auc_query': zero_results['auc'],
            'zero_ap_query': zero_results['ap'],
            'auc': results['auc'],
            'ap': results['ap'],
            'delta_auc': results['auc'] - zero_results['auc'],
            'delta_ap': results['ap'] - zero_results['ap'],
            'query_samples': len(results['labels']),
            'query_positives': int(np.asarray(results['labels']).sum()),
            'selected_readout': adaptation['name'],
            'selected_features': '|'.join(adaptation['selected_features']),
            'selected_c': adaptation['C'],
            'selected_alpha': adaptation['alpha'],
            'support_cv_metric': adaptation['cv_metric'],
            'support_cv_gain': adaptation['cv_gain'],
            'fallback_reason': adaptation.get('fallback_reason', ''),
            'support_hash': short_hash(support_node_ids),
            'evaluation_pool_hash': pool_hash,
            'evaluation_protocol': (
                'nested support; fixed max-pool-excluded labeled query; '
                'support-only model selection'
            ),
            'checkpoint': str(artifact_path),
        }
        upsert_csv_row(
            args.results_csv,
            row,
            key_fields=(
                'run_name', 'stage', 'adaptation_mode', 'seed',
                'shot', 'max_support_abnormal', 'max_support_normal',
            ),
        )
        print(
            f"shot={shot}: AUC {zero_results['auc']:.6f} -> "
            f"{results['auc']:.6f}; AP {zero_results['ap']:.6f} -> "
            f"{results['ap']:.6f}; readout={adaptation['name']}"
        )


if __name__ == '__main__':
    main()
