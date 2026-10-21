"""Compare source-selected checkpoints without modifying experiment CSV files."""

import argparse
from pathlib import Path

import numpy as np
import torch

from train_anomaly_gfm import AnomalyGFMTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='+')
    parser.add_argument(
        '--yearwise-validation',
        action='store_true',
        help=(
            'Evaluate temporal prefixes ending before the final target year. '
            'This keeps the final year untouched while selecting a scoring rule.'
        ),
    )
    args = parser.parse_args()

    first = torch.load(args.checkpoints[0], map_location='cpu')
    config = dict(first['config'])
    config['output_dir'] = str(Path(args.checkpoints[0]).resolve().parents[1])
    trainer = AnomalyGFMTrainer(config)
    trainer.prepare_data(mode='dynamic')
    trainer.create_model()

    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        if args.yearwise_validation:
            original_test_data = trainer.test_data
            sequences = original_test_data['temporal_sequences'][0]
            adjacencies = original_test_data['temporal_adjacencies'][0]
            labels = original_test_data['temporal_labels'][0]
            start_year = int(config.get('start_year', 2018))
            validation_metrics = []
            diagnostic_metrics = {}
            try:
                # At least two steps are needed by the temporal-change diagnostics.
                # The last prefix is deliberately excluded: it is the held-out test year.
                for prefix_length in range(2, len(sequences)):
                    prefix_data = dict(original_test_data)
                    prefix_data['temporal_sequences'] = [sequences[:prefix_length]]
                    prefix_data['temporal_adjacencies'] = [adjacencies[:prefix_length]]
                    prefix_data['temporal_labels'] = [labels[:prefix_length]]
                    prefix_data['sequence_length'] = prefix_length
                    prefix_data['features'] = sequences[prefix_length - 1]
                    prefix_data['adjacency_matrix'] = adjacencies[prefix_length - 1]
                    prefix_data['labels'] = labels[prefix_length - 1]
                    trainer.test_data = prefix_data

                    metrics = trainer.evaluate_zero_shot()
                    auc = metrics.get('dynamic_auc', float('nan'))
                    ap = metrics.get('dynamic_ap', float('nan'))
                    year = start_year + prefix_length - 1
                    validation_metrics.append((auc, ap))
                    for name, values in metrics.get('diagnostics', {}).items():
                        diagnostic_metrics.setdefault(name, []).append(
                            (values['auc'], values['ap'])
                        )
                    print(
                        f"YEARWISE_VALIDATION path={checkpoint_path} year={year} "
                        f"auc={auc:.8f} ap={ap:.8f}"
                    )
            finally:
                trainer.test_data = original_test_data

            aucs = [item[0] for item in validation_metrics]
            aps = [item[1] for item in validation_metrics]
            print(
                f"VALIDATION_SUMMARY path={checkpoint_path} years={len(validation_metrics)} "
                f"mean_auc={np.nanmean(aucs):.8f} mean_ap={np.nanmean(aps):.8f}"
            )
            print('VALIDATION_DIAGNOSTIC_SUMMARY')
            for name, values in diagnostic_metrics.items():
                print(
                    f"  {name}: mean_auc={np.nanmean([v[0] for v in values]):.8f} "
                    f"mean_ap={np.nanmean([v[1] for v in values]):.8f}"
                )
        else:
            metrics = trainer.evaluate_zero_shot()
            print(
                f"CHECKPOINT_RESULT path={checkpoint_path} "
                f"auc={metrics.get('dynamic_auc', float('nan')):.8f} "
                f"ap={metrics.get('dynamic_ap', float('nan')):.8f}"
            )


if __name__ == '__main__':
    main()
