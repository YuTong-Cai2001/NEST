"""Compare source-selected checkpoints without modifying experiment CSV files."""

import argparse
from pathlib import Path

import torch

from train_anomaly_gfm import AnomalyGFMTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='+')
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
        metrics = trainer.evaluate_zero_shot()
        print(
            f"CHECKPOINT_RESULT path={checkpoint_path} "
            f"auc={metrics.get('dynamic_auc', float('nan')):.8f} "
            f"ap={metrics.get('dynamic_ap', float('nan')):.8f}"
        )


if __name__ == '__main__':
    main()
