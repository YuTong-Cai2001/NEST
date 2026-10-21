"""Parameter-efficient target-domain few-shot adaptation for SPAD."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from few_shot_finetune import (
    FewShotFinetuner,
    FewShotSupportSet,
    set_global_seed,
    upsert_csv_row,
)


class ResidualPrototypeAdapter(nn.Module):
    """Small residual adapter with a bounded target-score correction."""

    def __init__(
        self, representation_dim: int, bottleneck_dim: int = 8,
        max_residual_weight: float = 1.0,
    ):
        super().__init__()
        self.down = nn.Linear(representation_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, representation_dim, bias=False)
        self.max_residual_weight = float(max_residual_weight)
        self.residual_weight_raw = nn.Parameter(torch.tensor(-1.3862944))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    @property
    def residual_weight(self) -> torch.Tensor:
        return self.max_residual_weight * torch.sigmoid(
            self.residual_weight_raw
        )

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(representations)))
        return F.normalize(representations + residual, p=2, dim=-1)


class AnchoredAdapterTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.seed = int(config['seed'])
        self.bottleneck_dim = int(config.get('bottleneck_dim', 8))
        self.learning_rate = float(config.get('learning_rate', 3e-3))
        self.weight_decay = float(config.get('weight_decay', 1e-4))
        self.max_epochs = int(config.get('max_epochs', 250))
        self.patience = int(config.get('patience', 40))
        self.cv_folds = int(config.get('cv_folds', 5))
        self.base_logit_scale = float(config.get('base_logit_scale', 3.0))
        self.max_residual_weight = float(
            config.get('max_residual_weight', 1.0)
        )
        self.normal_shrinkage_tau = float(
            config.get('normal_shrinkage_tau', 50.0)
        )
        self.anchor_sample_size = int(config.get('anchor_sample_size', 4096))

    @staticmethod
    def ranking_metric(labels, scores):
        return 0.5 * (
            roc_auc_score(labels, scores)
            + average_precision_score(labels, scores)
        )

    @staticmethod
    def _hash(node_ids):
        values = np.asarray(node_ids, dtype=np.int64)
        return hashlib.sha256(values.tobytes()).hexdigest()[:16]

    def _make_adapter(self, dim, seed):
        set_global_seed(seed)
        return ResidualPrototypeAdapter(
            representation_dim=dim,
            bottleneck_dim=self.bottleneck_dim,
            max_residual_weight=self.max_residual_weight,
        ).to(self.device)

    def _support_views(self, temporal_representations, node_ids, labels):
        self.temporal_view_count = int(temporal_representations.shape[0])
        ids = torch.as_tensor(node_ids, dtype=torch.long, device=self.device)
        views = temporal_representations[:, ids]
        views = views.permute(1, 0, 2).reshape(-1, views.shape[-1])
        node_labels = torch.as_tensor(
            np.asarray(labels)[node_ids], dtype=torch.long, device=self.device
        )
        view_labels = node_labels.repeat_interleave(
            temporal_representations.shape[0]
        )
        return views, view_labels

    def _prototypes(
        self, adapter, train_views, train_labels, normal_prior,
    ):
        adapted = adapter(train_views)
        support_normal = F.normalize(
            adapted[train_labels == 0].mean(dim=0), p=2, dim=0
        )
        anomaly = F.normalize(
            adapted[train_labels == 1].mean(dim=0), p=2, dim=0
        )
        normal_count = (
            int((train_labels == 0).sum().item())
            / max(getattr(self, 'temporal_view_count', 1), 1)
        )
        rho = normal_count / (normal_count + self.normal_shrinkage_tau)
        normal = F.normalize(
            (1 - rho) * normal_prior + rho * support_normal,
            p=2, dim=0,
        )
        return normal, anomaly, adapted

    def _score(
        self, adapter, representations, base_scores,
        normal_prototype, anomaly_prototype,
    ):
        adapted = adapter(representations)
        margin = (
            adapted @ anomaly_prototype - adapted @ normal_prototype
        )
        base_logits = self.base_logit_scale * (base_scores - 0.5)
        logits = base_logits + adapter.residual_weight * margin
        return logits, margin, adapted

    @staticmethod
    def _focal_loss(logits, labels, gamma=2.0):
        targets = labels.float()
        probabilities = torch.sigmoid(logits)
        binary_ce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.where(targets > 0.5, probabilities, 1 - probabilities)
        class_weight = torch.where(
            targets > 0.5,
            torch.full_like(targets, 0.75),
            torch.full_like(targets, 0.25),
        )
        return (class_weight * (1 - pt).pow(gamma) * binary_ce).mean()

    @staticmethod
    def _pairwise_ranking_loss(logits, labels, temperature=0.25):
        positive = logits[labels == 1]
        negative = logits[labels == 0]
        differences = positive[:, None] - negative[None, :]
        return F.softplus(-differences / temperature).mean()

    def _training_loss(
        self, adapter, train_views, train_labels, train_base_scores,
        normal_prior, anchor_representations,
    ):
        normal, anomaly, adapted = self._prototypes(
            adapter, train_views, train_labels, normal_prior
        )
        margin = adapted @ anomaly - adapted @ normal
        logits = (
            self.base_logit_scale * (train_base_scores - 0.5)
            + adapter.residual_weight * margin
        )
        ranking_loss = self._pairwise_ranking_loss(logits, train_labels)
        focal_loss = self._focal_loss(logits, train_labels)
        prototype_logits = torch.stack([
            adapted @ normal, adapted @ anomaly,
        ], dim=1) / 0.2
        prototype_loss = F.cross_entropy(prototype_logits, train_labels)

        original_anchor = F.normalize(
            anchor_representations, p=2, dim=-1
        )
        adapted_anchor = adapter(anchor_representations)
        representation_anchor = (
            1 - (adapted_anchor * original_anchor).sum(dim=-1)
        ).mean()
        score_anchor = 0.05 * adapter.residual_weight.pow(2)
        anchor_loss = representation_anchor + score_anchor
        total = (
            ranking_loss
            + 0.5 * focal_loss
            + 0.25 * prototype_loss
            + anchor_loss
        )
        return total

    def _train_fold(
        self, temporal_representations, latest_representations,
        base_scores, labels, train_ids, validation_ids,
        normal_prior, fold_seed, anchor_ids,
    ):
        dim = temporal_representations.shape[-1]
        adapter = self._make_adapter(dim, fold_seed)
        optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        train_views, train_labels = self._support_views(
            temporal_representations, train_ids, labels
        )
        train_node_base = torch.as_tensor(
            base_scores[train_ids], dtype=torch.float32, device=self.device
        )
        train_base = train_node_base.repeat_interleave(
            temporal_representations.shape[0]
        )
        validation_tensor = latest_representations[
            torch.as_tensor(validation_ids, dtype=torch.long, device=self.device)
        ]
        validation_base = torch.as_tensor(
            base_scores[validation_ids], dtype=torch.float32,
            device=self.device,
        )
        validation_labels = np.asarray(labels)[validation_ids].astype(int)
        anchor_representations = latest_representations[
            torch.as_tensor(anchor_ids, dtype=torch.long, device=self.device)
        ]
        base_metric = self.ranking_metric(
            validation_labels, base_scores[validation_ids]
        )

        best_metric = -np.inf
        best_epoch = 1
        epochs_without_gain = 0
        for epoch in range(1, self.max_epochs + 1):
            adapter.train()
            optimizer.zero_grad()
            loss = self._training_loss(
                adapter, train_views, train_labels, train_base,
                normal_prior, anchor_representations,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()

            adapter.eval()
            with torch.no_grad():
                normal, anomaly, _ = self._prototypes(
                    adapter, train_views, train_labels, normal_prior
                )
                validation_logits, _, _ = self._score(
                    adapter, validation_tensor, validation_base,
                    normal, anomaly,
                )
                metric = self.ranking_metric(
                    validation_labels,
                    validation_logits.detach().cpu().numpy(),
                )
            if metric > best_metric + 1e-8:
                best_metric = metric
                best_epoch = epoch
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
            if epochs_without_gain >= self.patience:
                break
        return best_epoch, float(best_metric), float(base_metric)

    def select_epoch(
        self, temporal_representations, latest_representations,
        base_scores, labels, support_ids, normal_prior,
    ):
        support_labels = np.asarray(labels)[support_ids].astype(int)
        splitter = StratifiedKFold(
            n_splits=min(
                self.cv_folds,
                int(np.bincount(support_labels, minlength=2).min()),
            ),
            shuffle=True,
            random_state=self.seed,
        )
        rng = np.random.default_rng(self.seed + 1000)
        population = np.arange(latest_representations.shape[0])
        anchor_ids = rng.choice(
            population,
            size=min(self.anchor_sample_size, len(population)),
            replace=False,
        )
        fold_results = []
        for fold_index, (train_index, validation_index) in enumerate(
            splitter.split(support_ids, support_labels)
        ):
            fold_results.append(self._train_fold(
                temporal_representations,
                latest_representations,
                base_scores,
                labels,
                support_ids[train_index],
                support_ids[validation_index],
                normal_prior,
                self.seed + fold_index + 1,
                anchor_ids,
            ))
        best_epochs = [item[0] for item in fold_results]
        selected_epoch = max(1, int(round(float(np.median(best_epochs)))))
        return {
            'selected_epoch': selected_epoch,
            'fold_best_epochs': best_epochs,
            'cv_metric': float(np.mean([item[1] for item in fold_results])),
            'base_cv_metric': float(np.mean([item[2] for item in fold_results])),
        }, anchor_ids

    def fit_final(
        self, temporal_representations, latest_representations,
        base_scores, labels, support_ids, normal_prior,
        selected_epoch, anchor_ids,
    ):
        adapter = self._make_adapter(
            temporal_representations.shape[-1], self.seed
        )
        optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        train_views, train_labels = self._support_views(
            temporal_representations, support_ids, labels
        )
        train_node_base = torch.as_tensor(
            base_scores[support_ids], dtype=torch.float32, device=self.device
        )
        train_base = train_node_base.repeat_interleave(
            temporal_representations.shape[0]
        )
        anchor_representations = latest_representations[
            torch.as_tensor(anchor_ids, dtype=torch.long, device=self.device)
        ]
        for _ in range(selected_epoch):
            adapter.train()
            optimizer.zero_grad()
            loss = self._training_loss(
                adapter, train_views, train_labels, train_base,
                normal_prior, anchor_representations,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()

        adapter.eval()
        with torch.no_grad():
            normal, anomaly, _ = self._prototypes(
                adapter, train_views, train_labels, normal_prior
            )
            all_base = torch.as_tensor(
                base_scores, dtype=torch.float32, device=self.device
            )
            logits, margin, _ = self._score(
                adapter, latest_representations, all_base,
                normal, anomaly,
            )
        return {
            'adapter': adapter,
            'normal_prototype': normal,
            'anomaly_prototype': anomaly,
            'scores': logits.detach().cpu().numpy(),
            'margin': margin.detach().cpu().numpy(),
            'residual_weight': float(
                adapter.residual_weight.detach().cpu().item()
            ),
            'trainable_parameters': sum(
                parameter.numel() for parameter in adapter.parameters()
            ),
        }


def nested_support(pool_ids, labels, num_abnormal, num_normal):
    pool_labels = np.asarray(labels)[pool_ids]
    abnormal = pool_ids[pool_labels == 1][:num_abnormal]
    normal = pool_ids[pool_labels == 0][:num_normal]
    if len(abnormal) != num_abnormal or len(normal) != num_normal:
        raise ValueError('Insufficient fixed-pool nodes for requested support')
    return np.concatenate([abnormal, normal])


def main():
    parser = argparse.ArgumentParser(
        description='SPAD anchored parameter-efficient few-shot adaptation'
    )
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--normal_multiplier', type=int, default=5)
    parser.add_argument('--max_support_abnormal', type=int, default=20)
    parser.add_argument('--max_support_normal', type=int, default=100)
    parser.add_argument('--start_year', type=int, default=2018)
    parser.add_argument('--end_year', type=int, default=2022)
    parser.add_argument('--bottleneck_dim', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--max_epochs', type=int, default=250)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--results_csv', default='results/few_shot_adapter_trial.csv')
    parser.add_argument('--figraph_path', default='data/FiGraph-main/data/')
    args = parser.parse_args()

    num_normal = args.shot * args.normal_multiplier
    if args.shot > args.max_support_abnormal:
        raise ValueError('shot exceeds max_support_abnormal')
    if num_normal > args.max_support_normal:
        raise ValueError('normal support exceeds max_support_normal')

    finetuner = FewShotFinetuner({
        'model_path': args.model_path,
        'start_year': args.start_year,
        'end_year': args.end_year,
        'num_abnormal': args.shot,
        'num_normal': num_normal,
        'seed': args.seed,
        'figraph_path': args.figraph_path,
        'adaptation_mode': 'score_fusion',
    })
    model = finetuner.load_pretrained_model()
    for parameter in model.parameters():
        parameter.requires_grad = False
    test_data, labels = finetuner.load_test_data()
    output = finetuner._full_graph_forward(model, test_data)
    base_scores = finetuner._percentile_rank(
        output['recency_neighbor_inconsistency'].detach().cpu().numpy()
    )
    temporal_representations = output['fused_sequence'][0].detach()
    latest_representations = temporal_representations[-1]
    normal_prior = output['consensus_prototype'][0].detach()

    pool_builder = FewShotSupportSet(
        num_abnormal=args.max_support_abnormal,
        num_normal=args.max_support_normal,
        seed=args.seed,
    )
    pool_ids, _ = pool_builder.build_support_set(test_data, labels)
    support_ids = nested_support(
        pool_ids, labels, args.shot, num_normal
    )

    zero_results = finetuner.evaluate_score_array(
        base_scores, labels, pool_ids,
        evaluation_name='Adapter-Trial-ZeroShot-FixedQuery',
    )
    trainer = AnchoredAdapterTrainer(vars(args))
    selection, anchor_ids = trainer.select_epoch(
        temporal_representations,
        latest_representations,
        base_scores,
        labels,
        support_ids,
        normal_prior,
    )
    fitted = trainer.fit_final(
        temporal_representations,
        latest_representations,
        base_scores,
        labels,
        support_ids,
        normal_prior,
        selection['selected_epoch'],
        anchor_ids,
    )
    adapted_results = finetuner.evaluate_score_array(
        fitted['scores'], labels, pool_ids,
        evaluation_name='Adapter-Trial-FewShot-FixedQuery',
    )

    source_path = Path(args.model_path)
    artifact_path = source_path.with_name(
        f'{source_path.stem}_fewshot_adapter_{args.shot}a_{num_normal}n.pth'
    )
    torch.save({
        'source_checkpoint': str(source_path),
        'adapter_state_dict': fitted['adapter'].state_dict(),
        'normal_prototype': fitted['normal_prototype'].cpu(),
        'anomaly_prototype': fitted['anomaly_prototype'].cpu(),
        'support_node_ids': support_ids,
        'evaluation_pool_ids': pool_ids,
        'selection': selection,
        'config': vars(args),
        'zero_shot_query_results': zero_results,
        'adapted_query_results': adapted_results,
    }, artifact_path)

    run_name = source_path.resolve().parents[1].name
    row = {
        'run_name': run_name,
        'stage': 'few_shot_adapter_trial',
        'seed': args.seed,
        'shot': args.shot,
        'num_abnormal': args.shot,
        'num_normal': num_normal,
        'max_support_abnormal': args.max_support_abnormal,
        'max_support_normal': args.max_support_normal,
        'query_samples': len(adapted_results['labels']),
        'query_positives': int(np.asarray(adapted_results['labels']).sum()),
        'zero_auc_query': zero_results['auc'],
        'auc': adapted_results['auc'],
        'delta_auc': adapted_results['auc'] - zero_results['auc'],
        'zero_ap_query': zero_results['ap'],
        'ap': adapted_results['ap'],
        'delta_ap': adapted_results['ap'] - zero_results['ap'],
        'selected_epoch': selection['selected_epoch'],
        'fold_best_epochs': '|'.join(
            str(value) for value in selection['fold_best_epochs']
        ),
        'support_cv_metric': selection['cv_metric'],
        'base_cv_metric': selection['base_cv_metric'],
        'support_cv_gain': (
            selection['cv_metric'] - selection['base_cv_metric']
        ),
        'residual_weight': fitted['residual_weight'],
        'trainable_parameters': fitted['trainable_parameters'],
        'bottleneck_dim': args.bottleneck_dim,
        'learning_rate': args.learning_rate,
        'support_hash': trainer._hash(support_ids),
        'evaluation_pool_hash': trainer._hash(pool_ids),
        'evaluation_protocol': (
            'fixed max-pool-excluded query; five-fold support-only epoch '
            'selection; frozen SPAD backbone; query labels evaluation only'
        ),
        'checkpoint': str(artifact_path),
    }
    upsert_csv_row(
        args.results_csv,
        row,
        key_fields=('run_name', 'stage', 'seed', 'shot'),
    )
    print('=' * 68)
    print(
        f"AUC: {zero_results['auc']:.6f} -> {adapted_results['auc']:.6f} "
        f"({row['delta_auc']:+.6f})"
    )
    print(
        f"AP:  {zero_results['ap']:.6f} -> {adapted_results['ap']:.6f} "
        f"({row['delta_ap']:+.6f})"
    )
    print(
        f"Support CV: {selection['base_cv_metric']:.6f} -> "
        f"{selection['cv_metric']:.6f}; epoch={selection['selected_epoch']}; "
        f"trainable={fitted['trainable_parameters']}"
    )
    print('=' * 68)


if __name__ == '__main__':
    main()
