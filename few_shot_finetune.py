import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import argparse
import csv
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import AnomalyGFMDataProcessor
from anomaly_gfm_complete_model import AnomalyGFMModel, create_model_config
from sparse_graph_utils import SparseGraphProcessor
import scipy.sparse as sp


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def upsert_csv_row(path: str, row: Dict, key_fields=('run_name', 'stage', 'seed')):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if output_path.exists():
        with output_path.open('r', newline='', encoding='utf-8-sig') as handle:
            existing_rows = list(csv.DictReader(handle))
    existing_rows = [
        old for old in existing_rows
        if not all(str(old.get(key, '')) == str(row.get(key, '')) for key in key_fields)
    ]
    fieldnames = list(row.keys())
    for old in existing_rows:
        for field in old:
            if field not in fieldnames:
                fieldnames.append(field)
    with output_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows + [row])


class FewShotSupportSet:
    """Few-shot Support Set Builder"""
    
    def __init__(
        self, num_abnormal: int = 10, num_normal: int = 50, seed: int = 42,
    ):
        """
        Args:
            num_abnormal: Number of abnormal nodes
            num_normal: Number of normal nodes
        """
        self.num_abnormal = num_abnormal
        self.num_normal = num_normal
        self.seed = int(seed)
    
    def build_support_set(self, temporal_data: Dict, labels: np.ndarray, 
                          last_time_idx: int = -1) -> Tuple[np.ndarray, str]:
        """
        Build support set
        
        Args:
            temporal_data: Temporal data dictionary
            labels: Node labels for the last time window (num_nodes,)
            last_time_idx: Index of the last time window (default -1)
            
        Returns:
            support_node_ids: Support set node ID array
            strategy: Strategy name used
        """
        print(f"\n Building few-shot support set...")
        print(f"   - Target: {self.num_abnormal} abnormal nodes + {self.num_normal} normal nodes")
        
        # Get labels for the last time window
        if isinstance(labels, list):
            labels = labels[last_time_idx] if last_time_idx < len(labels) else labels[-1]
        
        labels = np.array(labels).flatten()
        
        # Separate abnormal and normal nodes
        abnormal_nodes = np.where(labels == 1)[0]
        normal_nodes = np.where(labels == 0)[0]
        
        num_anomaly_available = len(abnormal_nodes)
        num_normal_available = len(normal_nodes)
        
        
        rng = np.random.default_rng(self.seed)
        if num_anomaly_available >= self.num_abnormal:
            rng.shuffle(abnormal_nodes)
            rng.shuffle(normal_nodes)
            support_abnormal = abnormal_nodes[:self.num_abnormal]
            support_normal = normal_nodes[:min(self.num_normal, num_normal_available)]
            support_node_ids = np.concatenate([support_abnormal, support_normal])
            strategy = "mixed"
        elif num_anomaly_available >= 5:
            rng.shuffle(normal_nodes)
            support_abnormal = abnormal_nodes  
            support_normal = normal_nodes[:self.num_normal]
            support_node_ids = np.concatenate([support_abnormal, support_normal])
            strategy = "mixed_all_abnormal"
        else:
            rng.shuffle(normal_nodes)
            support_node_ids = normal_nodes[:self.num_normal]
            strategy = "normal_only"
        
        print(f"    Support set built:")
        support_labels = labels[support_node_ids]
        print(f"      - Abnormal nodes: {int((support_labels == 1).sum())}")
        print(f"      - Normal nodes: {int((support_labels == 0).sum())}")
        print(f"      - Total support set size: {len(support_node_ids)} nodes")
        
        return support_node_ids, strategy


def few_shot_collate_fn(batch):
    """
    Custom collate function: Handle sparse matrices in FewShotDataset
    
    Args:
        batch: List of (temporal_tensor, node_adjacencies, node_label, node_id)
    
    Returns:
        temporal_tensors: (batch_size, sequence_length, feature_dim)
        adjacency_lists: List[List[scipy.sparse.csr_matrix]] - One list per sample
        labels_batch: (batch_size, 1)
        node_ids: List[int]
    """
    if not batch:
        return [], [], torch.empty(0), []
    
    temporal_tensors = []
    adjacency_lists = []
    labels_list = []
    node_ids = []
    
    for item in batch:
        temporal_tensor, node_adjacencies, node_label, node_id = item
        temporal_tensors.append(temporal_tensor)
        adjacency_lists.append(node_adjacencies)  
        labels_list.append(node_label)
        node_ids.append(node_id)
    
    # Stack tensors
    temporal_batch = torch.stack(temporal_tensors)  # (batch_size, sequence_length, feature_dim)
    labels_batch = torch.stack(labels_list)  # (batch_size, 1)
    
    return temporal_batch, adjacency_lists, labels_batch, node_ids


class FewShotDataset(Dataset):
    """Few-shot fine-tuning dataset"""
    
    def __init__(self, temporal_data: Dict, support_node_ids: np.ndarray, 
                 labels: np.ndarray, sequence_length: int = 5):
        """
        Args:
            temporal_data: Temporal data dictionary
            support_node_ids: Support set node IDs
            labels: Node labels (last time window)
            sequence_length: Temporal sequence length (5 time windows)
        """
        self.temporal_sequences = temporal_data['temporal_sequences'][0]  
        self.temporal_adjacencies = temporal_data['temporal_adjacencies'][0]
        self.support_node_ids = support_node_ids
        self.labels = np.array(labels).flatten() if isinstance(labels, list) else labels.flatten()
        self.sequence_length = min(sequence_length, len(self.temporal_sequences))
        max_node_id = len(self.labels) - 1
        self.support_node_ids = self.support_node_ids[self.support_node_ids <= max_node_id]
        
    
    def __len__(self):
        return len(self.support_node_ids)
    
    def __getitem__(self, idx):
        node_id = self.support_node_ids[idx]
        label = self.labels[node_id]
        
        node_features = []
        node_adjacencies = []
        
        for t in range(self.sequence_length):
            if t < len(self.temporal_sequences):
                # Node features (feature_dim,)
                node_feat = self.temporal_sequences[t][node_id]
                node_features.append(node_feat)
                
                # Adjacency matrix (maintain graph integrity)
                adj = self.temporal_adjacencies[t]
                node_adjacencies.append(adj)

        # Temporal features: (sequence_length, feature_dim)
        temporal_tensor = torch.FloatTensor(np.stack(node_features))
        
        # Label
        node_label = torch.FloatTensor([label])
        
        return temporal_tensor, node_adjacencies, node_label, node_id


class FewShotFinetuner:
    """Few-shot fine-tuner"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = int(config.get('seed', 42))
        set_global_seed(self.seed)
        
        self.model_path = config.get('model_path')
        if not self.model_path:
            raise ValueError('model_path is required')
        
        # Fine-tuning parameters
        self.num_abnormal = config.get('num_abnormal', 10)
        self.num_normal = config.get('num_normal', 50)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.num_epochs = config.get('num_epochs', 20)
        self.batch_size = config.get('batch_size', 32)
        self.lambda_fs = float(config.get('lambda_fs', 0.5))
        self.max_neighbors_per_node = int(config.get('max_neighbors_per_node', 100))
        self.adaptation_mode = config.get('adaptation_mode', 'score_fusion')
        self.cv_folds = int(config.get('cv_folds', 5))
        self.min_cv_gain = float(config.get('min_cv_gain', 0.05))
        self.results_csv = config.get('results_csv', 'results/few_shot_runs.csv')
        
        # Data paths
        self.figraph_path = config.get('figraph_path', 'data/FiGraph-main/data/')
        self.start_year = config.get('start_year', 2018)
        self.end_year = config.get('end_year', 2022)
        
        # Sparse graph processor
        self.sparse_processor = SparseGraphProcessor()
    
    def load_pretrained_model(self) -> AnomalyGFMModel:
        """Load pre-trained model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        model_config = checkpoint.get('resolved_model_config')
        if model_config is None:
            raise ValueError(
                'Checkpoint does not contain resolved_model_config. '
                'Please use a checkpoint produced by the repaired zero-shot script.'
            )
        model = AnomalyGFMModel(model_config)

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Zero-shot checkpoints wrap the model in SparseGraphModelAdapter.
        if any(key.startswith('base_model.') for key in state_dict):
            state_dict = {
                key[len('base_model.'):]: value
                for key, value in state_dict.items()
                if key.startswith('base_model.')
            }

        incompatible = model.load_state_dict(state_dict, strict=False)
        loaded_keys = len(model.state_dict()) - len(incompatible.missing_keys)
        load_ratio = loaded_keys / max(len(model.state_dict()), 1)
        if load_ratio < 0.8:
            raise RuntimeError(
                f'Only {load_ratio:.1%} of model tensors were restored; '
                f'missing={incompatible.missing_keys[:10]}'
            )
        self.checkpoint = checkpoint
        self.model_config = model_config
        
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode initially
        
        print(f"    Model loaded successfully")
        return model
    
    def freeze_and_unfreeze_parameters(self, model: AnomalyGFMModel):
        """
        Freeze and unfreeze parameters
        """
        

        for param in model.parameters():
            param.requires_grad = False

        fusion_network = model.fusion_network.fusion_network
        
        # Unfreeze Layer 2 (indices 3-5)
        for i in range(3, 6):
            if i < len(fusion_network):
                for param in fusion_network[i].parameters():
                    param.requires_grad = True
                print(f"    Unfreeze fusion network Layer 2: {type(fusion_network[i]).__name__}")
        
        # Unfreeze Layer 3 (indices 6-8)
        for i in range(6, len(fusion_network)):
            for param in fusion_network[i].parameters():
                param.requires_grad = True
            print(f"    Unfreeze fusion network Layer 3: {type(fusion_network[i]).__name__}")
        

        model.normal_prototype.requires_grad = True
        model.score_scale_raw.requires_grad = True
        model.score_threshold.requires_grad = True
        model.prototype_score_weight_raw.requires_grad = True
        model.temporal_score_weight_raw.requires_grad = True
        model.temporal_score_threshold.requires_grad = True
        if model.abnormal_prototype is not None:
            model.abnormal_prototype.requires_grad = False
        print(f"    Unfreeze normal prototype parameters (single-prototype fine-tuning)")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n    Parameter statistics:")
        print(f"      - Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"      - Frozen parameters: {total_params - trainable_params:,} ({(total_params-trainable_params)/total_params*100:.2f}%)")

    @staticmethod
    def _latest_snapshot_local_score(
        fused_sequence: torch.Tensor, edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Incident-neighbour cosine distance on the final snapshot."""
        node_state = F.normalize(fused_sequence[:, -1], p=2, dim=-1)
        batch_size, num_nodes, _ = node_state.shape
        scores = node_state.new_zeros(batch_size, num_nodes)
        if edge_index.numel() == 0:
            return scores
        source, target = edge_index[0], edge_index[1]
        valid = (
            (source >= 0) & (source < num_nodes)
            & (target >= 0) & (target < num_nodes)
        )
        source, target = source[valid], target[valid]
        for batch_idx in range(batch_size):
            state = node_state[batch_idx]
            neighbour_sum = torch.zeros_like(state)
            degree = state.new_zeros(num_nodes)
            neighbour_sum.index_add_(0, target, state[source])
            neighbour_sum.index_add_(0, source, state[target])
            ones = torch.ones_like(source, dtype=state.dtype)
            degree.index_add_(0, target, ones)
            degree.index_add_(0, source, ones)
            observed = degree > 0
            neighbour_mean = neighbour_sum / degree.clamp_min(1).unsqueeze(-1)
            local_distance = 1 - F.cosine_similarity(
                state, neighbour_mean, dim=-1
            )
            scores[batch_idx, observed] = local_distance[observed]
        return scores

    def _full_graph_forward(
        self, model: AnomalyGFMModel, test_data: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Run one deterministic full-target-graph forward pass."""
        model.eval()
        sequence_length = min(
            self.model_config['sequence_length'],
            len(test_data['temporal_sequences'][0]),
        )
        features = test_data['temporal_sequences'][0][-sequence_length:]
        adjacencies = test_data['temporal_adjacencies'][0][-sequence_length:]
        temporal_tensor = torch.as_tensor(
            np.stack(features), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        adjacency_info = []
        for adjacency in adjacencies:
            edge_index, edge_attr = self.sparse_processor.sparse_to_pyg_format(
                adjacency
            )
            adjacency_info.append({
                'edge_index': edge_index.to(self.device),
                'edge_attr': (
                    edge_attr.to(self.device) if edge_attr is not None else None
                ),
                'num_nodes': temporal_tensor.shape[2],
            })
        with torch.no_grad():
            output = model(
                temporal_tensor, adjacency_info, temporal_tensor[:, -1]
            )
            output['latest_snapshot_local'] = self._latest_snapshot_local_score(
                output['fused_sequence'], adjacency_info[-1]['edge_index']
            )
        return output

    @staticmethod
    def _percentile_rank(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
        order = np.argsort(values, kind='stable')
        ranks = np.empty(values.shape[0], dtype=np.float64)
        ranks[order] = np.arange(values.shape[0], dtype=np.float64)
        return ranks / max(values.shape[0] - 1, 1)

    def extract_score_features(
        self, model: AnomalyGFMModel, test_data: Dict,
    ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
        """Extract frozen SPAD scores and representations for adaptation."""
        output = self._full_graph_forward(model, test_data)
        feature_tensors = [
            ('recency_dynamic_local', output['recency_neighbor_inconsistency']),
            ('uniform_dynamic_local', output['dynamic_neighbor_inconsistency']),
            ('latest_snapshot_local', output['latest_snapshot_local']),
            ('prototype_transport_residual', output['prototype_transport_residual']),
            ('transported_prototype_distance', output['anomaly_distances']),
        ]
        names = [name for name, _ in feature_tensors]
        columns = []
        for _, tensor in feature_tensors:
            values = tensor.detach().cpu().numpy().reshape(-1)
            columns.append(self._percentile_rank(values))
        features = np.column_stack(columns)
        if not np.isfinite(features).all():
            raise RuntimeError('Non-finite score features after percentile ranking')
        representations = {
            'mean_fused': output['fused_representations'][0]
                .detach().cpu().numpy(),
            'latest_fused': output['fused_sequence'][0, -1]
                .detach().cpu().numpy(),
            'temporal_change': (
                output['fused_sequence'][0, -1]
                - output['fused_sequence'][0, 0]
            ).detach().cpu().numpy(),
        }
        return features, names, representations

    def _prototype_rank(
        self, representations: np.ndarray, labels: np.ndarray,
        train_indices: np.ndarray,
    ) -> np.ndarray:
        """Rank nodes by their cosine margin to target support prototypes."""
        vectors = np.asarray(representations, dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
        train_labels = np.asarray(labels)[train_indices].astype(int)
        positive = vectors[train_indices][train_labels == 1]
        negative = vectors[train_indices][train_labels == 0]
        positive_prototype = positive.mean(axis=0)
        negative_prototype = negative.mean(axis=0)
        positive_prototype /= max(np.linalg.norm(positive_prototype), 1e-12)
        negative_prototype /= max(np.linalg.norm(negative_prototype), 1e-12)
        margin = (
            vectors @ positive_prototype - vectors @ negative_prototype
        )
        return self._percentile_rank(margin)

    @staticmethod
    def _ranking_metric(labels: np.ndarray, scores: np.ndarray) -> float:
        return 0.5 * (
            roc_auc_score(labels, scores)
            + average_precision_score(labels, scores)
        )

    def fit_score_fusion(
        self, features: np.ndarray, feature_names: List[str],
        representations: Dict[str, np.ndarray], labels: np.ndarray,
        support_node_ids: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """Fit a conservative support-only few-shot score readout.

        The zero-shot ranking remains an explicit anchor.  Target labels may
        only add a small correction, selected by stratified support-set CV.
        This prevents ten positive examples from replacing a strong global
        ranking with an unstable target-only classifier.
        """
        support_labels = np.asarray(labels)[support_node_ids].astype(int)
        support_features = features[support_node_ids]
        class_counts = np.bincount(support_labels, minlength=2)
        folds = min(self.cv_folds, int(class_counts.min()))
        if folds < 2:
            raise ValueError('At least two support examples per class are required')

        feature_sets = {
            'local_temporal': [0, 1, 2],
            'local_plus_transport': [0, 1, 2, 3],
            'all_scores': list(range(features.shape[1])),
        }
        c_values = (0.05, 0.2, 1.0, 5.0)
        alpha_values = (0.05, 0.10, 0.20, 0.35)
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.seed
        )

        base_fold_scores = []
        for _, val_idx in splitter.split(support_features, support_labels):
            base_fold_scores.append(self._ranking_metric(
                support_labels[val_idx], support_features[val_idx, 0]
            ))
        base_cv = float(np.mean(base_fold_scores))
        best = {
            'name': 'zero_shot_recency',
            'indices': [0],
            'representation': None,
            'C': None,
            'alpha': 0.0,
            'cv_metric': base_cv,
        }

        for set_name, indices in feature_sets.items():
            for c_value in c_values:
                for alpha in alpha_values:
                    fold_scores = []
                    for train_idx, val_idx in splitter.split(
                        support_features, support_labels
                    ):
                        classifier = LogisticRegression(
                            C=c_value,
                            class_weight='balanced',
                            solver='liblinear',
                            max_iter=2000,
                            random_state=self.seed,
                        )
                        classifier.fit(
                            support_features[train_idx][:, indices],
                            support_labels[train_idx],
                        )
                        calibrated = classifier.predict_proba(
                            support_features[val_idx][:, indices]
                        )[:, 1]
                        blended = (
                            (1 - alpha) * support_features[val_idx, 0]
                            + alpha * calibrated
                        )
                        fold_scores.append(self._ranking_metric(
                            support_labels[val_idx], blended
                        ))
                    cv_metric = float(np.mean(fold_scores))
                    if cv_metric > best['cv_metric'] + 1e-12:
                        best = {
                            'name': set_name,
                            'indices': indices,
                            'representation': None,
                            'C': c_value,
                            'alpha': alpha,
                            'cv_metric': cv_metric,
                        }

        # A target-domain positive/negative prototype margin is the natural
        # few-shot extension of SPAD's source normal prototype.  Prototypes are
        # rebuilt inside every fold, so validation labels never enter fitting.
        for representation_name, representation in representations.items():
            for alpha in alpha_values:
                fold_scores = []
                for train_idx, val_idx in splitter.split(
                    support_features, support_labels
                ):
                    train_node_ids = support_node_ids[train_idx]
                    prototype_score = self._prototype_rank(
                        representation, labels, train_node_ids
                    )
                    blended = (
                        (1 - alpha) * support_features[val_idx, 0]
                        + alpha * prototype_score[support_node_ids[val_idx]]
                    )
                    fold_scores.append(self._ranking_metric(
                        support_labels[val_idx], blended
                    ))
                cv_metric = float(np.mean(fold_scores))
                if cv_metric > best['cv_metric'] + 1e-12:
                    best = {
                        'name': 'support_prototype',
                        'indices': [0],
                        'representation': representation_name,
                        'C': None,
                        'alpha': alpha,
                        'cv_metric': cv_metric,
                    }

        if best['cv_metric'] < base_cv + self.min_cv_gain:
            best = {
                'name': 'zero_shot_recency', 'indices': [0], 'C': None,
                'representation': None, 'alpha': 0.0,
                'cv_metric': base_cv,
            }

        if best['name'] == 'zero_shot_recency':
            final_scores = features[:, 0].copy()
            best.update({
                'coef': [1.0], 'intercept': 0.0,
                'selected_features': [feature_names[0]],
            })
        elif best['name'] == 'support_prototype':
            prototype_score = self._prototype_rank(
                representations[best['representation']], labels,
                support_node_ids,
            )
            final_scores = (
                (1 - best['alpha']) * features[:, 0]
                + best['alpha'] * prototype_score
            )
            best.update({
                'coef': [], 'intercept': 0.0,
                'selected_features': [
                    feature_names[0],
                    f"support_{best['representation']}_prototype_margin",
                ],
            })
        else:
            classifier = LogisticRegression(
                C=best['C'],
                class_weight='balanced',
                solver='liblinear',
                max_iter=2000,
                random_state=self.seed,
            )
            classifier.fit(
                support_features[:, best['indices']], support_labels
            )
            calibrated = classifier.predict_proba(
                features[:, best['indices']]
            )[:, 1]
            final_scores = (
                (1 - best['alpha']) * features[:, 0]
                + best['alpha'] * calibrated
            )
            best.update({
                'coef': classifier.coef_[0].tolist(),
                'intercept': float(classifier.intercept_[0]),
                'selected_features': [
                    feature_names[index] for index in best['indices']
                ],
            })
        best['base_cv_metric'] = base_cv
        best['cv_gain'] = best['cv_metric'] - base_cv
        return final_scores, best

    def evaluate_score_array(
        self, scores: np.ndarray, labels: np.ndarray,
        support_node_ids: np.ndarray, evaluation_name: str,
    ) -> Dict:
        valid_mask = np.isin(labels, [0, 1])
        valid_mask[np.asarray(support_node_ids, dtype=np.int64)] = False
        query_labels = np.asarray(labels)[valid_mask].astype(int)
        query_scores = np.asarray(scores)[valid_mask]
        if np.unique(query_labels).size != 2:
            raise ValueError('Query set does not contain both valid classes')
        eval_results = AnomalyGFMDataProcessor.evaluate_ranking_performance(
            query_labels, query_scores, evaluation_name
        )
        return {
            'auc': eval_results['roc_auc'],
            'ap': eval_results['average_precision'],
            'scores': query_scores,
            'labels': query_labels,
            'eval_results': eval_results,
        }
    
    def load_test_data(self) -> Tuple[Dict, np.ndarray]:
        """Load test data (FiGraph 2018-2022)"""
        

        data_config = {
            'temporal_window': 7,
            'target_dim': self.model_config['feature_dim'],
            'use_svd': False,
            'alignment_mode': self.checkpoint.get('config', {}).get(
                'alignment_mode', 'invariant'
            ),
            'figraph_path': self.figraph_path,
            'cache_dir': 'cache/',
            'enable_cache': True,
            'seed': self.seed,
        }
        
        processor = AnomalyGFMDataProcessor(data_config)
        
        # Load FiGraph dynamic graph sequence
        figraph_temporal = processor.load_figraph_temporal_sequence(
            start_year=self.start_year,
            end_year=self.end_year
        )

        aligner_path = self.checkpoint.get('feature_aligner_path')
        if not aligner_path or not os.path.exists(aligner_path):
            candidate = Path(self.model_path).resolve().parents[1] / 'feature_aligner.pkl'
            aligner_path = str(candidate)
        if not os.path.exists(aligner_path):
            raise FileNotFoundError(
                f'Feature aligner not found: {aligner_path}. '
                'Run zero-shot training before few-shot fine-tuning.'
            )
        processor.load_feature_aligner(aligner_path)
        for batch_idx, sequence in enumerate(figraph_temporal['temporal_sequences']):
            for time_idx, features in enumerate(sequence):
                figraph_temporal['temporal_sequences'][batch_idx][time_idx] = (
                    processor.transform_target_features(features)
                )
            figraph_temporal['temporal_sequences'][batch_idx] = (
                processor.append_structural_temporal_features(
                    figraph_temporal['temporal_sequences'][batch_idx],
                    figraph_temporal['temporal_adjacencies'][batch_idx],
                )
            )
        figraph_temporal['feature_dim'] = (
            figraph_temporal['temporal_sequences'][0][0].shape[1]
        )
        
        if figraph_temporal.get('temporal_labels'):
            labels = figraph_temporal['temporal_labels'][0][-1]  # Last time window
        else:
            num_nodes = figraph_temporal['num_nodes']
            labels = np.zeros(num_nodes, dtype=np.float32)
            print(f"    Warning: Labels not found, using zero labels")
        
        print(f"    Data loaded successfully:")
        print(f"      - Sequence length: {figraph_temporal['sequence_length']} years")
        print(f"      - Number of nodes: {figraph_temporal['num_nodes']:,}")
        print(f"      - Feature dimension: {figraph_temporal['feature_dim']}")
        print(f"      - Abnormal nodes: {(labels == 1).sum():,} ({(labels == 1).mean()*100:.2f}%)")
        
        return figraph_temporal, labels
    
    def _legacy_finetune_unused(self, model: AnomalyGFMModel, train_loader: DataLoader):
        """Deprecated node-wise implementation retained only for old checkpoints."""
        print(f"\n Starting fine-tuning training...")
        
        # Set model to training mode
        model.train()
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.learning_rate, weight_decay=1e-5)
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                try:
                    optimizer.zero_grad()
                    
                    # Unpack batch data
                    temporal_tensors, adjacency_lists, labels_batch, node_ids = batch_data
                    
                    # Move to device
                    temporal_tensors = temporal_tensors.to(self.device)
                    labels_batch = labels_batch.to(self.device).squeeze()
                    
                    if hasattr(model, 'use_feature_projection') and model.use_feature_projection:
                        batch_size, seq_len, actual_feat_dim = temporal_tensors.shape
                        temporal_tensors_flat = temporal_tensors.view(-1, actual_feat_dim)  # (batch_size * seq_len, actual_feat_dim)
                        temporal_tensors_proj = model.feature_projection(temporal_tensors_flat)  # (batch_size * seq_len, expected_feat_dim)
                        temporal_tensors = temporal_tensors_proj.view(batch_size, seq_len, -1)  # (batch_size, seq_len, expected_feat_dim)
                    
                    batch_size, seq_len, feature_dim = temporal_tensors.shape
                    temporal_batch = temporal_tensors.unsqueeze(2)  # (batch_size, seq_len, 1, feature_dim)
                    
                    # Handle adjacency matrices (maintain graph integrity)
                    adj_info = []
                    for t in range(seq_len):
                        if t < len(adjacency_lists[0]):
                            adj = adjacency_lists[0][t]  # scipy sparse matrix
                            # Convert to model-expected format (using sparse processor)
                            graph_info = self.sparse_processor.zero_shot_graph_processing(
                                adj, batch_size=1
                            )
                            sparse_adj_info = {
                                'edge_index': graph_info['edge_index'].to(self.device),
                                'edge_attr': graph_info['edge_attr'].to(self.device) if graph_info['edge_attr'] is not None else None,
                                'num_nodes': adj.shape[0]
                            }
                            adj_info.append(sparse_adj_info)
                    
                    # Get current time window node features (for GNN)
                    current_features = temporal_tensors[:, -1, :].unsqueeze(1)  # (batch_size, 1, feature_dim)
                    
                    # Forward pass
                    output = model(temporal_batch, adj_info, current_features)
                    
                    # Get fused representations and anomaly scores
                    fused_representations = output['fused_representations']  # (batch_size, 1, output_dim)
                    anomaly_scores = output['anomaly_scores']  # (batch_size, 1)
                    
                    # Compute loss
                    anomaly_scores_flat = anomaly_scores.squeeze()  # (batch_size,)
                    labels_flat = labels_batch  # (batch_size,)
                    
                    anomaly_probs = torch.sigmoid(anomaly_scores_flat)
                    
                    # Binary cross-entropy loss
                    loss = F.binary_cross_entropy(anomaly_probs, labels_flat)
                    
                    # Single prototype center loss: only use normal prototype, consistent with zero-shot phase
                    normal_proto = model.normal_prototype
                    
                    # Compute similarity to normal prototype
                    representations_flat = fused_representations.squeeze(1)  # (batch_size, output_dim)
                    representations_norm = F.normalize(representations_flat, p=2, dim=1)
                    normal_proto_norm = F.normalize(normal_proto, p=2, dim=1)
                    sim_to_normal = torch.mm(representations_norm, normal_proto_norm.t()).squeeze(1)
                    
                    normal_mask = (labels_flat == 0)
                    
                    proto_loss = torch.tensor(0.0, device=self.device)
                    if normal_mask.sum() > 0:
                        proto_loss = (1 - sim_to_normal[normal_mask]).mean()
                    
                    # Total loss
                    total_loss = loss + 0.5 * proto_loss
                    
                    # Backward pass
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"    Batch {batch_idx} processing failed: {e}")
                    continue
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"   Epoch {epoch+1}/{self.num_epochs}: Loss = {avg_loss:.6f}")
        
        print(f"    Fine-tuning training completed")
        return model

    def _build_support_subgraph(
        self, test_data: Dict, support_node_ids: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[sp.csr_matrix]]:
        """Build one deterministic 1-hop subgraph around all support nodes."""
        adj_last = test_data['temporal_adjacencies'][0][-1].tocsr()
        selected = set(int(node_id) for node_id in support_node_ids)
        rng = np.random.default_rng(self.seed)
        for node_id in support_node_ids:
            start, end = adj_last.indptr[node_id], adj_last.indptr[node_id + 1]
            neighbors = adj_last.indices[start:end]
            if len(neighbors) > self.max_neighbors_per_node:
                neighbors = rng.choice(
                    neighbors, self.max_neighbors_per_node, replace=False
                )
            selected.update(int(value) for value in neighbors)

        subgraph_nodes = np.asarray(sorted(selected), dtype=np.int64)
        temporal_features = [
            features[subgraph_nodes]
            for features in test_data['temporal_sequences'][0]
        ]
        temporal_adjacencies = [
            adjacency[subgraph_nodes][:, subgraph_nodes].tocsr()
            for adjacency in test_data['temporal_adjacencies'][0]
        ]
        return subgraph_nodes, temporal_features, temporal_adjacencies

    def finetune_subgraph(
        self, model: AnomalyGFMModel, test_data: Dict,
        labels: np.ndarray, support_node_ids: np.ndarray,
    ) -> AnomalyGFMModel:
        """Fine-tune on a support-centered graph while preserving neighborhoods."""
        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            trainable_params, lr=self.learning_rate, weight_decay=1e-5
        )

        nodes, feature_sequence, adjacency_sequence = self._build_support_subgraph(
            test_data, support_node_ids
        )
        sequence_length = min(
            self.model_config['sequence_length'], len(feature_sequence)
        )
        feature_sequence = feature_sequence[-sequence_length:]
        adjacency_sequence = adjacency_sequence[-sequence_length:]
        temporal_tensor = torch.as_tensor(
            np.stack(feature_sequence), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        adjacency_info = []
        for adjacency in adjacency_sequence:
            edge_index, edge_attr = self.sparse_processor.sparse_to_pyg_format(adjacency)
            adjacency_info.append({
                'edge_index': edge_index.to(self.device),
                'edge_attr': edge_attr.to(self.device) if edge_attr is not None else None,
                'num_nodes': len(nodes),
            })

        local_index = {node_id: idx for idx, node_id in enumerate(nodes.tolist())}
        support_positions = torch.as_tensor(
            [local_index[int(node_id)] for node_id in support_node_ids],
            dtype=torch.long,
            device=self.device,
        )
        support_labels = torch.as_tensor(
            labels[support_node_ids], dtype=torch.float32, device=self.device
        )

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            output = model(
                temporal_tensor, adjacency_info, temporal_tensor[:, -1]
            )
            logits = output['anomaly_logits'][0, support_positions]
            detection_loss = F.binary_cross_entropy_with_logits(
                logits, support_labels
            )
            similarities = output['sim_to_normal'][0, support_positions]
            normal_mask = support_labels == 0
            prototype_loss = (
                (1 - similarities[normal_mask]).mean()
                if normal_mask.any() else logits.new_tensor(0.0)
            )
            total_loss = detection_loss + self.lambda_fs * prototype_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == self.num_epochs:
                print(
                    f"   Epoch {epoch + 1}/{self.num_epochs}: "
                    f"total={total_loss.item():.6f}, "
                    f"det={detection_loss.item():.6f}, "
                    f"proto={prototype_loss.item():.6f}"
                )
        return model

    def evaluate_full_graph(
        self, model: AnomalyGFMModel, test_data: Dict,
        labels: np.ndarray, support_node_ids: np.ndarray,
        evaluation_name: str = 'Few-Shot-Finetuned-QuerySet',
    ) -> Dict:
        """Evaluate the query set in one structurally correct full-graph pass."""
        model.eval()
        sequence_length = min(
            self.model_config['sequence_length'],
            len(test_data['temporal_sequences'][0]),
        )
        features = test_data['temporal_sequences'][0][-sequence_length:]
        adjacencies = test_data['temporal_adjacencies'][0][-sequence_length:]
        temporal_tensor = torch.as_tensor(
            np.stack(features), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        adjacency_info = []
        for adjacency in adjacencies:
            edge_index, edge_attr = self.sparse_processor.sparse_to_pyg_format(adjacency)
            adjacency_info.append({
                'edge_index': edge_index.to(self.device),
                'edge_attr': edge_attr.to(self.device) if edge_attr is not None else None,
                'num_nodes': temporal_tensor.shape[2],
            })

        with torch.no_grad():
            output = model(
                temporal_tensor, adjacency_info, temporal_tensor[:, -1]
            )
            scores = output['anomaly_scores'][0].detach().cpu().numpy()

        valid_mask = np.isin(labels, [0, 1])
        valid_mask[np.asarray(support_node_ids, dtype=np.int64)] = False
        query_labels = np.asarray(labels)[valid_mask]
        query_scores = scores[valid_mask]
        if len(np.unique(query_labels)) < 2:
            raise ValueError('Query set does not contain both valid classes')
        eval_results = AnomalyGFMDataProcessor.evaluate_ranking_performance(
            query_labels, query_scores, evaluation_name
        )
        return {
            'auc': eval_results['roc_auc'],
            'ap': eval_results['average_precision'],
            'scores': query_scores,
            'labels': query_labels,
            'eval_results': eval_results,
        }
    
    def _legacy_evaluate_unused(self, model: AnomalyGFMModel, test_data: Dict, labels: np.ndarray,
                 support_node_ids: np.ndarray):
        """
        Evaluate model performance (query set)
        """
        
        model.eval()
        
        all_node_ids = np.arange(len(labels))
        query_node_ids = np.setdiff1d(all_node_ids, support_node_ids)
        
        print(f"   - Support set nodes: {len(support_node_ids)}")
        print(f"   - Query set nodes: {len(query_node_ids):,}")
        
        # Evaluate query set
        query_labels = labels[query_node_ids]
        query_scores = []
        
        # Preprocess adjacency matrices (all nodes share graph structure)
        adj_list = test_data['temporal_adjacencies'][0]
        adj_info = []
        for t in range(len(adj_list)):
            adj = adj_list[t]
            graph_info = self.sparse_processor.zero_shot_graph_processing(
                adj, batch_size=1
            )
            sparse_adj_info = {
                'edge_index': graph_info['edge_index'].to(self.device),
                'edge_attr': graph_info['edge_attr'].to(self.device) if graph_info['edge_attr'] is not None else None,
                'num_nodes': adj.shape[0]
            }
            adj_info.append(sparse_adj_info)
        
        with torch.no_grad():
            # Batch process query set
            batch_size = 100
            for i in tqdm(range(0, len(query_node_ids), batch_size), desc="Evaluating query set"):
                batch_node_ids = query_node_ids[i:i+batch_size]
                
                # Build batch data
                batch_temporal_list = []
                for node_id in batch_node_ids:
                    node_features = []
                    for t in range(len(test_data['temporal_sequences'][0])):
                        node_feat = test_data['temporal_sequences'][0][t][node_id]
                        node_features.append(node_feat)
                    
                    temporal_tensor = torch.FloatTensor(np.stack(node_features)).unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, actual_feature_dim)
                    batch_temporal_list.append(temporal_tensor)
                
                # Merge batches
                temporal_batch = torch.cat(batch_temporal_list, dim=0).to(self.device)  # (batch_size, seq_len, 1, actual_feature_dim)
                
                if hasattr(model, 'use_feature_projection') and model.use_feature_projection:
                    batch_size, seq_len, num_nodes, actual_feat_dim = temporal_batch.shape
                    temporal_batch_flat = temporal_batch.view(-1, actual_feat_dim)  # (batch_size * seq_len * num_nodes, actual_feat_dim)
                    temporal_batch_proj = model.feature_projection(temporal_batch_flat)  # (batch_size * seq_len * num_nodes, expected_feat_dim)
                    temporal_batch = temporal_batch_proj.view(batch_size, seq_len, num_nodes, -1)  # (batch_size, seq_len, num_nodes, expected_feat_dim)
                
                current_features = temporal_batch[:, -1, :, :]  # (batch_size, num_nodes, expected_feature_dim)
                
                # Forward pass
                output = model(temporal_batch, adj_info, current_features)
                
                if 'margin_scores' in output:
                    scores = output['margin_scores'].squeeze().cpu().numpy()
                elif 'anomaly_scores' in output:
                    scores = output['anomaly_scores'].squeeze().cpu().numpy()
                elif 'fused_representations' in output and 'normal_prototype' in output:
                    fused_representations = output['fused_representations']  # (batch_size, num_nodes, output_dim)
                    normal_prototype = output['normal_prototype']  # (1, output_dim)
                    
                    # Compute anomaly scores: distance from normal prototype
                    batch_size_curr, num_nodes_curr, feature_dim_curr = fused_representations.shape
                    representations_flat = fused_representations.view(-1, feature_dim_curr)
                    
                    # Normalize
                    representations_norm = F.normalize(representations_flat, p=2, dim=1)
                    normal_prototype_norm = F.normalize(normal_prototype, p=2, dim=1)
                    
                    # Compute similarity
                    sim_to_normal = torch.mm(representations_norm, normal_prototype_norm.t()).squeeze(1)
                    
                    # Use temperature-adjusted distance calculation (refer to training code)
                    temperature = getattr(model, 'score_temperature', torch.tensor(0.5)).item()
                    temp_adjusted_sim = sim_to_normal / temperature
                    scores = (1 - torch.sigmoid(temp_adjusted_sim)).cpu().numpy()
                    scores = scores.flatten()
                else:
                    raise ValueError(f"Model output format not supported, available keys: {list(output.keys())}")
                
                query_scores.extend(scores)
        
        query_scores = np.array(query_scores)
        
        if (query_labels == 1).sum() > 0 and (query_labels == 0).sum() > 0:
            eval_results = AnomalyGFMDataProcessor.evaluate_ranking_performance(
                query_labels, query_scores, "Few-Shot-Finetuned-QuerySet"
            )
            
            return {
                'auc': eval_results['roc_auc'],
                'ap': eval_results['average_precision'],
                'scores': query_scores,
                'labels': query_labels,
                'eval_results': eval_results  # Contains complete evaluation results
            }
        else:
            print(f"    Query set labels are imbalanced, cannot calculate AUC/AP")
            return {
                'auc': 0.0, 
                'ap': 0.0, 
                'scores': query_scores, 
                'labels': query_labels,
                'eval_results': {}
            }
    
    def run(self):
        """Run the selected few-shot adaptation workflow."""
        print("=" * 60)
        print(" SPAD Few-Shot Adaptation")
        print("=" * 60)
        
        # Load pre-trained model
        model = self.load_pretrained_model()
        
        if self.adaptation_mode == 'network':
            self.freeze_and_unfreeze_parameters(model)
        elif self.adaptation_mode == 'score_fusion':
            for parameter in model.parameters():
                parameter.requires_grad = False
            print("    Encoder and representation model frozen")
        else:
            raise ValueError(
                f'Unknown adaptation_mode: {self.adaptation_mode}'
            )
        
        # Load test data
        test_data, labels = self.load_test_data()
        
        if test_data['feature_dim'] != self.model_config['feature_dim']:
            raise ValueError(
                f"Aligned feature dimension {test_data['feature_dim']} does not "
                f"match checkpoint dimension {self.model_config['feature_dim']}"
            )
        
        # Build support set
        support_set_builder = FewShotSupportSet(
            num_abnormal=self.num_abnormal,
            num_normal=self.num_normal,
            seed=self.seed,
        )
        support_node_ids, strategy = support_set_builder.build_support_set(
            test_data, labels, last_time_idx=-1
        )

        if self.adaptation_mode == 'score_fusion':
            (
                score_features, score_feature_names, score_representations,
            ) = self.extract_score_features(model, test_data)
            zero_shot_query_results = self.evaluate_score_array(
                score_features[:, 0], labels, support_node_ids,
                evaluation_name='Few-Shot-Pretrained-QuerySet',
            )
            adapted_scores, adaptation = self.fit_score_fusion(
                score_features, score_feature_names, score_representations,
                labels, support_node_ids,
            )
            results = self.evaluate_score_array(
                adapted_scores, labels, support_node_ids,
                evaluation_name='Few-Shot-ScoreFusion-QuerySet',
            )
            save_path = self.model_path.replace(
                '.pth', '_fewshot_score_fusion.pth'
            )
            torch.save({
                'support_node_ids': support_node_ids,
                'strategy': strategy,
                'adaptation_mode': self.adaptation_mode,
                'adaptation': adaptation,
                'score_feature_names': score_feature_names,
                'zero_shot_query_results': zero_shot_query_results,
                'results': results,
            }, save_path)
            print(
                f"    Selected readout: {adaptation['name']} | "
                f"features={adaptation['selected_features']} | "
                f"C={adaptation['C']} | alpha={adaptation['alpha']} | "
                f"support-CV gain={adaptation['cv_gain']:.6f}"
            )
        else:
            zero_shot_query_results = self.evaluate_full_graph(
                model, test_data, labels, support_node_ids,
                evaluation_name='Few-Shot-Pretrained-QuerySet',
            )
            model = self.finetune_subgraph(
                model, test_data, labels, support_node_ids
            )
            results = self.evaluate_full_graph(
                model, test_data, labels, support_node_ids,
                evaluation_name='Few-Shot-Finetuned-QuerySet',
            )
            adaptation = {
                'name': 'network', 'selected_features': [], 'C': None,
                'alpha': None, 'cv_metric': None, 'base_cv_metric': None,
                'cv_gain': None,
            }
            save_path = self.model_path.replace('.pth', '_finetuned.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'support_node_ids': support_node_ids,
                'strategy': strategy,
                'adaptation_mode': self.adaptation_mode,
                'zero_shot_query_results': zero_shot_query_results,
                'results': results,
            }, save_path)
        print(f"\n Adaptation artifact saved to: {save_path}")

        support_hash = hashlib.sha256(
            np.asarray(support_node_ids, dtype=np.int64).tobytes()
        ).hexdigest()[:16]
        result_row = {
            'run_name': Path(self.model_path).resolve().parents[1].name,
            'stage': 'few_shot',
            'method_version': self.checkpoint.get('config', {}).get(
                'method_version', 'dipt'
            ),
            'adaptation_mode': self.adaptation_mode,
            'seed': self.seed,
            'start_year': self.start_year,
            'end_year': self.end_year,
            'num_abnormal': self.num_abnormal,
            'num_normal': self.num_normal,
            'lambda_fs': self.lambda_fs,
            'cv_folds': self.cv_folds,
            'min_cv_gain': self.min_cv_gain,
            'zero_auc_query': zero_shot_query_results['auc'],
            'zero_ap_query': zero_shot_query_results['ap'],
            'auc': results['auc'],
            'ap': results['ap'],
            'delta_auc': results['auc'] - zero_shot_query_results['auc'],
            'delta_ap': results['ap'] - zero_shot_query_results['ap'],
            'query_samples': len(results['labels']),
            'query_positives': int(np.asarray(results['labels']).sum()),
            'selected_readout': adaptation['name'],
            'selected_features': '|'.join(adaptation['selected_features']),
            'selected_c': adaptation['C'],
            'selected_alpha': adaptation['alpha'],
            'support_cv_metric': adaptation['cv_metric'],
            'support_cv_gain': adaptation['cv_gain'],
            'support_hash': support_hash,
            'evaluation_protocol': (
                'support-only model selection; disjoint labeled query nodes'
            ),
            'checkpoint': save_path,
        }
        upsert_csv_row(
            self.results_csv, result_row,
            key_fields=(
                'run_name', 'stage', 'adaptation_mode', 'seed',
                'num_abnormal', 'num_normal',
            ),
        )
        
        return model, results


def main():
    parser = argparse.ArgumentParser(description='SPAD few-shot adaptation')
    parser.add_argument('--model_path', type=str, required=True,
        help='Path to a repaired zero-shot best_source_loss.pth checkpoint')
    parser.add_argument('--start_year', type=int, default=2018, help='Start year (default: 2018)')
    parser.add_argument('--end_year', type=int, default=2022, help='End year (default: 2022)')
    parser.add_argument('--num_abnormal', type=int, default=10, help='Number of abnormal nodes (default: 10)')
    parser.add_argument('--num_normal', type=int, default=50, help='Number of normal nodes (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=1e-4 , help='Learning rate (default: 1e-4)')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lambda_fs', type=float, default=0.5)
    parser.add_argument('--max_neighbors_per_node', type=int, default=100)
    parser.add_argument(
        '--adaptation_mode', choices=['score_fusion', 'network'],
        default='score_fusion',
    )
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument(
        '--min_cv_gain', type=float, default=0.05,
        help='Minimum support-CV gain required to alter the zero-shot ranking',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_csv', type=str, default='results/few_shot_runs.csv')
    parser.add_argument('--figraph_path', type=str, default='data/FiGraph-main/data/',
        help='FiGraph data path (default: data/FiGraph-main/data/)')
    
    args = parser.parse_args()
    
    # Print used configuration
    print("=" * 60)
    print(" Few-Shot Adaptation Configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Time range: {args.start_year}-{args.end_year}")
    print(f"  Support set: {args.num_abnormal} abnormal + {args.num_normal} normal")
    if args.adaptation_mode == 'network':
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Number of epochs: {args.num_epochs}")
        print(f"  Batch size: {args.batch_size}")
    else:
        print(f"  Support CV folds: {args.cv_folds}")
        print(f"  Minimum CV gain: {args.min_cv_gain}")
    print(f"  Adaptation mode: {args.adaptation_mode}")
    print("=" * 60)
    print()
    
    # Create configuration
    config = {
        'model_path': args.model_path,
        'start_year': args.start_year,
        'end_year': args.end_year,
        'num_abnormal': args.num_abnormal,
        'num_normal': args.num_normal,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'lambda_fs': args.lambda_fs,
        'max_neighbors_per_node': args.max_neighbors_per_node,
        'adaptation_mode': args.adaptation_mode,
        'cv_folds': args.cv_folds,
        'min_cv_gain': args.min_cv_gain,
        'seed': args.seed,
        'results_csv': args.results_csv,
        'figraph_path': args.figraph_path
    }
    
    # Run few-shot adaptation
    finetuner = FewShotFinetuner(config)
    model, results = finetuner.run()
    
    print("\n" + "=" * 60)
    print(" Few-shot adaptation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

