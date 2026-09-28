import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import NeighborLoader, DataLoader as PyGDataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import AnomalyGFMDataProcessor
from anomaly_gfm_complete_model import AnomalyGFMModel, ContrastiveLoss, PrototypeContrastiveLoss, create_model_config
from sparse_graph_utils import SparseGraphProcessor, convert_dataset_to_sparse
from model_adapter import create_sparse_model_adapter
import scipy.sparse as sp


def set_global_seed(seed: int):
    """Set all RNGs used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def upsert_csv_row(path: str, row: Dict, key_fields=('run_name', 'stage', 'seed')):
    """Insert or replace one run so reruns cannot bias aggregate statistics."""
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



def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU memory cleared")



def temporal_collate_fn(batch):
    """
    Custom collate function for handling sparse matrices and complex structures in temporal batch data
    
    Args:
        batch: List[List[Data]] - temporal batch data
               - outer list: samples in the batch
               - inner list: temporal subgraph sequences for each sample
    
    Returns:
        Processed batch data, maintaining original structure
    """
    if not batch:
        return []
    return batch


def sparse_collate_fn(batch):
    """
    Optimized sparse matrix collate function for handling sparse matrices in TemporalGraphDataset
    Reduce memory usage and add memory monitoring
    
    Args:
        batch: batch data containing sparse matrices
    
    Returns:
        Processed batch data, converting sparse matrices to list format
    """
    if not batch:
        return []
    
    
    temporal_tensors = []
    adjacency_lists = []
    timestamps = []
    labels_list = []
    
    for item in batch:
        if len(item) == 4:
            temp_tensor, adj_list, timestamp, label = item
            if torch.cuda.is_available() and not temp_tensor.is_cuda:
                temp_tensor = temp_tensor.cuda()
            temporal_tensors.append(temp_tensor)
            adjacency_lists.append(adj_list)  # Keep sparse list format
            timestamps.append(timestamp)
            labels_list.append(label)
        elif len(item) == 3:
            temp_tensor, adj_list, timestamp = item
            if torch.cuda.is_available() and not temp_tensor.is_cuda:
                temp_tensor = temp_tensor.cuda()
            temporal_tensors.append(temp_tensor)
            adjacency_lists.append(adj_list)  # Keep sparse list format
            timestamps.append(timestamp)
            labels_list.append(None)
    
    # Clean up original batch data immediately
    del batch
    
    # Stack tensor data
    temporal_batch = torch.stack(temporal_tensors)
    timestamps_batch = torch.tensor(timestamps, device=temporal_batch.device) if timestamps[0] is not None else None
    
    # Clean up intermediate variables
    del temporal_tensors, timestamps
    
    # Process labels (if available)
    if labels_list[0] is not None:
        labels_batch = torch.stack([label for label in labels_list if label is not None])
        del labels_list
        result = (temporal_batch, adjacency_lists, timestamps_batch, labels_batch)
    else:
        del labels_list
        result = (temporal_batch, adjacency_lists, timestamps_batch)
    
    
    return result


class TemporalGraphDataset(Dataset):
    """
    Temporal graph dataset
    
    Create temporal feature sequences and corresponding graph snapshot sequences for each node
    """
    
    def __init__(self, temporal_data: Dict, sequence_length: int = 10, labels: np.ndarray = None):
        """
        Args:
            temporal_data: temporal data obtained from data_processor
            sequence_length: temporal sequence length
            labels: node labels (for supervised training)
        """
        self.temporal_sequences = temporal_data['temporal_sequences']
        self.temporal_adjacencies = temporal_data['temporal_adjacencies']
        self.valid_timestamps = temporal_data['valid_timestamps']
        self.sequence_length = sequence_length
        if labels is not None:
            unique_vals = np.unique(labels)
            if np.any((unique_vals > 1) | (unique_vals < 0)):
                print(
                    f"   Detected non-training label values {unique_vals.tolist()}; "
                    "only labels 0/1 will contribute to supervised losses"
                )
            # DGraphFin uses 2/3 for background/unlabelled nodes.  Mapping
            # every value > 0 to fraud would turn most of the graph into false
            # anomalies.  Preserve the values and mask to {0, 1} in the loss.
            self.labels = np.asarray(labels, dtype=np.float32)
        else:
            self.labels = labels
        
        print(f"TemporalGraphDataset initialized:")
        print(f"   - Number of valid sequences: {len(self.temporal_sequences)}")
        print(f"   - Sequence length: {self.sequence_length}")
        print(f"   - Has labels: {labels is not None}")
        if len(self.temporal_sequences) > 0:
            print(f"   - Number of nodes: {self.temporal_sequences[0][0].shape[0]}")
            print(f"   - Feature dimension: {self.temporal_sequences[0][0].shape[1]}")
    
    def __len__(self):
        return len(self.temporal_sequences)
    
    def __getitem__(self, idx):
        """
        Get a temporal sequence sample (supports sparse format)
        
        Returns:
            temporal_sequence: temporal feature sequence (seq_len, num_nodes, feature_dim)
            adjacency_sequence: adjacency matrix sequence (keep sparse format)
            timestamp: timestamp
            labels: node labels (if available)
        """
        temporal_sequence = self.temporal_sequences[idx][-self.sequence_length:]
        adjacency_sequence = self.temporal_adjacencies[idx][-self.sequence_length:]
        timestamp = self.valid_timestamps[idx]
        
        # Convert to torch tensor
        temporal_tensor = torch.stack([
            torch.FloatTensor(feat) for feat in temporal_sequence
        ])  # (seq_len, num_nodes, feature_dim)
        
        adjacency_sparse = adjacency_sequence  
        
        if self.labels is not None:
            num_nodes = temporal_tensor.shape[1]
            node_labels = torch.FloatTensor(self.labels[:num_nodes])
            return temporal_tensor, adjacency_sparse, timestamp, node_labels
        else:
            return temporal_tensor, adjacency_sparse, timestamp, None


class AnomalyGFMTrainer:
    """
    AnomalyGFM Trainer
    
    Implement complete training, validation, and testing pipeline
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = int(config.get('seed', 42))
        set_global_seed(self.seed)

        self.run_name = config.get('run_name', 'spad_full')
        self.output_dir = os.path.abspath(
            config.get('output_dir', os.path.join('outputs', self.run_name))
        )
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.results_csv = config.get(
            'results_csv', os.path.join('results', 'zero_shot_runs.csv')
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training parameters
        self.num_epochs = config.get('num_epochs', 100)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.batch_size = config.get('batch_size', 4)
        self.sequence_length = config.get('sequence_length', 10)
        
        self.feature_dim = config.get('feature_dim', 17)  
        
        self.contrastive_weight = config.get('contrastive_weight', 1.0)
        self.anomaly_weight = config.get('anomaly_weight', 1.0)
        self.lambda_self = float(config.get('lambda_self', 0.5))
        self.lambda_proto = float(config.get('lambda_proto', 0.3))
        self.lambda_reg = float(config.get('lambda_reg', 0.1))
        self.alpha_decorr = float(config.get('alpha_decorr', 0.1))
        self.temporal_margin = float(config.get('temporal_margin', 0.15))
        self.variance_floor = float(config.get('variance_floor', 0.08))
        
        self.sparse_processor = SparseGraphProcessor(device=str(self.device))
        
        print(f"AnomalyGFM Trainer initialized:")
        print(f"   - Device: {self.device}")
        print(f"   - Number of epochs: {self.num_epochs}")
        print(f"   - Learning rate: {self.learning_rate}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Sequence length: {self.sequence_length}")
        print(f"   - Integrated sparse graph optimization")

    def prepare_data(self, target_year: int = 2014, mode: str = 'dynamic'):
        """
        Unified data preparation method - supports dynamic and static graph modes
        
        Args:
            target_year: FiGraph target domain year (static mode)
            mode: 'static' - traditional static graph mode, 'dynamic' - new dynamic graph mode
        """
        
        if mode == 'dynamic':
            return self._prepare_dynamic_data()
        raise NotImplementedError(
            "The paper pipeline is dynamic. Use --mode dynamic; the legacy "
            "static path was incomplete and has been disabled."
        )
    

    def _prepare_dynamic_data(self, start_year: int = None, end_year: int = None):
        """
        Prepare true zero-shot dynamic graph prediction data
        
        Training set: DGraphFin dynamic graph sequences (source domain, with labels)
        Test set: FiGraph dynamic graph sequences (target domain, zero-shot inference)
        
        Args:
            start_year: FiGraph start year (from config)
            end_year: FiGraph end year (from config)
        """
        # Get year parameters from config
        if start_year is None:
            start_year = self.config.get('start_year', 2018)
        if end_year is None:
            end_year = self.config.get('end_year', 2022)
            
        
        data_config = {
            'temporal_window': 7,
            'target_dim': 16,
            'use_svd': False,
            'alignment_mode': self.config.get('alignment_mode', 'invariant'),
            'dgraphfin_path': self.config.get('dgraphfin_path', 'data/DGraphFin2/'),
            'figraph_path': self.config.get('figraph_path', 'data/FiGraph-main/data/'),
            'cache_dir': 'cache/',
            'enable_cache': True,
            'seed': self.seed,
        }
        
        processor = AnomalyGFMDataProcessor(data_config)
        
        dgraphfin_data = processor.load_dgraphfin_data()
        
        snapshots = processor.create_temporal_snapshots(dgraphfin_data, window_size=7)
        
        dgraphfin_temporal = processor.prepare_temporal_sequences(
            dgraphfin_data, snapshots
        )
        
        figraph_temporal = processor.load_figraph_temporal_sequence(
            start_year=start_year, 
            end_year=end_year
        )
        
        print(f"   Successfully loaded FiGraph dynamic sequence:")
        print(f"      - Sequence length: {figraph_temporal['sequence_length']} years")
        print(f"      - Number of nodes: {figraph_temporal['num_nodes']:,}")
        print(f"      - Feature dimension: {figraph_temporal['feature_dim']}")
        print(f"      - Edge evolution stats: {figraph_temporal['edge_evolution_stats']}")
        print(f"Performing cross-domain dynamic graph feature space alignment...")
        
        # Step 1: fit on the actual source subset used for training and the
        # first target snapshot, then reuse exactly the same transforms for
        # every year.
        dgraphfin_ref_features = dgraphfin_temporal['temporal_sequences'][0][0]
        figraph_ref_features = figraph_temporal['temporal_sequences'][0][0]  
        
        
        # Step 2: Compute feature alignment transformation (one-time calculation)
        dgraphfin_aligned, figraph_aligned = processor.fit_feature_aligner(
            dgraphfin_ref_features,
            figraph_ref_features
        )
        
        # Step 3: apply the fitted transforms to all source/target time steps.
        for batch_idx in range(len(dgraphfin_temporal['temporal_sequences'])):
            for time_idx in range(len(dgraphfin_temporal['temporal_sequences'][batch_idx])):
                current_features = dgraphfin_temporal['temporal_sequences'][batch_idx][time_idx]
                dgraphfin_temporal['temporal_sequences'][batch_idx][time_idx] = (
                    processor.transform_source_features(current_features)
                )

        for batch_idx in range(len(figraph_temporal['temporal_sequences'])):
            for time_idx in range(len(figraph_temporal['temporal_sequences'][batch_idx])):
                current_features = figraph_temporal['temporal_sequences'][batch_idx][time_idx]
                figraph_temporal['temporal_sequences'][batch_idx][time_idx] = (
                    figraph_aligned if time_idx == 0 else
                    processor.transform_target_features(current_features)
                )

        # Step 4: append topology/evolution coordinates whose semantics are
        # identical in both domains (degree context, temporal change and edge
        # churn).  The explicit switch is used only by the component ablation;
        # the full model keeps these descriptors enabled.
        if self.config.get('use_structural_temporal_descriptors', True):
            for batch_idx in range(len(dgraphfin_temporal['temporal_sequences'])):
                dgraphfin_temporal['temporal_sequences'][batch_idx] = (
                    processor.append_structural_temporal_features(
                        dgraphfin_temporal['temporal_sequences'][batch_idx],
                        dgraphfin_temporal['temporal_adjacencies'][batch_idx],
                    )
                )
            for batch_idx in range(len(figraph_temporal['temporal_sequences'])):
                figraph_temporal['temporal_sequences'][batch_idx] = (
                    processor.append_structural_temporal_features(
                        figraph_temporal['temporal_sequences'][batch_idx],
                        figraph_temporal['temporal_adjacencies'][batch_idx],
                    )
                )

        self.feature_dim = dgraphfin_temporal['temporal_sequences'][0][0].shape[1]
        dgraphfin_temporal['feature_dim'] = self.feature_dim
        figraph_temporal['feature_dim'] = self.feature_dim
        
        
        self.feature_aligner_path = os.path.join(
            self.output_dir, 'feature_aligner.pkl'
        )
        processor.save_feature_aligner(self.feature_aligner_path)

        selected_node_ids = dgraphfin_temporal.get(
            'selected_node_ids',
            np.arange(dgraphfin_temporal['num_nodes']),
        )
        train_labels = np.asarray(dgraphfin_data['labels'])[selected_node_ids]
        
        train_dataset = TemporalGraphDataset(
            dgraphfin_temporal, 
            self.sequence_length,
            labels=train_labels
        )
        
        convert_dataset_to_sparse(train_dataset)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=min(self.batch_size, len(train_dataset)), 
            shuffle=True,
            collate_fn=sparse_collate_fn,  # Fixed: use sparse matrix collate function
            num_workers=0
        )
        
        self.train_loader = train_loader
        self.train_data = {
            'features': dgraphfin_temporal['temporal_sequences'][0][-1],
            'labels': train_labels,
            'temporal_data': dgraphfin_temporal,
            'dataset': 'DGraphFin-Dynamic'
        }
        
        self.test_data = {
            'labels': figraph_temporal['temporal_labels'][0][-1] if figraph_temporal['temporal_labels'] else None,
            'dataset': f'FiGraph-Dynamic-{start_year}-{end_year}',
            'mode': 'dynamic',
            
            'temporal_sequences': figraph_temporal['temporal_sequences'],
            'temporal_adjacencies': figraph_temporal['temporal_adjacencies'],
            'temporal_labels': figraph_temporal['temporal_labels'],
            'sequence_length': figraph_temporal['sequence_length'],
            'num_nodes': figraph_temporal['num_nodes'],
            'feature_dim': figraph_temporal['feature_dim'],
            
            'features': figraph_temporal['temporal_sequences'][0][-1] if figraph_temporal['temporal_sequences'] else None,
            'adjacency_matrix': figraph_temporal['temporal_adjacencies'][0][-1] if figraph_temporal['temporal_adjacencies'] else None,
        }
        
        print(f"Dynamic graph data preparation completed!")
        print(f"   - Training domain: DGraphFin dynamic sequences ({len(snapshots)} time windows)")
        print(f"   - Test domain: FiGraph dynamic sequences ({figraph_temporal['sequence_length']} years)")
        print(f"   - Aligned feature dimension: {self.feature_dim}")
        print(f"   - Training paradigm: Dynamic graph temporal modeling")
        print(f"   - Evaluation paradigm: Zero-shot dynamic graph prediction")

    
    def create_model(self):
        """
        Create TGN temporal graph neural network model
        """
        
        # Create model configuration
        model_config = create_model_config(self.feature_dim, self.sequence_length)
        model_config.update(self.config.get('model_config', {}))
        model_config['prototype_mode'] = self.config.get(
            'prototype_mode', model_config.get('prototype_mode', 'single')
        )
        self.resolved_model_config = model_config.copy()
        
        
        base_model = AnomalyGFMModel(model_config).to(self.device)
        
        self.model = create_sparse_model_adapter(
            base_model=base_model,
            device=self.device,
            memory_threshold_mb=100.0  
        )
        
        
        # Loss functions
        temperature = model_config.get('temperature', 0.1)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.prototype_contrastive_loss = PrototypeContrastiveLoss(temperature=temperature)  # New prototype contrastive loss
        self.anomaly_loss_fn = nn.BCEWithLogitsLoss()
        
        # Loss weights
        self.proto_loss_weight = self.config.get('proto_loss_weight', 0.5)  # Prototype loss weight

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    
    
    def evaluate_zero_shot(self) -> Dict[str, float]:
        """
        Evaluate the repaired dynamic-graph zero-shot pipeline.
        """
        if self.test_data.get('mode') != 'dynamic':
            raise ValueError('Only dynamic-graph evaluation is supported.')
        result = self._evaluate_dynamic_zero_shot()
        clear_gpu_memory()
        return result
    

    def _evaluate_dynamic_zero_shot(self) -> Dict[str, float]:
        """
        Dynamic graph zero-shot evaluation - performing true dynamic graph prediction on FiGraph dynamic sequences
        """
        
        self.model.eval()
        
        if self.test_data.get('mode') != 'dynamic':
            raise ValueError('Dynamic FiGraph data is required for zero-shot evaluation.')
        
        temporal_sequences = self.test_data['temporal_sequences']
        temporal_adjacencies = self.test_data['temporal_adjacencies']
        temporal_labels = self.test_data['temporal_labels']
        test_labels = self.test_data['labels']
        sequence_length = self.test_data['sequence_length']
        
        print(f"   Dynamic sequence information:")
        print(f"      - Temporal length: {sequence_length} years") 
        print(f"      - Number of batches: {len(temporal_sequences)}")
        if temporal_sequences:
            print(f"      - Number of nodes: {temporal_sequences[0][0].shape[0]:,}")
            print(f"      - Feature dimension: {temporal_sequences[0][0].shape[1]}")
        
        with torch.no_grad():
            try:
                
                batch_temporal_sequences = temporal_sequences[0]  
                batch_temporal_adjacencies = temporal_adjacencies[0]
                
                seq_len = len(batch_temporal_sequences)
                num_nodes = batch_temporal_sequences[0].shape[0]
                feature_dim = batch_temporal_sequences[0].shape[1]
                
                temporal_tensor = np.stack(batch_temporal_sequences, axis=0)  # (seq_len, num_nodes, feature_dim)
                temporal_tensor = torch.FloatTensor(temporal_tensor).unsqueeze(0).to(self.device)  # (1, seq_len, num_nodes, feature_dim)
                
                
                temporal_adjacency_info = []
                
                for t, adj_matrix in enumerate(batch_temporal_adjacencies):
                    
                    graph_info = self.sparse_processor.zero_shot_graph_processing(
                        adj_matrix, 
                        batch_size=1  
                    )
                    
                    sparse_adjacency_info = {
                        'edge_index': graph_info['edge_index'].to(self.device),
                        'edge_attr': graph_info['edge_attr'].to(self.device) if graph_info['edge_attr'] is not None else None,
                        'num_nodes': num_nodes
                    }
                    temporal_adjacency_info.append(sparse_adjacency_info)
                
                outputs = self.model(
                    temporal_tensor,                    
                    temporal_adjacency_info,           
                    temporal_tensor[:, -1]             
                )
                
                
                if 'anomaly_scores' in outputs:
                    anomaly_scores = outputs['anomaly_scores'].cpu().numpy().flatten()
                else:
                    raise ValueError(f"Model output format not supported. Available keys: {list(outputs.keys())}")
                
                
                if test_labels is not None:
                    if len(test_labels) == len(anomaly_scores):
                        valid_mask = np.isin(test_labels, [0, 1])
                        valid_labels = np.asarray(test_labels)[valid_mask]
                        valid_scores = np.asarray(anomaly_scores)[valid_mask]
                        excluded_unlabeled = int((~valid_mask).sum())
                        print(
                            '\n Evaluation protocol: labeled target nodes only '
                            f'(excluded Label=-1/unlabeled nodes: {excluded_unlabeled:,})'
                        )
                        if len(np.unique(valid_labels)) < 2:
                            raise ValueError('Target snapshot has fewer than two valid classes')
                        eval_results = AnomalyGFMDataProcessor.evaluate_ranking_performance(
                            valid_labels, valid_scores, "FiGraph-Dynamic-ZeroShot"
                        )

                        diagnostic_scores = {
                            'transported_prototype_distance': outputs['anomaly_distances']
                                .detach().cpu().numpy().reshape(-1),
                            'source_prototype_distance': outputs[
                                'learned_prototype_distances'
                            ].detach().cpu().numpy().reshape(-1),
                            'prototype_transport_residual': outputs[
                                'prototype_transport_residual'
                            ].detach().cpu().numpy().reshape(-1),
                            'source_prototype_similarity': outputs[
                                'source_prototype_similarity'
                            ].detach().cpu().numpy().reshape(-1),
                            'dynamic_neighbor_inconsistency': outputs[
                                'dynamic_neighbor_inconsistency'
                            ].detach().cpu().numpy().reshape(-1),
                            'recency_neighbor_inconsistency': outputs[
                                'recency_neighbor_inconsistency'
                            ].detach().cpu().numpy().reshape(-1),
                            'temporal_volatility': outputs['temporal_volatility']
                                .detach().cpu().numpy().reshape(-1),
                            'residual_magnitude': torch.norm(
                                outputs['residual_features'], dim=-1
                            ).detach().cpu().numpy().reshape(-1),
                            'attribute_energy': torch.norm(
                                temporal_tensor[:, -1, :, :16], dim=-1
                            ).detach().cpu().numpy().reshape(-1),
                            'structural_energy': torch.norm(
                                temporal_tensor[:, -1, :, 16:], dim=-1
                            ).detach().cpu().numpy().reshape(-1),
                            'input_temporal_change': torch.norm(
                                temporal_tensor[:, -1] - temporal_tensor[:, -2],
                                dim=-1,
                            ).detach().cpu().numpy().reshape(-1),
                        }
                        diagnostics = {}
                        print('\n Component ranking diagnostics:')
                        for name, values in diagnostic_scores.items():
                            auc = roc_auc_score(valid_labels, values[valid_mask])
                            ap = average_precision_score(valid_labels, values[valid_mask])
                            diagnostics[name] = {'auc': auc, 'ap': ap}
                            print(f'    {name}: AUC={auc:.4f}, AP={ap:.4f}')
                        
                        return {
                            'dynamic_auc': eval_results['roc_auc'],
                            'dynamic_ap': eval_results['average_precision'],
                            'num_samples': len(valid_scores),
                            'mode': 'dynamic_graph_prediction',
                            'sequence_length': seq_len,
                            'temporal_coverage': f'{sequence_length} years sequence',
                            'evaluation_protocol': 'labeled_target_nodes_only',
                            'excluded_unlabeled_nodes': excluded_unlabeled,
                            'diagnostics': diagnostics,
                        }
                    else:
                        print(f"Label count ({len(test_labels)}) does not match score count ({len(anomaly_scores)})")
                
                return {
                    'num_samples': len(anomaly_scores),
                    'mode': 'dynamic_graph_prediction',
                    'sequence_length': seq_len,
                    'scores_mean': float(np.mean(anomaly_scores)),
                    'scores_std': float(np.std(anomaly_scores)),
                    'temporal_coverage': f'{sequence_length} years sequence'
                }
                        
            except Exception as e:
                raise RuntimeError(
                    'Dynamic FiGraph zero-shot evaluation failed; refusing to '
                    'record placeholder AUC/AP values.'
                ) from e
    
    def _detect_consistency_loss_turning_point(
        self,
        consistency_records: List[Dict[str, float]],
        min_change_ratio: float = 0.0001
    ) -> Tuple[bool, str, Optional[Dict[str, float]]]:
        """
        Detect turning point based on temporal loss printed per batch (from rise to fall)
        
        Args:
            consistency_records: [{'epoch': int, 'batch_idx': int, 'value': float}, ...]
            min_change_ratio: Threshold for judging "obvious rise/fall" relative change ratio (default 0.01%, i.e., 0.0001)
        
        Returns:
            (is_turning_point, description, turning_record)
        """
        if len(consistency_records) < 3:
            return False, "Insufficient historical data", None
        
        values = [rec.get('value', 0.0) for rec in consistency_records]
        eps = 1e-8
        
        in_rise = False
        rise_start_idx = -1
        peak_idx = -1
        rise_count = 0  
        consecutive_drop = 0  
        
        for i in range(1, len(values)):
            prev_val = values[i - 1]
            curr_val = values[i]
            delta = curr_val - prev_val
            rel_change = abs(delta) / (abs(prev_val) + eps)
            
            if rel_change < min_change_ratio:
                if in_rise:
                    continue
                else:
                    continue
            elif delta > 0:  
                consecutive_drop = 0  
                
                if not in_rise:
                    in_rise = True
                    rise_start_idx = i - 1  
                    rise_count = 1
                    peak_idx = i  
                else:
                    rise_count += 1
                    peak_idx = i  
                    
            elif delta < 0:  
                if in_rise:
                    consecutive_drop += 1
                    if consecutive_drop >= 1:  
                        if rise_count >= 2:
                            turning_idx = peak_idx
                            turning_record = consistency_records[turning_idx]
                            description = (
                                f"Temporal loss turning from rise to fall: "
                                f"Rise ({values[rise_start_idx]:.4f}->{values[peak_idx]:.4f}, {rise_count} points), "
                                f"Fall ({values[peak_idx]:.4f}->{values[i]:.4f})"
                            )
                            return True, description, turning_record
                        else:
                            in_rise = False
                            rise_start_idx = -1
                            peak_idx = -1
                            rise_count = 0
                            consecutive_drop = 0
                else:
                    pass
        
        return False, "", None
    
    def train(self):
        """Complete training workflow"""
        print(f"\nStarting AnomalyGFM zero-shot training!")
        print(f"=" * 60)
        
        best_train_loss = float('inf')
        best_model_filename = 'best_source_loss.pth'
        train_history = []
        consistency_loss_history = []  
        consistency_loss_records = []  
        turning_point_model_info = None  
        turning_point_saved = False  
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch_sparse()
    
            consistency_loss = train_metrics.get('consistency_loss', 0.0)
            consistency_loss_history.append(consistency_loss)
            
            for record in train_metrics.get('consistency_series', []):
                consistency_loss_records.append({
                    'epoch': epoch,
                    'batch_idx': record.get('batch_idx', len(consistency_loss_records)),
                    'value': record.get('value', 0.0)
                })
            
            if not turning_point_saved:
                is_turning_point, description, turning_record = self._detect_consistency_loss_turning_point(consistency_loss_records)
                if is_turning_point:
                    turning_value = turning_record.get('value', consistency_loss) if turning_record else consistency_loss
                    turning_epoch = turning_record.get('epoch', epoch) if turning_record else epoch
                    turning_batch = turning_record.get('batch_idx', 0) if turning_record else 0

                    print(f"\n   Detected temporal loss turning point (Epoch {turning_epoch}, Batch {turning_batch}): {description}")
                    print(f"      Turning point temporal loss: {turning_value:.6f}")
                    model_filename = f'turning_point_epoch_{turning_epoch}_batch_{turning_batch}_consistency_{turning_value:.6f}.pth'
                    self.save_model(model_filename)
                    turning_point_model_info = {
                        'epoch': turning_epoch,
                        'batch_idx': turning_batch,
                        'consistency_loss': turning_value,
                        'filename': model_filename,
                        'description': description
                    }
                    turning_point_saved = True  
            
            # Select checkpoints exclusively from source-domain training loss.
            # Target labels are used once, after training, for final reporting.
            eval_metrics = {}
            current_loss = train_metrics['total_loss']
            if current_loss < best_train_loss:
                best_train_loss = current_loss
                self.save_model(best_model_filename)
            self.scheduler.step(current_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            # Record training history
            epoch_info = {
                    'epoch': epoch,
                'train_loss': train_metrics['total_loss'],
                'contrastive_loss': train_metrics.get('contrastive_loss', 0.0),
                'anomaly_loss': train_metrics.get('anomaly_loss', 0.0),
                'prototype_loss': train_metrics.get('prototype_loss', 0.0),
                'consistency_loss': train_metrics.get('consistency_loss', 0.0),
                'residual_loss': train_metrics.get('residual_loss', 0.0),
                'unsupervised_loss': train_metrics.get('unsupervised_loss', 0.0),
                'auc': eval_metrics.get('auc', 0.0),
                'ap': eval_metrics.get('ap', 0.0),
                'time': epoch_time
            }
            train_history.append(epoch_info)
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                print(f"Epoch {epoch:3d}/{self.num_epochs} | "
                      f"Total Loss: {train_metrics['total_loss']:.4f} | "
                      f"Time: {epoch_time:.2f}s")
                
                # Print detailed loss items
                print(f"   Loss Breakdown:")
                print(f"      - Anomaly Loss: {train_metrics.get('anomaly_loss', 0.0):.4f}")
                print(f"      - Contrastive Loss: {train_metrics.get('contrastive_loss', 0.0):.4f}")
                print(f"      - Prototype Loss: {train_metrics.get('prototype_loss', 0.0):.4f}")
                print(f"      - Consistency Loss: {train_metrics.get('consistency_loss', 0.0):.4f}")
                print(f"      - Residual Loss: {train_metrics.get('residual_loss', 0.0):.4f}")
            else:
                # Output simplified loss info every 5 epochs
                if epoch % 5 == 0:
                    print(f"Epoch {epoch:3d}/{self.num_epochs} | "
                          f"Total Loss: {train_metrics['total_loss']:.4f} | "
                          f"Anomaly: {train_metrics.get('anomaly_loss', 0.0):.4f} | "
                          f"Contrastive: {train_metrics.get('contrastive_loss', 0.0):.4f} | "
                          f"Prototype: {train_metrics.get('prototype_loss', 0.0):.4f}")
            
            # Clear GPU memory after each epoch
            clear_gpu_memory()
        
        print(f"\nTraining Complete!")
        print(f"   - Best source loss: {best_train_loss:.6f}")
        print(f"   - Total Training Time: {sum(h['time'] for h in train_history):.2f} seconds")
        if turning_point_model_info:
            print(f"   - Turning Point Detected: Epoch {turning_point_model_info['epoch']} (Batch {turning_point_model_info.get('batch_idx', 0)})")
        else:
            print(f"   - No Turning Point Detected")
        
        # Restore the source-selected checkpoint and evaluate the target once.
        best_model_path = os.path.join(self.checkpoint_dir, best_model_filename)
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        eval_metrics = self.evaluate_zero_shot()
        final_auc = eval_metrics.get('auc', eval_metrics.get('dynamic_auc', 0.0))
        final_ap = eval_metrics.get('ap', eval_metrics.get('dynamic_ap', 0.0))
        print(f"   - Final target AUC: {final_auc:.4f}")
        print(f"   - Final target AP: {final_ap:.6f}")

        result_row = {
            'run_name': self.run_name,
            'stage': 'zero_shot',
            'method_version': self.config.get('method_version', 'dipt'),
            'prototype_mode': self.config.get('prototype_mode', 'single'),
            'seed': self.seed,
            'start_year': self.config.get('start_year', ''),
            'end_year': self.config.get('end_year', ''),
            'lambda_self': self.lambda_self,
            'lambda_proto': self.lambda_proto,
            'lambda_reg': self.lambda_reg,
            'auc': final_auc,
            'ap': final_ap,
            'evaluation_protocol': eval_metrics.get(
                'evaluation_protocol', 'labeled_target_nodes_only'
            ),
            'checkpoint': best_model_path,
        }
        upsert_csv_row(self.results_csv, result_row)
        history_path = os.path.join(self.output_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump({
                'train_history': train_history,
                'consistency_loss_history': consistency_loss_history,
                'consistency_loss_records': consistency_loss_records,
                'turning_point_model_info': turning_point_model_info,
                'final_evaluation': eval_metrics
            }, f)
        print(f"\nTraining History Saved: {history_path}")
        
        return train_history
    
    def save_model(self, filename: str):
        """Save model"""
        save_path = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'resolved_model_config': self.resolved_model_config,
                    'feature_aligner_path': getattr(self, 'feature_aligner_path', None),
        }, save_path)
        
        print(f"   Model Saved: {save_path}")
    
    def train_epoch_sparse(self) -> Dict[str, float]:
        """
        Fixed training method - correctly handle data format returned by sparse_collate_fn
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        epoch_anomaly_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_prototype_loss = 0.0
        epoch_consistency_loss = 0.0
        epoch_residual_loss = 0.0
        consistency_series = []
        
        print(f"   Starting Sparse Format Training Loop...")
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            try:
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.optimizer.zero_grad()
                
                if len(batch_data) == 4:
                    temporal_batch, adjacency_lists, timestamps_batch, labels_batch = batch_data
                elif len(batch_data) == 3:
                    temporal_batch, adjacency_lists, timestamps_batch = batch_data
                    labels_batch = None
                else:
                    print(f"   Warning: Incorrect batch data format, length: {len(batch_data)}")
                    continue
                
                if not isinstance(temporal_batch, torch.Tensor):
                    print(f"   Warning: temporal_batch is not a Tensor: {type(temporal_batch)}")
                    continue
                
                temporal_batch = temporal_batch.to(self.device)
                if timestamps_batch is not None:
                    timestamps_batch = timestamps_batch.to(self.device)
                if labels_batch is not None:
                    labels_batch = labels_batch.to(self.device)
                
                batch_size, seq_len, num_nodes, feature_dim = temporal_batch.shape
                print(f"   Batch {batch_idx}: {temporal_batch.shape}, {len(adjacency_lists)} sparse matrices")
                
                adjacency_info = []
                if batch_size != 1:
                    raise ValueError(
                        'The full-graph temporal loader expects batch_size=1; '
                        f'got {batch_size}'
                    )
                for t in range(seq_len):
                    if not adjacency_lists or t >= len(adjacency_lists[0]):
                        raise ValueError(f'Missing adjacency matrix for timestep {t}')
                    adj_t = adjacency_lists[0][t]
                    edge_index, edge_attr = self.sparse_processor.sparse_to_pyg_format(adj_t)
                    adjacency_info.append({
                        'edge_index': edge_index.to(self.device),
                        'edge_attr': edge_attr.to(self.device) if edge_attr is not None else None,
                        'num_nodes': num_nodes,
                        'format': 'pyg_sparse',
                    })
                
                node_features = temporal_batch[:, -1]  # (batch_size, num_nodes, feature_dim)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                outputs = self.model(temporal_batch, adjacency_info, node_features)
                
                fused_representations = outputs['fused_representations']
                anomaly_logits = outputs['anomaly_logits']
                sim_to_normal = outputs['sim_to_normal']
                sim_to_abnormal = outputs.get('sim_to_abnormal')
                consistency_representations = outputs['consistency_representations']
                
                # Compute loss
                if labels_batch is not None:
                    # If labels available, use supervised loss
                    valid_mask = (labels_batch == 0) | (labels_batch == 1)
                    if valid_mask.sum() > 0:
                        valid_logits = anomaly_logits[valid_mask]
                        valid_labels = labels_batch[valid_mask].float()

                        anomaly_loss = F.binary_cross_entropy_with_logits(
                            valid_logits, valid_labels
                        )
                        
                        # Record anomaly detection loss
                        batch_anomaly_loss = anomaly_loss.item()
                        
                        normal_mask = labels_batch == 0
                        prototype_loss = torch.tensor(
                            0.0, device=fused_representations.device
                        )
                        if normal_mask.any():
                            prototype_loss = (1 - sim_to_normal[normal_mask]).mean()

                        # Explicit dual-prototype ablation.  The default SPAD
                        # path never instantiates or uses an abnormal prototype.
                        if (
                            self.config.get('prototype_mode') == 'dual'
                            and sim_to_abnormal is not None
                        ):
                            abnormal_mask = labels_batch == 1
                            if abnormal_mask.any():
                                prototype_loss = prototype_loss + (
                                    1 - sim_to_abnormal[abnormal_mask]
                                ).mean()
                            normal_p = F.normalize(
                                outputs['normal_prototype'], p=2, dim=1
                            )
                            abnormal_p = F.normalize(
                                outputs['abnormal_prototype'], p=2, dim=1
                            )
                            proto_distance = 1 - (normal_p * abnormal_p).sum()
                            prototype_loss = prototype_loss + 0.5 * F.relu(
                                0.5 - proto_distance
                            )
                        
                        batch_prototype_loss = float(prototype_loss.detach().cpu().item())
                        
                        # Contrastive loss disabled (single prototype)
                        contrastive_loss = torch.tensor(0.0, device=fused_representations.device)
                        batch_contrastive_loss = 0.0
                        
                        # Compute temporal consistency loss (law 1)
                        consistency_loss = self._compute_temporal_consistency_loss(
                            consistency_representations, labels_batch
                        )
                        batch_consistency_loss = consistency_loss.item()
                        consistency_series.append({
                            'batch_idx': batch_idx,
                            'value': batch_consistency_loss
                        })
                        
                        # Compute residual regularization loss (law 2)
                        residual_loss = self._compute_residual_regularization_loss(
                            consistency_representations
                        )
                        batch_residual_loss = residual_loss.item()
                        
                        # Five loss function combination: core loss + regularization losses
                        total_loss = (
                            anomaly_loss
                            + self.lambda_proto * prototype_loss
                            + self.lambda_self * consistency_loss
                            + self.lambda_reg * residual_loss
                        )
                    else:
                        # No valid labels, skip this batch
                        continue
                else:
                    # No labels, skip this batch
                    continue
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_anomaly_loss += batch_anomaly_loss
                epoch_contrastive_loss += batch_contrastive_loss
                epoch_prototype_loss += batch_prototype_loss
                epoch_consistency_loss += batch_consistency_loss
                epoch_residual_loss += batch_residual_loss
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"      Batch {batch_idx}: Total Loss={total_loss.item():.4f} | "
                          f"Anomaly={batch_anomaly_loss:.4f} | Contrastive={batch_contrastive_loss:.4f} | "
                          f"Prototype={batch_prototype_loss:.4f} | Consistency={batch_consistency_loss:.4f} | "
                          f"Residual={batch_residual_loss:.4f}")
                
                del temporal_batch, adjacency_lists, timestamps_batch
                if labels_batch is not None:
                    del labels_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f'Training failed at batch {batch_idx}; no checkpoint was '
                    'written from this failed epoch.'
                ) from e

        if num_batches == 0:
            raise RuntimeError(
                'No valid training batch was completed. Check source labels and '
                'the temporal graph data before rerunning.'
            )
        
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_anomaly_loss = epoch_anomaly_loss / max(num_batches, 1)
        avg_contrastive_loss = epoch_contrastive_loss / max(num_batches, 1)
        avg_prototype_loss = epoch_prototype_loss / max(num_batches, 1)
        avg_consistency_loss = epoch_consistency_loss / max(num_batches, 1)
        avg_residual_loss = epoch_residual_loss / max(num_batches, 1)
        
        clear_gpu_memory()
        
        return {
            'loss': avg_loss,
            'total_loss': avg_loss,
            'anomaly_loss': avg_anomaly_loss,
            'contrastive_loss': avg_contrastive_loss,
            'prototype_loss': avg_prototype_loss,
            'consistency_loss': avg_consistency_loss,
            'residual_loss': avg_residual_loss,
            'consistency_series': consistency_series
        }
    
    
    def _compute_temporal_consistency_loss(
        self, temporal_representations: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normal smoothness plus supervised volatility separation."""
        if temporal_representations.dim() != 4:
            raise ValueError(
                'Temporal consistency expects [batch, time, node, feature], '
                f'got {tuple(temporal_representations.shape)}'
            )
        if temporal_representations.shape[1] <= 1:
            return temporal_representations.new_tensor(0.0)

        normalized = F.normalize(temporal_representations, p=2, dim=-1)
        temporal_diffs = normalized[:, 1:] - normalized[:, :-1]
        # [batch, time - 1, node] -> [batch, node]
        node_volatility = torch.norm(temporal_diffs, dim=-1).mean(dim=1)
        if labels is not None:
            normal_mask = labels == 0
            abnormal_mask = labels == 1
            if normal_mask.any():
                normal_smoothness = node_volatility[normal_mask].mean()
                if abnormal_mask.any():
                    abnormal_volatility = node_volatility[abnormal_mask].mean()
                    separation = F.relu(
                        self.temporal_margin
                        + normal_smoothness
                        - abnormal_volatility
                    )
                    return normal_smoothness + separation
                return normal_smoothness
        return node_volatility.mean()
    
    def _compute_residual_regularization_loss(
        self, temporal_representations: torch.Tensor
    ) -> torch.Tensor:
        """Temporal intensity stability plus feature decorrelation."""
        if temporal_representations.dim() != 4:
            raise ValueError(
                'Representation regularization expects [batch, time, node, feature]'
            )

        # VICReg-style variance floor prevents the temporal/prototype losses
        # from satisfying their objectives by mapping every node to a constant.
        feature_dim = temporal_representations.shape[-1]
        flat_repr = temporal_representations.reshape(-1, feature_dim)
        centered = flat_repr - flat_repr.mean(dim=0, keepdim=True)
        feature_std = torch.sqrt(
            centered.var(dim=0, unbiased=False) + 1e-4
        )
        variance_loss = F.relu(self.variance_floor - feature_std).mean()
        denominator = max(flat_repr.shape[0] - 1, 1)
        covariance = centered.t().matmul(centered) / denominator
        off_diagonal = covariance - torch.diag(torch.diag(covariance))
        decorrelation = off_diagonal.pow(2).mean()
        return variance_loss + self.alpha_decorr * decorrelation


def main():
    parser = argparse.ArgumentParser(description='SPAD zero-shot dynamic graph training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Full-graph batch size (must be 1)')
    parser.add_argument('--sequence_length', type=int, default=8, help='Temporal sequence length')
    parser.add_argument('--mode', type=str, default='dynamic', choices=['dynamic'])
    parser.add_argument('--start_year', type=int, default=2018, help='FiGraph start year (dynamic mode)')
    parser.add_argument('--end_year', type=int, default=2022, help='FiGraph end year (dynamic mode)')
    parser.add_argument('--prototype_mode', choices=['single', 'dual'], default='single')
    parser.add_argument('--alignment_mode', choices=['invariant', 'svd'], default='invariant')
    parser.add_argument('--method_version', type=str, default='dipt')
    parser.add_argument('--disable_transport_residual', action='store_true')
    parser.add_argument('--disable_local_fusion', action='store_true')
    parser.add_argument('--disable_structural_temporal_descriptors', action='store_true')
    parser.add_argument('--disable_residual_fusion', action='store_true')
    parser.add_argument('--lambda_self', type=float, default=0.5)
    parser.add_argument('--lambda_proto', type=float, default=0.3)
    parser.add_argument('--lambda_reg', type=float, default=0.1)
    parser.add_argument('--alpha_decorr', type=float, default=0.1)
    parser.add_argument('--temporal_margin', type=float, default=0.15)
    parser.add_argument('--variance_floor', type=float, default=0.08)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_name', type=str, default='spad_full_seed42')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--results_csv', type=str, default='results/zero_shot_runs.csv')
    parser.add_argument('--dgraphfin_path', type=str, default='data/DGraphFin2/')
    parser.add_argument('--figraph_path', type=str, default='data/FiGraph-main/data/')
    
    args = parser.parse_args()
    
    config = {
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length,
        'weight_decay': 1e-5,
        'contrastive_weight': 1.0,
        'anomaly_weight': 0.5,
        'dgraphfin_path': args.dgraphfin_path,
        'figraph_path': args.figraph_path,
        'mode': args.mode,
        'start_year': args.start_year,
        'end_year': args.end_year,
        'prototype_mode': args.prototype_mode,
        'alignment_mode': args.alignment_mode,
        'method_version': args.method_version,
        'lambda_self': args.lambda_self,
        'lambda_proto': args.lambda_proto,
        'lambda_reg': args.lambda_reg,
        'alpha_decorr': args.alpha_decorr,
        'temporal_margin': args.temporal_margin,
        'variance_floor': args.variance_floor,
        'use_structural_temporal_descriptors': not args.disable_structural_temporal_descriptors,
        'seed': args.seed,
        'run_name': args.run_name,
        'output_dir': args.output_dir or os.path.join('outputs', args.run_name),
        'results_csv': args.results_csv,
        'model_config': {
            'projection_dim': 32,
            'hidden_dim': 64,
            'output_dim': 32,
            'tcn_hidden_dims': [32, 64, 32],  
            'gnn_hidden_dims': [32, 64, 32],  
            'dropout': 0.3,
            'temperature': 0.1,
            'use_transport_residual': not args.disable_transport_residual,
            'use_local_evolution_score': not args.disable_local_fusion,
            'use_residual_fusion': not args.disable_residual_fusion,
        }
    }
    
    print("SPAD Zero-Shot Dynamic Graph Anomaly Detection")
    print("=" * 60)
    print(f"Training Configuration (DGraphFin -> FiGraph {args.start_year}-{args.end_year}):")
    for key, value in config.items():
        if key != 'model_config':
            print(f"   - {key}: {value}")
    
    try:
        trainer = AnomalyGFMTrainer(config)
        
        trainer.prepare_data(mode='dynamic')
        print(f"  Enable Dynamic Graph Mode (FiGraph: {args.start_year}-{args.end_year})")
        
        trainer.create_model()
        
        history = trainer.train()
        
        print(f"\nTraining Completed!")
        print(f"\nZero-Shot Learning Summary:")
        print(f"   Training Set: DGraphFin ")
        print(f"   Test Set: FiGraph-{args.start_year}-{args.end_year}")
        print(f"   Outputs: {trainer.output_dir}")
        
    except Exception as e:
        print(f"\nError during training:")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
