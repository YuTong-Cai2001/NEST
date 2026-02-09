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
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import AnomalyGFMDataProcessor
from anomaly_gfm_complete_model import AnomalyGFMModel, ContrastiveLoss, PrototypeContrastiveLoss, create_model_config
from sparse_graph_utils import SparseGraphProcessor, convert_dataset_to_sparse
from model_adapter import create_sparse_model_adapter
import scipy.sparse as sp



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
                print(f"   Detected non-binary label values {unique_vals.tolist()}, will binarize at dataset stage (>0 as 1)")
            self.labels = (labels > 0).astype(np.float32)
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
        temporal_sequence = self.temporal_sequences[idx]
        adjacency_sequence = self.temporal_adjacencies[idx]
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
        
        # Training parameters
        self.num_epochs = config.get('num_epochs', 100)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.batch_size = config.get('batch_size', 4)
        self.sequence_length = config.get('sequence_length', 10)
        
        self.feature_dim = config.get('feature_dim', 17)  
        
        self.contrastive_weight = config.get('contrastive_weight', 1.0)
        self.anomaly_weight = config.get('anomaly_weight', 1.0)
        
        self.sparse_processor = SparseGraphProcessor(device=str(self.device))
        
        print(f"AnomalyGFM Trainer initialized:")
        print(f"   - Device: {self.device}")
        print(f"   - Number of epochs: {self.num_epochs}")
        print(f"   - Learning rate: {self.learning_rate}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Sequence length: {self.sequence_length}")
        print(f"   - Integrated sparse graph optimization")

    def prepare_data(self, target_year: int = 2014, mode: str = 'static'):
        """
        Unified data preparation method - supports dynamic and static graph modes
        
        Args:
            target_year: FiGraph target domain year (static mode)
            mode: 'static' - traditional static graph mode, 'dynamic' - new dynamic graph mode
        """
        
        if mode == 'dynamic':
            return self._prepare_dynamic_data()
        else:
            return self._prepare_static_data(target_year)
    

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
            'target_dim': self.feature_dim,  
            'use_svd': True,
            'dgraphfin_path': self.config.get('dgraphfin_path', 'data/DGraphFin2/'),
            'figraph_path': self.config.get('figraph_path', 'data/FiGraph-main/data/'),
            'cache_dir': 'cache/',
            'enable_cache': True  
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
        
        # Step 1: Collect reference features from source and target domains
        dgraphfin_ref_features = dgraphfin_data['node_features']  
        figraph_ref_features = figraph_temporal['temporal_sequences'][0][0]  
        
        
        # Step 2: Compute feature alignment transformation (one-time calculation)
        dgraphfin_aligned, figraph_aligned = processor.align_features(
            dgraphfin_ref_features,
            figraph_ref_features
        )
        
        self.feature_dim = dgraphfin_aligned.shape[1]
        
        # Step 3: Apply alignment transformation to all FiGraph time steps
        for batch_idx in range(len(figraph_temporal['temporal_sequences'])):
            for time_idx in range(len(figraph_temporal['temporal_sequences'][batch_idx])):
                current_features = figraph_temporal['temporal_sequences'][batch_idx][time_idx]
                
                _, aligned_features = processor.align_features(
                    dgraphfin_ref_features,  
                    current_features         
                )
                
                figraph_temporal['temporal_sequences'][batch_idx][time_idx] = aligned_features
        
        
        # Step 4: Update DGraphFin features (ensure using aligned features during training)
        dgraphfin_data['node_features'] = dgraphfin_aligned
        
        train_dataset = TemporalGraphDataset(
            dgraphfin_temporal, 
            self.sequence_length,
            labels=dgraphfin_data['labels']
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
            'features': dgraphfin_aligned,
            'labels': dgraphfin_data['labels'],
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
        self.anomaly_loss_fn = nn.BCELoss()
        
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
        Unified zero-shot evaluation function - supports static and dynamic graph modes
        """
        test_mode = self.test_data.get('mode', 'static')
        
        if test_mode == 'dynamic':
            result = self._evaluate_dynamic_zero_shot()
        else:
            result = self._evaluate_static_zero_shot()
        
        
        clear_gpu_memory()
        
        return result
    

    def _evaluate_dynamic_zero_shot(self) -> Dict[str, float]:
        """
        Dynamic graph zero-shot evaluation - performing true dynamic graph prediction on FiGraph dynamic sequences
        """
        
        self.model.eval()
        
        if self.test_data.get('mode') != 'dynamic':
            return self.evaluate_zero_shot()
        
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
                
                
                if 'margin_scores' in outputs:
                    anomaly_scores = outputs['margin_scores'].cpu().numpy().flatten()
                elif 'anomaly_scores' in outputs:
                    anomaly_scores = outputs['anomaly_scores'].cpu().numpy().flatten()
                elif 'fused_representations' in outputs and 'normal_prototype' in outputs:
                    fused_representations = outputs['fused_representations']  # (batch_size, num_nodes, output_dim)
                    normal_prototype = outputs['normal_prototype']            # (1, output_dim)
                    
                    batch_size, num_nodes, feature_dim = fused_representations.shape
                    representations_flat = fused_representations.view(-1, feature_dim)  # (batch_size * num_nodes, feature_dim)
                    
                    # Normalization
                    import torch.nn.functional as F
                    representations_norm = F.normalize(representations_flat, p=2, dim=1)
                    normal_prototype_norm = F.normalize(normal_prototype, p=2, dim=1)
                    
                    sim_to_normal = torch.mm(representations_norm, normal_prototype_norm.t()).squeeze(1)
                    
                    temperature = getattr(self.model, 'score_temperature', torch.tensor(0.5)).item()
                    
                    temp_adjusted_sim = sim_to_normal / temperature
                    anomaly_scores = (1 - torch.sigmoid(temp_adjusted_sim)).cpu().numpy()  
                else:
                    raise ValueError(f"Model output format not supported. Available keys: {list(outputs.keys())}")
                
                
                if test_labels is not None:
                    if len(test_labels) == len(anomaly_scores):
                        eval_results = AnomalyGFMDataProcessor.evaluate_ranking_performance(
                            test_labels, anomaly_scores, "DGraphFin-Dynamic-ZeroShot"
                        )
                        
                        return {
                            'dynamic_auc': eval_results['roc_auc'],
                            'dynamic_ap': eval_results['average_precision'],
                            'num_samples': len(anomaly_scores),
                            'mode': 'dynamic_graph_prediction',
                            'sequence_length': seq_len,
                            'temporal_coverage': f'{sequence_length} years sequence'
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
                print(f"Dynamic graph zero-shot inference error: {e}")
                import traceback
                traceback.print_exc()
                return {'dynamic_auc': 0.0, 'dynamic_ap': 0.0, 'mode': 'failed'}
    
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
        
        best_auc = 0.0
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
            
            # Evaluate every 10 epochs and the last epoch
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                eval_metrics = self.evaluate_zero_shot()
                current_auc = eval_metrics.get('auc', 0.0)
                
                # Save best model
                if current_auc > best_auc:
                    best_auc = current_auc
                    self.save_model(f'best_model_auc_{best_auc:.4f}.pth')
                
                # Learning rate scheduling
                self.scheduler.step(train_metrics['total_loss'])
            else:
                eval_metrics = {}
                current_auc = 0.0
            
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
        print(f"   - Best AUC: {best_auc:.4f}")
        print(f"   - Total Training Time: {sum(h['time'] for h in train_history):.2f} seconds")
        if turning_point_model_info:
            print(f"   - Turning Point Detected: Epoch {turning_point_model_info['epoch']} (Batch {turning_point_model_info.get('batch_idx', 0)})")
        else:
            print(f"   - No Turning Point Detected")
        
        turning_point_result = None
        if turning_point_model_info:
            print(f"\nStarting Turning Point Model Evaluation...")
            print(f"=" * 60)
            
            print(f"\n   Evaluating Turning Point Model (Epoch {turning_point_model_info['epoch']}, Batch {turning_point_model_info.get('batch_idx', 0)}):")
            print(f"      Description: {turning_point_model_info['description']}")
            print(f"      Temporal Loss: {turning_point_model_info['consistency_loss']:.6f}")
            
            model_path = os.path.join('zero_shot/checkpoints', turning_point_model_info['filename'])
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Use same evaluation function and test set for evaluation
                    eval_metrics = self.evaluate_zero_shot()
                    tp_auc = eval_metrics.get('auc', eval_metrics.get('dynamic_auc', 0.0))
                    tp_ap = eval_metrics.get('ap', eval_metrics.get('dynamic_ap', 0.0))
                    
                    turning_point_result = {
                        'epoch': turning_point_model_info['epoch'],
                        'batch_idx': turning_point_model_info.get('batch_idx', 0),
                        'consistency_loss': turning_point_model_info['consistency_loss'],
                        'auc': tp_auc,
                        'ap': tp_ap,
                        'filename': turning_point_model_info['filename'],
                        'description': turning_point_model_info['description']
                    }
                    
                    print(f"      AUC: {tp_auc:.4f}, AP: {tp_ap:.4f}")
                except Exception as e:
                    print(f"      Failed to Load Model: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"      Model File Not Found: {model_path}")
            
            if turning_point_result:
                print(f"\nTurning Point Model Evaluation Summary:")
                print(f"=" * 60)
                print(f"   - Epoch: {turning_point_result['epoch']}")
                print(f"   - Batch: {turning_point_result.get('batch_idx', 0)}")
                print(f"   - Temporal Loss: {turning_point_result['consistency_loss']:.6f}")
                print(f"   - AUC: {turning_point_result['auc']:.4f}")
                print(f"   - AP: {turning_point_result['ap']:.4f}")
                print(f"   - Description: {turning_point_result['description']}")
                
        
        history_path = os.path.join('zero_shot/checkpoints', 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump({
                'train_history': train_history,
                'consistency_loss_history': consistency_loss_history,
                'consistency_loss_records': consistency_loss_records,
                'turning_point_model_info': turning_point_model_info,
                'turning_point_result': turning_point_result
            }, f)
        print(f"\nTraining History Saved: {history_path}")
        
        return train_history
    
    def save_model(self, filename: str):
        """Save model"""
        save_path = os.path.join('zero_shot/checkpoints', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config
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
                for t in range(seq_len):
                    if t < len(adjacency_lists):
                        adj_list_t = adjacency_lists[t]  
                        
                        if isinstance(adj_list_t, list):
                            edge_index, batch_info = self.sparse_processor.batch_sparse_matrices(adj_list_t)
                            sparse_adj_info = {
                                'edge_index': edge_index,
                                'batch_info': batch_info,
                                'format': 'batched_sparse'
                            }
                        else:
                            edge_index, edge_attr = self.sparse_processor.sparse_to_pyg_format(adj_list_t)
                            sparse_adj_info = {
                                'edge_index': edge_index,
                                'edge_attr': edge_attr,
                                'format': 'pyg_sparse'
                            }
                        
                        adjacency_info.append(sparse_adj_info)
                    else:
                        if adjacency_info:
                            adjacency_info.append(adjacency_info[-1])
                
                node_features = temporal_batch[:, -1]  # (batch_size, num_nodes, feature_dim)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                outputs = self.model(temporal_batch, adjacency_info, node_features)
                
                fused_representations = outputs['fused_representations']
                normal_prototype = outputs['normal_prototype']
                
                # Compute distance to normal prototype as anomaly scores
                batch_size_curr, num_nodes_curr, feature_dim_curr = fused_representations.shape
                representations_flat = fused_representations.view(-1, feature_dim_curr)
                
                import torch.nn.functional as F
                representations_norm = F.normalize(representations_flat, p=2, dim=1)
                normal_prototype_norm = F.normalize(normal_prototype, p=2, dim=1)
                
                sim_to_normal = torch.mm(representations_norm, normal_prototype_norm.t()).squeeze(1)
                # Ensure anomaly scores are in [0,1] range: use sigmoid to convert similarity to probability
                anomaly_scores = 1 - torch.sigmoid(sim_to_normal)  # Distance as anomaly score, range [0,1]
                anomaly_scores = anomaly_scores.view(batch_size_curr, num_nodes_curr)
                
                # Compute loss
                if labels_batch is not None:
                    # If labels available, use supervised loss
                    valid_mask = (labels_batch >= 0)  # Filter invalid labels
                    if valid_mask.sum() > 0:
                        valid_scores = anomaly_scores[valid_mask]
                        valid_labels = labels_batch[valid_mask].float()
                        
                        # Binary cross-entropy loss
                        anomaly_loss = F.binary_cross_entropy(
                            valid_scores, valid_labels
                        )
                        
                        # Record anomaly detection loss
                        batch_anomaly_loss = anomaly_loss.item()
                        
                        # Single prototype: only pull normal samples closer to normal prototype (simple center loss)
                        representations_for_proto = fused_representations.view(-1, feature_dim_curr)
                        labels_for_proto = labels_batch.view(-1)
                        
                        valid_proto_mask = (labels_for_proto >= 0)
                        if valid_proto_mask.sum() > 0:
                            valid_representations = representations_for_proto[valid_proto_mask]
                            valid_proto_labels = labels_for_proto[valid_proto_mask]
                            
                            normal_mask = (valid_proto_labels == 0)  # Assume 0 is normal
                            if normal_mask.sum() > 0:
                                normal_reps = valid_representations[normal_mask]
                                normal_reps_norm = F.normalize(normal_reps, p=2, dim=1)
                                normal_proto_norm = F.normalize(self.model.base_model.normal_prototype, p=2, dim=1)
                                sim_to_normal = torch.mm(normal_reps_norm, normal_proto_norm.t()).squeeze(1)
                                # Center loss: improve similarity between normal samples and normal prototype
                                prototype_loss = 1 - sim_to_normal.mean()
                            else:
                                prototype_loss = torch.tensor(0.0, device=fused_representations.device)
                        else:
                            prototype_loss = torch.tensor(0.0, device=fused_representations.device)
                        
                        batch_prototype_loss = float(prototype_loss.detach().cpu().item())
                        
                        # Contrastive loss disabled (single prototype)
                        contrastive_loss = torch.tensor(0.0, device=fused_representations.device)
                        batch_contrastive_loss = 0.0
                        
                        # Compute temporal consistency loss (law 1)
                        consistency_loss = self._compute_temporal_consistency_loss(fused_representations)
                        batch_consistency_loss = consistency_loss.item()
                        consistency_series.append({
                            'batch_idx': batch_idx,
                            'value': batch_consistency_loss
                        })
                        
                        # Compute residual regularization loss (law 2)
                        residual_loss = self._compute_residual_regularization_loss(fused_representations)
                        batch_residual_loss = residual_loss.item()
                        
                        # Five loss function combination: core loss + regularization losses
                        total_loss = (
                            1.0 * anomaly_loss +           # Anomaly detection - main task
                            0.3 * prototype_loss +        # Single prototype center loss
                            0.2 * consistency_loss +      # Temporal consistency
                            0.2 * residual_loss           # Residual regularization
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
                print(f"   Batch {batch_idx} training error: {e}")
                import traceback
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
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
    
    
    def _compute_temporal_consistency_loss(self, fused_representations: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss (Rule 1)"""
        
        if len(fused_representations.shape) == 4:
            batch_size, num_temporal_windows, feature_dim, extra_dim = fused_representations.shape
        elif len(fused_representations.shape) == 3:
            batch_size, num_temporal_windows, feature_dim = fused_representations.shape
            extra_dim = 1
        else:
            return torch.tensor(0.0, device=fused_representations.device)
        
        if num_temporal_windows <= 1:
            return torch.tensor(0.0, device=fused_representations.device)
        
        # Flatten features to [batch_size, num_temporal_windows, feature_dim * extra_dim]
        if len(fused_representations.shape) == 4:
            temporal_features = fused_representations.view(batch_size, num_temporal_windows, -1)
        else:
            temporal_features = fused_representations
        
        temporal_diffs = torch.diff(temporal_features, dim=1)  # (batch_size, num_temporal_windows-1, feature_dim)
        
        # Temporal consistency: representations between adjacent time windows should be relatively smooth
        # Compute L2 norm of changes at each time step, then average
        consistency_loss = torch.mean(torch.norm(temporal_diffs, dim=-1))
        
        return consistency_loss
    
    def _compute_residual_regularization_loss(self, fused_representations: torch.Tensor) -> torch.Tensor:
        """Compute residual regularization loss (Rule 2) - Fixed version"""
        # fused_representations shape: [batch_size, num_temporal_windows, feature_dim, extra_dim]
        
        if len(fused_representations.shape) == 4:
            batch_size, num_temporal_windows, feature_dim, extra_dim = fused_representations.shape
        elif len(fused_representations.shape) == 3:
            batch_size, num_temporal_windows, feature_dim = fused_representations.shape
            extra_dim = 1
        else:
            # Return 0 if shape doesn't match expected format
            return torch.tensor(0.0, device=fused_representations.device)
        
        if num_temporal_windows <= 1:
            return torch.tensor(0.0, device=fused_representations.device)
        
        # Flatten features to [batch_size, num_temporal_windows, feature_dim * extra_dim]
        if len(fused_representations.shape) == 4:
            temporal_features = fused_representations.view(batch_size, num_temporal_windows, -1)
        else:
            temporal_features = fused_representations
        

        window_norms = torch.norm(temporal_features, dim=-1)  # (batch_size, num_temporal_windows)
        
        window_variance = torch.var(window_norms, dim=-1)  # (batch_size,)
        
        flat_repr = temporal_features.view(-1, temporal_features.shape[-1])  # (batch_size * num_temporal_windows, feature_dim)
        centered = flat_repr - flat_repr.mean(dim=0)
        cov_matrix = torch.mm(centered.t(), centered) / (flat_repr.shape[0] - 1)
        
        off_diagonal = cov_matrix - torch.diag(torch.diag(cov_matrix))
        independence_loss = torch.sum(off_diagonal ** 2)
        
        residual_loss = torch.mean(window_variance) + 0.1 * independence_loss
        
        return residual_loss


def main():
    parser = argparse.ArgumentParser(description='AnomalyGFM Zero-Shot Anomaly Detection Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=8, help='Temporal sequence length')
    parser.add_argument('--target_year', type=int, default=2014, help='FiGraph target domain year (static mode)')
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'dynamic'],
                       help='Data mode: static=static graph, dynamic=dynamic graph sequence')
    parser.add_argument('--start_year', type=int, default=2018, help='FiGraph start year (dynamic mode)')
    parser.add_argument('--end_year', type=int, default=2022, help='FiGraph end year (dynamic mode)')
    
    args = parser.parse_args()
    
    config = {
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length,
        'weight_decay': 1e-5,
        'contrastive_weight': 1.0,
        'anomaly_weight': 0.5,
        'dgraphfin_path': 'data/DGraphFin2/',
        'figraph_path': 'data/FiGraph-main/data/',
        'mode': args.mode,
        'target_year': args.target_year,
        'start_year': args.start_year,
        'end_year': args.end_year,
        'model_config': {
            'projection_dim': 32,
            'hidden_dim': 64,
            'output_dim': 32,
            'tcn_hidden_dims': [32, 64, 32],  
            'gnn_hidden_dims': [32, 64, 32],  
            'dropout': 0.3,
            'temperature': 0.1
        }
    }
    
    print("AnomalyGFM Zero-Shot Anomaly Detection")
    print("=" * 60)
    print(f"Training Configuration (DGraphFin -> FiGraph-{args.target_year}):")
    for key, value in config.items():
        if key != 'model_config':
            print(f"   - {key}: {value}")
    
    try:
        trainer = AnomalyGFMTrainer(config)
        
        if args.mode == 'dynamic':
            trainer.prepare_data(mode='dynamic')
            print(f"  Enable Dynamic Graph Mode (FiGraph: {args.start_year}-{args.end_year})")
        else:
            trainer.prepare_data(target_year=args.target_year, mode='static')
            print(f"  Use Static Graph Mode (FiGraph: {args.target_year})")
        
        trainer.create_model()
        
        history = trainer.train()
        
        print(f"\nTraining Completed!")
        print(f"\nZero-Shot Learning Summary:")
        print(f"   Training Set: DGraphFin ")
        print(f"   Test Set: FiGraph-{args.target_year} ")
        print(f"   Model saved to zero_shot/checkpoints/")
        
    except Exception as e:
        print(f"\nError during training:")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--dynamic':
        main_dynamic()
    else:
        main()