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
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import AnomalyGFMDataProcessor
from anomaly_gfm_complete_model import AnomalyGFMModel, create_model_config
from sparse_graph_utils import SparseGraphProcessor
import scipy.sparse as sp


class FewShotSupportSet:
    """Few-shot Support Set Builder"""
    
    def __init__(self, num_abnormal: int = 10, num_normal: int = 50):
        """
        Args:
            num_abnormal: Number of abnormal nodes
            num_normal: Number of normal nodes
        """
        self.num_abnormal = num_abnormal
        self.num_normal = num_normal
    
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
        
        
        if num_anomaly_available >= self.num_abnormal:
            np.random.shuffle(abnormal_nodes)
            np.random.shuffle(normal_nodes)
            support_abnormal = abnormal_nodes[:self.num_abnormal]
            support_normal = normal_nodes[:min(self.num_normal, num_normal_available)]
            support_node_ids = np.concatenate([support_abnormal, support_normal])
            strategy = "mixed"
        elif num_anomaly_available >= 5:
            np.random.shuffle(normal_nodes)
            support_abnormal = abnormal_nodes  
            support_normal = normal_nodes[:self.num_normal]
            support_node_ids = np.concatenate([support_abnormal, support_normal])
            strategy = "mixed_all_abnormal"
        else:
            np.random.shuffle(normal_nodes)
            support_node_ids = normal_nodes[:self.num_normal]
            strategy = "normal_only"
        
        print(f"    Support set built:")
        print(f"      - Abnormal nodes: {len(support_node_ids[support_node_ids < len(labels)]) if strategy != 'normal_only' else 0}")
        print(f"      - Normal nodes: {len(support_node_ids) - (len(support_node_ids[support_node_ids < len(labels)]) if strategy != 'normal_only' else 0)}")
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
        
        self.model_path = config.get('model_path', 
            'zero_shot/checkpoints/turning_point_epoch_16_batch_0_consistency_0.305305.pth')
        
        # Fine-tuning parameters
        self.num_abnormal = config.get('num_abnormal', 10)
        self.num_normal = config.get('num_normal', 50)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.num_epochs = config.get('num_epochs', 20)
        self.batch_size = config.get('batch_size', 32)
        
        # Data paths
        self.figraph_path = config.get('figraph_path', 'data/FiGraph-main/data/')
        self.start_year = config.get('start_year', 2018)
        self.end_year = config.get('end_year', 2022)
        
        # Sparse graph processor
        self.sparse_processor = SparseGraphProcessor()
    
    def load_pretrained_model(self) -> AnomalyGFMModel:
        """Load pre-trained model"""
        
        model_config = create_model_config(
            feature_dim=17,  # FiGraph feature dimension
            sequence_length=5  # 5 time windows
        )
        
        # Create model
        model = AnomalyGFMModel(model_config)
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
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
        model.abnormal_prototype.requires_grad = False
        print(f"    Unfreeze normal prototype parameters (single-prototype fine-tuning)")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n    Parameter statistics:")
        print(f"      - Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"      - Frozen parameters: {total_params - trainable_params:,} ({(total_params-trainable_params)/total_params*100:.2f}%)")
    
    def load_test_data(self) -> Tuple[Dict, np.ndarray]:
        """Load test data (FiGraph 2018-2022)"""
        

        data_config = {
            'temporal_window': 7,
            'target_dim': 17,
            'use_svd': True,
            'figraph_path': self.figraph_path,
            'cache_dir': 'cache/',
            'enable_cache': True
        }
        
        processor = AnomalyGFMDataProcessor(data_config)
        
        # Load FiGraph dynamic graph sequence
        figraph_temporal = processor.load_figraph_temporal_sequence(
            start_year=self.start_year,
            end_year=self.end_year
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
    
    def finetune(self, model: AnomalyGFMModel, train_loader: DataLoader):
        """Fine-tuning training"""
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
    
    def evaluate(self, model: AnomalyGFMModel, test_data: Dict, labels: np.ndarray, 
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
        """Run complete fine-tuning workflow"""
        print("=" * 60)
        print(" AnomalyGFM Few-Shot Fine-tuning")
        print("=" * 60)
        
        # Load pre-trained model
        model = self.load_pretrained_model()
        
        # Set parameter freeze/unfreeze
        self.freeze_and_unfreeze_parameters(model)
        
        # Load test data
        test_data, labels = self.load_test_data()
        
        actual_feature_dim = test_data['feature_dim']
        expected_feature_dim = 17  # Pre-trained model's expected dimension
        
        if actual_feature_dim != expected_feature_dim:
            print(f"\n Feature dimension mismatch: actual={actual_feature_dim}, expected={expected_feature_dim}")
            print(f"   Adding feature projection layer: {actual_feature_dim} -> {expected_feature_dim}")
            
            # Create feature projection layer (trainable)
            feature_projection = nn.Linear(actual_feature_dim, expected_feature_dim).to(self.device)
            # Use Xavier initialization
            nn.init.xavier_uniform_(feature_projection.weight)
            nn.init.zeros_(feature_projection.bias)
            
            # Add projection layer to model
            model.feature_projection = feature_projection
            model.actual_feature_dim = actual_feature_dim
            model.expected_feature_dim = expected_feature_dim
            model.use_feature_projection = True
            
            # Unfreeze projection layer parameters (allow fine-tuning)
            for param in feature_projection.parameters():
                param.requires_grad = True
            
            print(f"    Feature projection layer added and unfrozen")
        else:
            model.use_feature_projection = False
            model.actual_feature_dim = actual_feature_dim
            model.expected_feature_dim = expected_feature_dim
        
        # Build support set
        support_set_builder = FewShotSupportSet(
            num_abnormal=self.num_abnormal,
            num_normal=self.num_normal
        )
        support_node_ids, strategy = support_set_builder.build_support_set(
            test_data, labels, last_time_idx=-1
        )
        
        # Create dataset and data loader
        dataset = FewShotDataset(
            test_data, support_node_ids, labels, sequence_length=5
        )
        train_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0,
            collate_fn=few_shot_collate_fn  # Key fix: use custom collate function for sparse matrices
        )
        
        # Fine-tuning training
        model = self.finetune(model, train_loader)
        
        # Evaluation
        results = self.evaluate(model, test_data, labels, support_node_ids)
        
        # Save fine-tuned model
        save_path = self.model_path.replace('.pth', '_finetuned.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'support_node_ids': support_node_ids,
            'strategy': strategy,
            'results': results
        }, save_path)
        print(f"\n Fine-tuned model saved to: {save_path}")
        
        return model, results


def find_latest_checkpoint(checkpoint_dir='zero_shot/checkpoints'):
    """Automatically find the latest checkpoint file"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        return None
    
    # Sort by modification time, return the latest
    checkpoint_paths = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]
    latest_checkpoint = max(checkpoint_paths, key=os.path.getmtime)
    return latest_checkpoint


def main():
    parser = argparse.ArgumentParser(description='AnomalyGFM Few-Shot Fine-tuning')
    
    # Automatically find the latest checkpoint as default
    default_model_path = find_latest_checkpoint()
    if default_model_path is None:
        default_model_path = 'zero_shot/checkpoints/turning_point_epoch_16_batch_0_consistency_0.305305.pth'
    
    parser.add_argument('--model_path', type=str, default=default_model_path,
        help=f'Path to pre-trained model (default: auto-find latest checkpoint)')
    parser.add_argument('--start_year', type=int, default=2018, help='Start year (default: 2018)')
    parser.add_argument('--end_year', type=int, default=2022, help='End year (default: 2022)')
    parser.add_argument('--num_abnormal', type=int, default=10, help='Number of abnormal nodes (default: 10)')
    parser.add_argument('--num_normal', type=int, default=50, help='Number of normal nodes (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=1e-4 , help='Learning rate (default: 1e-4)')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--figraph_path', type=str, default='data/FiGraph-main/data/',
        help='FiGraph data path (default: data/FiGraph-main/data/)')
    
    args = parser.parse_args()
    
    # Print used configuration
    print("=" * 60)
    print(" Fine-tuning Configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Time range: {args.start_year}-{args.end_year}")
    print(f"  Support set: {args.num_abnormal} abnormal + {args.num_normal} normal")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
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
        'figraph_path': args.figraph_path
    }
    
    # Run fine-tuning
    finetuner = FewShotFinetuner(config)
    model, results = finetuner.run()
    
    print("\n" + "=" * 60)
    print(" Few-shot fine-tuning completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

