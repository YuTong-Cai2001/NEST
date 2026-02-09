import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union
import scipy.sparse as sp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import warnings
warnings.filterwarnings('ignore')


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) - Captures temporal evolution patterns of nodes
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 kernel_size: int = 3, dropout: float = 0.2):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions [64, 128, 256]
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super(TemporalConvNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        
        layers = []
        num_inputs = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            dilation = 2 ** i  # Exponentially growing dilation rate
            layers.append(
                TemporalBlock(
                    n_inputs=num_inputs,
                    n_outputs=hidden_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )
            num_inputs = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        # Add input dimension adapter (created at runtime)
        self.input_adapter = None
        
        
    def _create_input_adapter(self, actual_input_dim: int):
        """Dynamically create input adapter"""
        if actual_input_dim != self.input_dim:
            self.input_adapter = nn.Linear(actual_input_dim, self.input_dim).to(next(self.parameters()).device)
            return True
        return False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - Adaptive input dimension version
        
        Args:
            x: Temporal features (batch_size, actual_input_dim, seq_len) - Already converted to correct format externally
            
        Returns:
            Temporal representation (batch_size, output_dim, seq_len) - Maintains TCN native output format
        """
        batch_size, actual_input_dim, seq_len = x.shape
        
        if actual_input_dim != self.input_dim:
            if self.input_adapter is None:
                self._create_input_adapter(actual_input_dim)
            
            # Dimension conversion: (batch, actual_dim, seq) -> (batch, seq, actual_dim) -> adapter -> (batch, seq, expected_dim) -> (batch, expected_dim, seq)
            x = x.permute(0, 2, 1)  # (batch, seq, actual_dim)
            x = self.input_adapter(x)  # (batch, seq, expected_dim) 
            x = x.permute(0, 2, 1)  # (batch, expected_dim, seq)
        
        x = self.network(x)    # (batch_size, output_dim, seq_len)
        
        return x


class TemporalBlock(nn.Module):
    """Basic building block of TCN"""
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        """Weight initialization"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove future information after convolution to maintain causality"""
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network Encoder - Captures spatial structure information of graphs
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 gnn_type: str = 'gcn', dropout: float = 0.2):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            gnn_type: GNN type ('gcn', 'sage', 'gat')
            dropout: Dropout rate
        """
        super(GraphNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        
        layers = []
        num_inputs = input_dim
        
        for hidden_dim in hidden_dims:
            if self.gnn_type == 'gcn':
                layers.append(GCNConv(num_inputs, hidden_dim))
            elif self.gnn_type == 'sage':
                layers.append(SAGEConv(num_inputs, hidden_dim))
            elif self.gnn_type == 'gat':
                layers.append(GATConv(num_inputs, hidden_dim, heads=1, concat=False))
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
            
            num_inputs = hidden_dim
        
        self.gnn_layers = nn.ModuleList(layers)
        self.dropout_layer = nn.Dropout(dropout)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            
        Returns:
            Node representations (num_nodes, output_dim)
        """
        h = x
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            h = gnn_layer(h, edge_index)
            
            if i < len(self.gnn_layers) - 1:  
                h = F.relu(h)
                h = self.dropout_layer(h)
        
        return h


class ProjectionHead(nn.Module):
    """
    Learnable Projection Head - Projects high-dimensional representations to low-dimensional space for residual computation
    """
    
    def __init__(self, input_dim: int, projection_dim: int, 
                 hidden_dim: Optional[int] = None):
        """
        Args:
            input_dim: Input feature dimension
            projection_dim: Dimension after projection
            hidden_dim: Hidden layer dimension (if None, use single-layer projection)
        """
        super(ProjectionHead, self).__init__()
        
        if hidden_dim is None:
            # Single-layer projection
            self.projection = nn.Linear(input_dim, projection_dim)
        else:
            # Two-layer MLP projection
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, projection_dim)
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input representations (..., input_dim)
            
        Returns:
            Projected representations (..., projection_dim)
        """
        return self.projection(x)


class ResidualComputationModule(nn.Module):
    """
    Residual Computation Module - Computes temporal residuals and spatial-temporal residuals in projection space
    """
    
    def __init__(self, projection_dim: int):
        """
        Args:
            projection_dim: Projection space dimension
        """
        super(ResidualComputationModule, self).__init__()
        
        self.projection_dim = projection_dim
        
        # Temporal residual transformation layer
        self.temporal_transform = nn.Linear(projection_dim, projection_dim)
        
        # Spatial-temporal interaction transformation layer
        self.spatial_temporal_transform = nn.Linear(projection_dim, projection_dim)
        
        # Residual fusion layer
        self.residual_fusion = nn.Linear(projection_dim * 2, projection_dim)
        
        self.init_weights()
        
    def init_weights(self):
        """Weight initialization"""
        nn.init.xavier_uniform_(self.temporal_transform.weight)
        nn.init.xavier_uniform_(self.spatial_temporal_transform.weight)
        
    def forward(self, h_temp_current: torch.Tensor, h_temp_prev: torch.Tensor,
                h_spatial_current: torch.Tensor, h_spatial_prev: torch.Tensor) -> torch.Tensor:
        """
        Compute learnable temporal residuals (vectorized version)
        
        Args:
            h_temp_current: Current timestep temporal representation (..., projection_dim)
            h_temp_prev: Previous timestep temporal representation (..., projection_dim) 
            h_spatial_current: Current timestep spatial representation (..., projection_dim)
            h_spatial_prev: Previous timestep spatial representation (..., projection_dim)
            
        Returns:
            Fused residual representation (..., projection_dim)
        """
        # Temporal residual computation (supports batch processing of any dimension)
        temporal_diff = h_temp_current - h_temp_prev
        temporal_residual = self.temporal_transform(temporal_diff)
        
        # Spatial-temporal interaction residual computation
        spatial_diff = h_spatial_current - h_spatial_prev
        spatial_temporal_interaction = self.spatial_temporal_transform(
            temporal_diff * spatial_diff  # Element-wise product to capture interaction
        )
        
        # Fuse both residuals
        combined_residual = torch.cat([temporal_residual, spatial_temporal_interaction], dim=-1)
        fused_residual = self.residual_fusion(combined_residual)
        
        return fused_residual


class HierarchicalFusionNetwork(nn.Module):
    """
    Hierarchical Fusion Network - Fuses multiple information sources
    """
    
    def __init__(self, residual_dim: int, spatial_dim: int, 
                 feature_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            residual_dim: Residual feature dimension
            spatial_dim: Spatial feature dimension
            feature_dim: Original feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
        """
        super(HierarchicalFusionNetwork, self).__init__()
        
        input_dim = residual_dim + spatial_dim + feature_dim
        
        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (vectorized version)
        
        Args:
            combined_features: Concatenated features (..., total_feature_dim)
                             Can be (num_nodes, total_feature_dim) or 
                             (batch_size, num_nodes, total_feature_dim)
            
        Returns:
            Fused representation (..., output_dim)
        """
        # Pass through fusion network (supports batch processing of any dimension)
        fused_representation = self.fusion_network(combined_features)
        
        return fused_representation


class AnomalyGFMModel(nn.Module):
    """
    AnomalyGFM Temporal Graph Neural Network Model (TGN Enhanced Version)
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Model configuration dictionary
        """
        super(AnomalyGFMModel, self).__init__()
        
        self.config = config
        
        # Basic parameters
        self.feature_dim = config['feature_dim']
        self.sequence_length = config['sequence_length']
        self.projection_dim = config['projection_dim']
        self.hidden_dim = config['hidden_dim']
        
        # TCN Temporal Encoder - Adaptive input dimension
        tcn_input_dim = self.feature_dim  # Use actual feature dimension instead of preset value
        self.tcn_encoder = TemporalConvNet(
            input_dim=tcn_input_dim,
            hidden_dims=config['tcn_hidden_dims'],
            kernel_size=config.get('tcn_kernel_size', 3),
            dropout=config.get('dropout', 0.2)
        )
        
        print(f"TCN Encoder Config: input_dim={tcn_input_dim}, hidden_dims={config['tcn_hidden_dims']}")
        
        # GNN Spatial Encoder - Adaptive input dimension version
        gnn_hidden_dims = config['gnn_hidden_dims']
        gnn_input_dim = self.feature_dim  
        self.gnn_encoder = GraphNeuralNetwork(
            input_dim=gnn_input_dim,
            hidden_dims=gnn_hidden_dims,
            gnn_type=config.get('gnn_type', 'gcn'),
            dropout=config.get('dropout', 0.2)
        )
        
        print(f" GNN Encoder Config: input_dim={gnn_input_dim}, hidden_dims={gnn_hidden_dims}")
        
        # Debug: Verify actual input dimension of GNN first layer
        first_layer = self.gnn_encoder.gnn_layers[0]
        
        # Projection heads
        self.temporal_projection = ProjectionHead(
            input_dim=self.tcn_encoder.output_dim,
            projection_dim=self.projection_dim,
            hidden_dim=config.get('projection_hidden_dim')
        )
        
        self.spatial_projection = ProjectionHead(
            input_dim=self.gnn_encoder.output_dim,
            projection_dim=self.projection_dim,
            hidden_dim=config.get('projection_hidden_dim')
        )
        
        # Residual computation module
        self.residual_module = ResidualComputationModule(self.projection_dim)
        
        # Hierarchical fusion network
        self.fusion_network = HierarchicalFusionNetwork(
            residual_dim=self.projection_dim,
            spatial_dim=self.gnn_encoder.output_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            output_dim=config['output_dim']
        )
        
        self.normal_prototype = nn.Parameter(torch.randn(1, config['output_dim']))
        self.abnormal_prototype = nn.Parameter(torch.randn(1, config['output_dim']))
        
        # Initialize prototype parameters
        nn.init.xavier_uniform_(self.normal_prototype)
        nn.init.xavier_uniform_(self.abnormal_prototype)
        
        # NEW: Learnable temperature parameter for temperature-weighted softmax fusion
        self.score_temperature = nn.Parameter(torch.tensor(5.0))
        
    def forward(self, temporal_sequences: torch.Tensor,
                adjacency_matrices: Union[List[torch.Tensor], List[Dict]],
                node_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            temporal_sequences: Temporal feature sequences (batch_size, seq_len, num_nodes, feature_dim)
            adjacency_matrices: Dynamic adjacency matrix sequences [adj_t0, adj_t1, ..., adj_tn]
            node_features: Current timestep node features (num_nodes, feature_dim)
            
        Returns:
            Dictionary containing dynamic graph representations and anomaly scores
        """
        batch_size, seq_len, num_nodes, feature_dim = temporal_sequences.shape
        
        
        structural_evolution = self._compute_structural_evolution(adjacency_matrices)
        temporal_input = temporal_sequences.view(-1, seq_len, feature_dim)  
        temporal_input = temporal_input.permute(0, 2, 1)  
        
        temporal_representations = self.tcn_encoder(temporal_input)  
        
        tcn_output_dim = temporal_representations.shape[1]
        temporal_representations = temporal_representations.permute(0, 2, 1)  
        temporal_representations = temporal_representations.reshape(batch_size, num_nodes, seq_len, tcn_output_dim)
        temporal_representations = temporal_representations.permute(0, 2, 1, 3)  

        spatial_representations = self._temporal_gnn_encoding(
            temporal_sequences, adjacency_matrices, structural_evolution, 
            batch_size, seq_len, num_nodes
        )
        
        temporal_projected = self.temporal_projection(temporal_representations)
        spatial_projected = self.spatial_projection(spatial_representations)
        
        if seq_len >= 2:
            h_temp_current = temporal_projected[:, -1]     # (batch_size, num_nodes, projection_dim)
            h_temp_prev = temporal_projected[:, -2]        # (batch_size, num_nodes, projection_dim)
            h_spatial_current = spatial_projected[:, -1]   # (batch_size, num_nodes, projection_dim)
            h_spatial_prev = spatial_projected[:, -2]      # (batch_size, num_nodes, projection_dim)
            
            # Vectorized residual computation (process entire batch at once)
            residual_features = self.residual_module(
                h_temp_current, h_temp_prev,
                h_spatial_current, h_spatial_prev
        else:
            # If sequence length is insufficient, use zero residuals
            residual_features = torch.zeros(batch_size, num_nodes, self.projection_dim, 
                                          device=temporal_sequences.device)
        
        current_spatial = spatial_representations[:, -1]  # (batch_size, num_nodes, gnn_output_dim)
        current_features = temporal_sequences[:, -1]      # (batch_size, num_nodes, feature_dim)
        
        # Vectorized fusion: directly concatenate and process on batch dimension
        combined_features = torch.cat([
            residual_features,   # (batch_size, num_nodes, projection_dim)
            current_spatial,     # (batch_size, num_nodes, gnn_output_dim)
            current_features     # (batch_size, num_nodes, feature_dim)
        ], dim=-1)  # (batch_size, num_nodes, total_feature_dim)
        
        # Fusion processing for entire batch at once
        fused_representations = self.fusion_network(combined_features)  # (batch_size, num_nodes, output_dim)
        
        batch_size, num_nodes, feature_dim = fused_representations.shape
        representations_flat = fused_representations.view(-1, feature_dim)  # (batch_size * num_nodes, feature_dim)
        
        # Normalization
        representations_norm = F.normalize(representations_flat, p=2, dim=1)
        normal_prototype_norm = F.normalize(self.normal_prototype, p=2, dim=1)

        abnormal_prototype_norm = F.normalize(self.abnormal_prototype, p=2, dim=1)
        
        sim_to_normal = torch.mm(representations_norm, normal_prototype_norm.t()).squeeze(1)
        sim_to_abnormal = torch.mm(representations_norm, abnormal_prototype_norm.t()).squeeze(1)
        
        distance_scores_flat = 1 - sim_to_normal
        
        contrast_scores_flat = sim_to_abnormal - sim_to_normal

        alpha = 0.6  
        beta = 0.4   
        anomaly_scores_flat = alpha * distance_scores_flat + beta * contrast_scores_flat
        
        margin_scores_flat = sim_to_abnormal - sim_to_normal
        

        anomaly_scores = anomaly_scores_flat.view(batch_size, num_nodes)  # (batch_size, num_nodes)
        sim_to_normal_reshaped = sim_to_normal.view(batch_size, num_nodes)
        sim_to_abnormal_reshaped = sim_to_abnormal.view(batch_size, num_nodes)
        margin_scores = margin_scores_flat.view(batch_size, num_nodes)
        

        return {
            'fused_representations': fused_representations,
            'anomaly_scores': anomaly_scores,  
            'sim_to_normal': sim_to_normal_reshaped,      
            'sim_to_abnormal': sim_to_abnormal_reshaped,  
            'margin_scores': margin_scores,               
            'temporal_representations': temporal_representations,
            'spatial_representations': spatial_representations,
            'temporal_projected': temporal_projected,
            'spatial_projected': spatial_projected,
            'residual_features': residual_features,
            'normal_prototype': self.normal_prototype,
            'abnormal_prototype': self.abnormal_prototype
        }
    
    def _compute_structural_evolution(self, adjacency_matrices: Union[List[torch.Tensor], List[Dict]]) -> Dict[str, torch.Tensor]:
        """
        Compute graph structural evolution features - TGN key component
        """
        seq_len = len(adjacency_matrices)
        
        if seq_len < 2:
            # If only one timestep, return zero evolution
            return {'evolution_features': torch.zeros(1), 'has_evolution': False}
        
        edge_changes = []  # Record edge changes
        degree_changes = []  # Record degree changes
        
        for t in range(1, seq_len):
            prev_adj = adjacency_matrices[t-1]
            curr_adj = adjacency_matrices[t]
            
            # Extract edge_index (compatible with sparse dict and dense tensor)
            if isinstance(prev_adj, dict) and 'edge_index' in prev_adj:
                prev_edges = prev_adj['edge_index']
                curr_edges = curr_adj['edge_index']
            else:
                # Handle dense tensor (backward compatibility)
                prev_edges, _ = dense_to_sparse(prev_adj)
                curr_edges, _ = dense_to_sparse(curr_adj)
            
            # Compute edge set changes
            prev_edge_set = set(map(tuple, prev_edges.t().tolist()))
            curr_edge_set = set(map(tuple, curr_edges.t().tolist()))
            
            # Count changes
            new_edges = len(curr_edge_set - prev_edge_set)
            removed_edges = len(prev_edge_set - curr_edge_set)
            edge_changes.append([new_edges, removed_edges])
            
            # Compute degree distribution changes (memory-safe version) - Avoid CUDA bincount errors
            try:
                prev_max_node = prev_edges[0].max().item() if prev_edges[0].numel() > 0 else 0
                curr_max_node = curr_edges[0].max().item() if curr_edges[0].numel() > 0 else 0
                max_node_id = max(prev_max_node, curr_max_node)
                
                if max_node_id > 100000:  
                    degree_change = abs(curr_edges[0].numel() - prev_edges[0].numel()) / 1000.0
                else:
                    prev_degrees = torch.bincount(prev_edges[0].clamp(0, max_node_id))
                    curr_degrees = torch.bincount(curr_edges[0].clamp(0, max_node_id))
                    
                    max_len = max(len(prev_degrees), len(curr_degrees))
                    if len(prev_degrees) < max_len:
                        prev_degrees = F.pad(prev_degrees, (0, max_len - len(prev_degrees)))
                    if len(curr_degrees) < max_len:
                        curr_degrees = F.pad(curr_degrees, (0, max_len - len(curr_degrees)))
                    
                    degree_change = torch.norm(curr_degrees.float() - prev_degrees.float()).item()
                    
                degree_changes.append(degree_change)
                
            except Exception as e:
                degree_changes.append(abs(curr_edges[0].numel() - prev_edges[0].numel()) / 1000.0)
        
        # Aggregate evolution features
        if edge_changes:
            edge_evolution = torch.tensor(edge_changes).float()  # (seq_len-1, 2)
            degree_evolution = torch.tensor(degree_changes).float()  # (seq_len-1,)
            
            evolution_features = torch.cat([
                edge_evolution.flatten(),  # Edge change features
                degree_evolution          # Degree change features
            ], dim=0)
        else:
            evolution_features = torch.zeros(4)  # Default dimension
        
        return {
            'evolution_features': evolution_features,
            'edge_changes': edge_changes,
            'degree_changes': degree_changes,
            'has_evolution': len(edge_changes) > 0
        }
    
    def _temporal_gnn_encoding(self, temporal_sequences: torch.Tensor,
                              adjacency_matrices: Union[List[torch.Tensor], List[Dict]],
                              structural_evolution: Dict[str, torch.Tensor],
                              batch_size: int, seq_len: int, num_nodes: int) -> torch.Tensor:
        """
        Temporal GNN encoding - TGN core upgrade
        """
        spatial_representations = []
        
        gnn_output_dim = self.gnn_encoder.output_dim  
        print(f" TGN Memory Initialization: gnn_output_dim = {gnn_output_dim}")
        node_memory = torch.zeros(batch_size, num_nodes, gnn_output_dim, 
                                device=temporal_sequences.device)
        
        for t in range(seq_len):
            if t >= len(adjacency_matrices):
                print(f" Warning: Timestep {t} exceeds adjacency_matrices length {len(adjacency_matrices)}, skipping")
                continue
                
            adj_info = adjacency_matrices[t]
            
            if isinstance(adj_info, dict) and 'edge_index' in adj_info:
                edge_index = adj_info['edge_index']
                edge_attr = adj_info.get('edge_attr', None)
                
                
            else:
                edge_index, edge_attr = dense_to_sparse(adj_info)
                warnings.warn(" TGN recommends using sparse input for better efficiency")
            
            node_features_t = temporal_sequences[:, t]  # (batch_size, num_nodes, feature_dim)
            
            batch_edge_indices = []
            batch_node_features = []
            batch_memory_features = []
            
            for b in range(batch_size):
                offset_edge_index = edge_index + b * num_nodes
                
                max_expected_node = (b + 1) * num_nodes - 1
                if offset_edge_index.numel() > 0:
                    max_actual_node = offset_edge_index.max().item()
                    if max_actual_node > batch_size * num_nodes - 1:
                        valid_mask = offset_edge_index < batch_size * num_nodes
                        offset_edge_index = offset_edge_index.masked_select(valid_mask.all(dim=0, keepdim=True)).reshape(2, -1)
                
                batch_edge_indices.append(offset_edge_index)
                
                # Current features
                current_features = node_features_t[b]  # (num_nodes, feature_dim)
                
                if t > 0:
                    memory_features = node_memory[b]  # (num_nodes, gnn_output_dim)
                    if not hasattr(self, 'memory_projector'):
                        self.memory_projector = nn.Linear(gnn_output_dim, current_features.shape[-1]).to(current_features.device)
                    
                    projected_memory = self.memory_projector(memory_features)
                    enhanced_features = current_features + projected_memory
                else:
                    enhanced_features = current_features
                
                batch_node_features.append(enhanced_features)
                batch_memory_features.append(node_memory[b])
            
            combined_edge_index = torch.cat(batch_edge_indices, dim=1)
            combined_node_features = torch.cat(batch_node_features, dim=0)

            combined_spatial_rep = self.gnn_encoder(combined_node_features, combined_edge_index)

            spatial_rep_t = combined_spatial_rep.view(batch_size, num_nodes, -1)

            if t > 0 and structural_evolution['has_evolution']:

                evolution_strength = torch.norm(structural_evolution['evolution_features']).item()
                memory_decay = max(0.1, 1.0 - evolution_strength * 0.1)  
                
                node_memory = memory_decay * node_memory + (1 - memory_decay) * spatial_rep_t
            else:

                node_memory = 0.8 * node_memory + 0.2 * spatial_rep_t
            
            spatial_representations.append(spatial_rep_t)
        
        return torch.stack(spatial_representations, dim=1)  # (batch_size, seq_len, num_nodes, gnn_output_dim)


class PrototypeContrastiveLoss(nn.Module):
    """
    Prototype Contrastive Learning Loss - Core of Zero-Shot Learning
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 0.5,
                 normal_weight: float = 1.0, abnormal_weight: float = 1.0, 
                 separation_weight: float = 0.5):
        """
        Args:
            temperature: Temperature parameter controlling contrastive learning strictness
            margin: Margin parameter for pushing abnormal samples away from normal prototype
            normal_weight: Loss weight for normal samples
            abnormal_weight: Loss weight for abnormal samples
            separation_weight: Prototype separation loss weight
        """
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.normal_weight = normal_weight
        self.abnormal_weight = abnormal_weight
        self.separation_weight = separation_weight
        
    def forward(self, representations: torch.Tensor, 
                labels: torch.Tensor,
                normal_prototype: torch.Tensor,
                abnormal_prototype: torch.Tensor):
        """
        Compute prototype contrastive loss
        
        Args:
            representations: Node representations (batch_size, num_nodes, feature_dim)
            labels: Labels (batch_size, num_nodes) - 0 for normal, 1 for abnormal
            normal_prototype: Normal prototype (1, feature_dim)
            abnormal_prototype: Abnormal prototype (1, feature_dim)
            
        Returns:
            loss: Total loss
            metrics: Dictionary containing various metrics
        """
        device = representations.device
        
        if representations.dim() == 3:
            batch_size, num_nodes, feature_dim = representations.shape
            representations_flat = representations.view(-1, feature_dim)
            labels_flat = labels.view(-1)
        elif representations.dim() == 2:
            representations_flat = representations
            labels_flat = labels
            feature_dim = representations.shape[1]
        else:
            raise ValueError(f"Expected representations to have 2 or 3 dimensions, got {representations.dim()}")
        
        representations_norm = F.normalize(representations_flat, p=2, dim=1)
        normal_prototype_norm = F.normalize(normal_prototype, p=2, dim=1)
        abnormal_prototype_norm = F.normalize(abnormal_prototype, p=2, dim=1)
        
        sim_to_normal = torch.mm(representations_norm, normal_prototype_norm.t()).squeeze(1)  # (N,)
        sim_to_abnormal = torch.mm(representations_norm, abnormal_prototype_norm.t()).squeeze(1)  # (N,)
        
        dist_to_normal = 1 - sim_to_normal
        dist_to_abnormal = 1 - sim_to_abnormal
        

        normal_mask = (labels_flat == 0)
        abnormal_mask = (labels_flat == 1)
        

        if normal_mask.sum() > 0:
            normal_loss = dist_to_normal[normal_mask].mean()
        else:
            normal_loss = torch.tensor(0.0, device=device, dtype=representations.dtype)
        

        if abnormal_mask.sum() > 0:

            abnormal_to_abnormal_loss = dist_to_abnormal[abnormal_mask].mean()

            abnormal_to_normal_loss = F.relu(self.margin - dist_to_normal[abnormal_mask]).mean()
            abnormal_loss = abnormal_to_abnormal_loss + abnormal_to_normal_loss
        else:
            abnormal_loss = torch.tensor(0.0, device=device, dtype=representations.dtype)
        
        prototype_distance = 1 - torch.mm(normal_prototype_norm, abnormal_prototype_norm.t()).squeeze()
        separation_loss = F.relu(self.margin - prototype_distance)
        

        total_loss = (self.normal_weight * normal_loss + 
                     self.abnormal_weight * abnormal_loss + 
                     self.separation_weight * separation_loss)
        

        metrics = {
            'normal_loss': normal_loss.item() if normal_loss.requires_grad else normal_loss.item(),
            'abnormal_loss': abnormal_loss.item() if abnormal_loss.requires_grad else abnormal_loss.item(),
            'separation_loss': separation_loss.item() if separation_loss.requires_grad else separation_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics


class ContrastiveLoss(PrototypeContrastiveLoss):
    """
    Backward-compatible loss function alias
    """
    
    def __init__(self, temperature: float = 0.1):
        super(ContrastiveLoss, self).__init__(temperature=temperature)
        
    def forward(self, representations: torch.Tensor, 
                labels: torch.Tensor,
                normal_prototype: torch.Tensor,
                abnormal_prototype: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible interface - returns only total loss
        
        Args:
            representations: Node representations
            labels: Labels
            normal_prototype: Normal prototype (required)
            abnormal_prototype: Abnormal prototype (required)
            
        Returns:
            Total loss tensor
        """
        total_loss, metrics = super().forward(representations, labels, normal_prototype, abnormal_prototype)
        return total_loss


def create_model_config(feature_dim: int, sequence_length: int) -> Dict:
    """
    Create model configuration
    
    Args:
        feature_dim: Feature dimension
        sequence_length: Sequence length
        
    Returns:
        Model configuration dictionary
    """
    config = {
        # Basic parameters
        'feature_dim': feature_dim,
        'sequence_length': sequence_length,
        'projection_dim': 64,
        'hidden_dim': 128,
        'output_dim': 64,
        
        # TCN parameters
        'tcn_hidden_dims': [64, 128, 64],
        'tcn_kernel_size': 3,
        
        # GNN parameters  
        'gnn_hidden_dims': [64, 128, 64],
        'gnn_type': 'gcn',
        
        # Training parameters
        'dropout': 0.2,
        'projection_hidden_dim': 32,
        'temperature': 0.1,
    }
    
    return config
