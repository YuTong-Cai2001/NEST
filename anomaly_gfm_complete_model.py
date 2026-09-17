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

        # The spatial memory projection must be registered before the optimizer is
        # created.  The previous implementation created this layer lazily in
        # forward(), so its parameters were neither optimized nor reliably loaded
        # from checkpoints.
        self.memory_projector = nn.Linear(
            self.gnn_encoder.output_dim, self.feature_dim
        )

        self.prototype_mode = config.get('prototype_mode', 'single')
        if self.prototype_mode not in {'single', 'dual'}:
            raise ValueError(
                f"prototype_mode must be 'single' or 'dual', got {self.prototype_mode!r}"
            )

        self.normal_prototype = nn.Parameter(torch.randn(1, config['output_dim']))
        nn.init.xavier_uniform_(self.normal_prototype)

        # The full SPAD model is strictly single-prototype.  A dual-prototype
        # parameter is created only for the explicit ablation requested from the
        # command line.
        if self.prototype_mode == 'dual':
            self.abnormal_prototype = nn.Parameter(
                torch.randn(1, config['output_dim'])
            )
            nn.init.xavier_uniform_(self.abnormal_prototype)
        else:
            self.register_parameter('abnormal_prototype', None)

        # Convert a cosine distance into a supervised binary logit.  Ranking
        # metrics use the raw distance; BCE uses this calibrated logit.
        self.score_scale_raw = nn.Parameter(
            torch.tensor(float(config.get('score_scale_init', 5.0)))
        )
        self.score_threshold = nn.Parameter(
            torch.tensor(float(config.get('score_threshold_init', 0.5)))
        )
        self.prototype_score_weight_raw = nn.Parameter(
            torch.tensor(float(config.get('prototype_score_weight_init', 0.5)))
        )
        self.temporal_score_weight_raw = nn.Parameter(
            torch.tensor(float(config.get('temporal_score_weight_init', -1.0)))
        )
        self.temporal_score_threshold = nn.Parameter(
            torch.tensor(float(config.get('temporal_score_threshold_init', 0.0)))
        )
        self.use_consensus_prototype = bool(
            config.get('use_consensus_prototype', True)
        )
        self.use_transport_residual = bool(
            config.get('use_transport_residual', True)
        )
        self.use_local_evolution_score = bool(
            config.get(
                'use_local_evolution_score',
                config.get('use_global_local_fusion', True),
            )
        )
        self.consensus_ratio = float(config.get('consensus_ratio', 0.7))
        self.attribute_descriptor_dim = int(
            config.get('attribute_descriptor_dim', 16)
        )
        
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
        # [B, T, N, F] -> [B*N, F, T].  A direct view would mix the
        # time and node axes in memory.
        temporal_input = temporal_sequences.permute(0, 2, 1, 3).contiguous()
        temporal_input = temporal_input.view(batch_size * num_nodes, seq_len, feature_dim)
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
            residual_changes = self.residual_module(
                temporal_projected[:, 1:], temporal_projected[:, :-1],
                spatial_projected[:, 1:], spatial_projected[:, :-1]
            )
            residual_sequence = torch.cat([
                residual_changes.new_zeros(
                    batch_size, 1, num_nodes, self.projection_dim
                ),
                residual_changes,
            ], dim=1)
        else:
            residual_sequence = torch.zeros(
                batch_size, seq_len, num_nodes, self.projection_dim,
                device=temporal_sequences.device,
            )

        # Build a fused representation at every time step.  The previous
        # implementation fused only the last snapshot, so its temporal loss
        # was applied to a different space than its anomaly score.
        combined_features = torch.cat([
            residual_sequence,
            spatial_representations,
            temporal_sequences,
        ], dim=-1)
        fused_sequence = self.fusion_network(combined_features)
        fused_representations = fused_sequence.mean(dim=1)

        batch_size, num_nodes, feature_dim = fused_representations.shape
        representations_flat = fused_representations.view(-1, feature_dim)  # (batch_size * num_nodes, feature_dim)
        
        # Normalization
        representations_norm = F.normalize(representations_flat, p=2, dim=1)
        normal_prototype_norm = F.normalize(self.normal_prototype, p=2, dim=1)

        learned_sim_to_normal = torch.mm(
            representations_norm, normal_prototype_norm.t()
        ).squeeze(1)
        learned_distance_scores_flat = 1 - learned_sim_to_normal

        normalized_fused_sequence = F.normalize(fused_sequence, p=2, dim=-1)
        if seq_len >= 2:
            temporal_volatility = torch.norm(
                normalized_fused_sequence[:, 1:]
                - normalized_fused_sequence[:, :-1],
                dim=-1,
            ).mean(dim=1)
        else:
            temporal_volatility = fused_representations.new_zeros(
                batch_size, num_nodes
            )
        volatility_mean = temporal_volatility.mean(dim=1, keepdim=True)
        volatility_std = temporal_volatility.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        temporal_volatility_z = (
            temporal_volatility - volatility_mean
        ) / volatility_std

        # Label-free local structural-evolution evidence.  For every snapshot,
        # compare a node with the mean representation of its incident
        # neighbours, then average only over snapshots in which it has at least
        # one neighbour.  This is diagnostic during method development and has
        # no training-time cost.
        dynamic_neighbor_inconsistency = fused_representations.new_zeros(
            batch_size, num_nodes
        )
        if not self.training:
            observed_snapshots = fused_representations.new_zeros(
                batch_size, num_nodes
            )
            recency_weighted_sum = fused_representations.new_zeros(
                batch_size, num_nodes
            )
            recency_weight_total = fused_representations.new_zeros(
                batch_size, num_nodes
            )
            for time_idx in range(min(seq_len, len(adjacency_matrices))):
                adj_info = adjacency_matrices[time_idx]
                if isinstance(adj_info, dict) and 'edge_index' in adj_info:
                    edge_index = adj_info['edge_index']
                else:
                    edge_index, _ = dense_to_sparse(adj_info)
                if edge_index.numel() == 0:
                    continue
                source, target = edge_index[0], edge_index[1]
                valid_edges = (
                    (source >= 0) & (source < num_nodes)
                    & (target >= 0) & (target < num_nodes)
                )
                source = source[valid_edges]
                target = target[valid_edges]
                if source.numel() == 0:
                    continue
                for batch_idx in range(batch_size):
                    node_state = normalized_fused_sequence[batch_idx, time_idx]
                    neighbour_sum = torch.zeros_like(node_state)
                    degree = node_state.new_zeros(num_nodes)
                    # Treat the graph as an incident-neighbour relation.  This
                    # is robust to datasets storing each edge once or in both
                    # directions and does not require edge labels.
                    neighbour_sum.index_add_(0, target, node_state[source])
                    neighbour_sum.index_add_(0, source, node_state[target])
                    degree.index_add_(0, target, torch.ones_like(target, dtype=node_state.dtype))
                    degree.index_add_(0, source, torch.ones_like(source, dtype=node_state.dtype))
                    observed = degree > 0
                    neighbour_mean = neighbour_sum / degree.clamp_min(1).unsqueeze(-1)
                    local_distance = 1 - F.cosine_similarity(
                        node_state, neighbour_mean, dim=-1
                    )
                    dynamic_neighbor_inconsistency[batch_idx, observed] += (
                        local_distance[observed]
                    )
                    observed_snapshots[batch_idx, observed] += 1
                    recency_weight = float(time_idx + 1)
                    recency_weighted_sum[batch_idx, observed] += (
                        recency_weight * local_distance[observed]
                    )
                    recency_weight_total[batch_idx, observed] += recency_weight
            dynamic_neighbor_inconsistency = (
                dynamic_neighbor_inconsistency
                / observed_snapshots.clamp_min(1)
            )
            recency_neighbor_inconsistency = (
                recency_weighted_sum / recency_weight_total.clamp_min(1)
            )
        else:
            recency_neighbor_inconsistency = dynamic_neighbor_inconsistency

        # Robust unlabeled prototype transport at inference.  Target graph
        # attributes live in a different semantic space, so directly reusing
        # the source prototype can reverse the ranking direction.  Because
        # anomalies are rare, low-volatility informative target nodes provide
        # a label-free estimate of the target normal population center.
        consensus_prototype = None
        if (not self.training) and self.use_consensus_prototype:
            normalized_by_batch = representations_norm.view(
                batch_size, num_nodes, feature_dim
            )
            transported = []
            attr_dim = min(
                self.attribute_descriptor_dim, temporal_sequences.shape[-1]
            )
            for batch_idx in range(batch_size):
                informative = torch.norm(
                    temporal_sequences[batch_idx, -1, :, :attr_dim], dim=-1
                ) > 1e-6
                candidate_volatility = temporal_volatility[batch_idx][informative]
                if candidate_volatility.numel() >= 2:
                    cutoff = torch.quantile(
                        candidate_volatility.detach(), self.consensus_ratio
                    )
                    reliable = informative & (
                        temporal_volatility[batch_idx] <= cutoff
                    )
                else:
                    reliable = informative
                if reliable.sum() < 2:
                    transported.append(normal_prototype_norm.squeeze(0))
                else:
                    transported.append(F.normalize(
                        normalized_by_batch[batch_idx][reliable].mean(
                            dim=0, keepdim=True
                        ),
                        p=2,
                        dim=-1,
                    ).squeeze(0))
            consensus_prototype = torch.stack(transported, dim=0)
            sim_to_normal_reshaped = torch.einsum(
                'bnd,bd->bn', normalized_by_batch, consensus_prototype
            )
            sim_to_normal = sim_to_normal_reshaped.reshape(-1)
        else:
            sim_to_normal = learned_sim_to_normal

        distance_scores_flat = 1 - sim_to_normal
        # Directional prototype-transport residual.  A normal target node is
        # expected to move with the normal population from the learned source
        # centre toward the transported target centre.  A large positive value
        # means that the node is relatively closer to the obsolete source
        # normal direction while remaining far from the target normal centre.
        # This score is label-free and contains no fitted target-side weight.
        transport_residual_flat = (
            distance_scores_flat - learned_distance_scores_flat
        )

        def percentile_rank(values: torch.Tensor) -> torch.Tensor:
            """Deterministic empirical-CDF rank within each target graph."""
            ranks = torch.empty_like(values)
            order = torch.argsort(values, dim=1, stable=True)
            positions = torch.arange(
                values.shape[1], device=values.device, dtype=values.dtype
            ).unsqueeze(0).expand_as(values)
            ranks.scatter_(1, order, positions)
            return ranks / max(values.shape[1] - 1, 1)

        source_similarity_by_batch = -learned_distance_scores_flat.view(
            batch_size, num_nodes
        )
        score_scale = F.softplus(self.score_scale_raw) + 1e-6
        if self.prototype_mode == 'dual':
            abnormal_prototype_norm = F.normalize(
                self.abnormal_prototype, p=2, dim=1
            )
            sim_to_abnormal = torch.mm(
                representations_norm, abnormal_prototype_norm.t()
            ).squeeze(1)
            anomaly_logits_flat = score_scale * (sim_to_abnormal - sim_to_normal)
        elif consensus_prototype is not None:
            # Directional Prototype-Transport Residual (DPTR).  Normal target
            # nodes should follow the population shift from the learned source
            # normal centre to the robust target normal centre.  Rank a node by
            # how much farther it remains from the target centre than from the
            # obsolete source centre.  This uses no target labels and no target-
            # fitted mixing coefficient.  Temporal information still enters by
            # selecting the reliable nodes that define the target centre.
            sim_to_abnormal = None
            if self.use_transport_residual:
                if self.use_local_evolution_score:
                    # The source prototype and temporal objectives shape a
                    # transferable representation during training.  At target
                    # inference, rank nodes by recency-weighted disagreement
                    # with their dynamic neighbourhood.  Percentile calibration
                    # removes dataset-specific score scale without target labels.
                    anomaly_logits_flat = score_scale * (
                        percentile_rank(
                            recency_neighbor_inconsistency
                        ).reshape(-1) - 0.5
                    )
                else:
                    anomaly_logits_flat = score_scale * transport_residual_flat
            else:
                # Explicit ablation: retain robust target-prototype transport
                # but remove the directional source-to-target residual.
                anomaly_logits_flat = score_scale * (
                    distance_scores_flat - self.score_threshold
                )
        else:
            sim_to_abnormal = None
            prototype_weight = F.softplus(self.prototype_score_weight_raw)
            temporal_weight = F.softplus(self.temporal_score_weight_raw)
            anomaly_logits_flat = score_scale * (
                prototype_weight * (
                    distance_scores_flat - self.score_threshold
                )
                + temporal_weight * (
                    temporal_volatility_z.reshape(-1)
                    - self.temporal_score_threshold
                )
            )

        anomaly_scores_flat = torch.sigmoid(anomaly_logits_flat)

        anomaly_scores = anomaly_scores_flat.view(batch_size, num_nodes)  # (batch_size, num_nodes)
        anomaly_logits = anomaly_logits_flat.view(batch_size, num_nodes)
        anomaly_distances = distance_scores_flat.view(batch_size, num_nodes)
        sim_to_normal_reshaped = sim_to_normal.view(batch_size, num_nodes)
        sim_to_abnormal_reshaped = (
            sim_to_abnormal.view(batch_size, num_nodes)
            if sim_to_abnormal is not None else None
        )

        return {
            'fused_representations': fused_representations,
            'anomaly_scores': anomaly_scores,
            'anomaly_logits': anomaly_logits,
            'anomaly_distances': anomaly_distances,
            'learned_prototype_distances': learned_distance_scores_flat.view(
                batch_size, num_nodes
            ),
            'prototype_transport_residual': transport_residual_flat.view(
                batch_size, num_nodes
            ),
            'source_prototype_similarity': source_similarity_by_batch,
            'dynamic_neighbor_inconsistency': dynamic_neighbor_inconsistency,
            'recency_neighbor_inconsistency': recency_neighbor_inconsistency,
            'sim_to_normal': sim_to_normal_reshaped,
            'sim_to_abnormal': sim_to_abnormal_reshaped,
            'temporal_representations': temporal_representations,
            'spatial_representations': spatial_representations,
            'temporal_projected': temporal_projected,
            # Canonical [batch, time, node, feature] tensor used by L_self
            # and representation regularization.  Including the projected
            # spatial state is essential because DGraphFin node attributes are
            # static while its snapshot adjacencies evolve over time.
            'consistency_representations': fused_sequence,
            'spatial_projected': spatial_projected,
            'residual_features': residual_sequence[:, -1],
            'residual_sequence': residual_sequence,
            'fused_sequence': fused_sequence,
            'temporal_volatility': temporal_volatility,
            'temporal_volatility_z': temporal_volatility_z,
            'normal_prototype': self.normal_prototype,
            'consensus_prototype': consensus_prototype,
            'abnormal_prototype': self.abnormal_prototype,
            'prototype_mode': self.prototype_mode,
        }
    
    def _compute_structural_evolution(self, adjacency_matrices: Union[List[torch.Tensor], List[Dict]]) -> Dict[str, torch.Tensor]:
        """
        Compute normalized structural-change statistics without materializing
        Python sets of every edge (which is prohibitive on million-edge graphs).
        """
        seq_len = len(adjacency_matrices)
        if seq_len < 2:
            return {'evolution_features': torch.zeros(1), 'has_evolution': False}

        edge_changes = []
        degree_changes = []
        for t in range(1, seq_len):
            prev_adj = adjacency_matrices[t-1]
            curr_adj = adjacency_matrices[t]
            if isinstance(prev_adj, dict) and 'edge_index' in prev_adj:
                prev_edges = prev_adj['edge_index']
                curr_edges = curr_adj['edge_index']
                num_nodes = int(max(
                    prev_adj.get('num_nodes', 0), curr_adj.get('num_nodes', 0)
                ))
            else:
                prev_edges, _ = dense_to_sparse(prev_adj)
                curr_edges, _ = dense_to_sparse(curr_adj)
                num_nodes = int(max(prev_adj.shape[-1], curr_adj.shape[-1]))

            prev_count = int(prev_edges.shape[1])
            curr_count = int(curr_edges.shape[1])
            denominator = max(prev_count, 1)
            signed_change = (curr_count - prev_count) / denominator
            edge_changes.append([
                max(signed_change, 0.0), max(-signed_change, 0.0)
            ])

            if num_nodes > 0:
                prev_degree = torch.bincount(
                    prev_edges[0], minlength=num_nodes
                ).float()
                curr_degree = torch.bincount(
                    curr_edges[0], minlength=num_nodes
                ).float()
                relative_degree_change = (
                    (curr_degree - prev_degree).abs().mean()
                    / (prev_degree.mean() + 1e-6)
                )
                degree_changes.append(float(relative_degree_change.item()))
            else:
                degree_changes.append(0.0)

        if edge_changes:
            edge_evolution = torch.tensor(edge_changes, dtype=torch.float32)
            degree_evolution = torch.tensor(degree_changes, dtype=torch.float32)
            evolution_features = torch.cat([
                edge_evolution.flatten(), degree_evolution
            ], dim=0)
        else:
            evolution_features = torch.zeros(1)

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
        'prototype_mode': 'single',
        'score_scale_init': 5.0,
        'score_threshold_init': 0.5,
    }
    
    return config
