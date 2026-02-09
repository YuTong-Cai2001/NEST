#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import scipy.sparse as sp
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings
from sparse_graph_utils import SparseGraphProcessor

class SparseGraphModelAdapter(nn.Module):
    """
    Sparse Graph Model Adapter
    """
    
    def __init__(self, base_model: nn.Module, device: str = 'cpu', 
                 memory_threshold_mb: float = 100.0):
        """
        Initialize sparse graph adapter
        
        Args:
            base_model: Original AnomalyGFM model
            device: Computing device
            memory_threshold_mb: Memory threshold, warn but keep sparse if exceeded
        """
        super().__init__()
        self.base_model = base_model
        self.device = device
        self.memory_threshold_mb = memory_threshold_mb
        self.sparse_processor = SparseGraphProcessor(device=device)
        
        # Statistics
        self.stats = {
            'sparse_calls': 0,
            'dense_fallbacks': 0,  
            'memory_saved_mb': 0.0,
            'total_graphs_processed': 0
        }
        
    def forward(self, temporal_sequences: torch.Tensor, 
                adjacency_info: Union[List[Dict], List[torch.Tensor], List[sp.csr_matrix]], 
                node_features: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass - Prioritize sparse processing, completely avoid dense conversion
        
        Args:
            temporal_sequences: Temporal feature sequences [batch, seq_len, num_nodes, feature_dim]
            adjacency_info: Sparse graph info or other formats
            node_features: Node features [batch, num_nodes, feature_dim]
            
        Returns:
            Model output dictionary
        """
        self.stats['total_graphs_processed'] += len(adjacency_info) if isinstance(adjacency_info, list) else 1
        
        if self._is_sparse_format(adjacency_info):
            self.stats['sparse_calls'] += 1
            return self._handle_pure_sparse_input(temporal_sequences, adjacency_info, node_features)
        
        elif self._is_scipy_sparse_format(adjacency_info):
            self.stats['sparse_calls'] += 1
            return self._handle_scipy_sparse_input(temporal_sequences, adjacency_info, node_features)
        
        else:
            warnings.warn(" Using dense format input, recommend converting to sparse format to save memory")
            return self.base_model(temporal_sequences, adjacency_info, node_features)
    
    def _is_sparse_format(self, adjacency_info: Any) -> bool:
        """Detect if input is PyG sparse format"""
        if isinstance(adjacency_info, list) and len(adjacency_info) > 0:
            first_item = adjacency_info[0]
            return isinstance(first_item, dict) and 'edge_index' in first_item
        return False
    
    def _is_scipy_sparse_format(self, adjacency_info: Any) -> bool:
        """Detect if input is scipy sparse format"""
        if isinstance(adjacency_info, list) and len(adjacency_info) > 0:
            first_item = adjacency_info[0]
            return hasattr(first_item, 'toarray')  # scipy sparse matrix characteristic
        return False
    
    def _handle_pure_sparse_input(self, temporal_sequences: torch.Tensor, 
                                sparse_adjacency_info: List[Dict], 
                                node_features: torch.Tensor) -> Dict[str, Any]:
        """
        Handle PyG format sparse graph input
        """
        print(f" Adapter: Passing sparse format graph info (timesteps: {len(sparse_adjacency_info)})")
        
        # Calculate memory savings statistics
        total_edges = sum(info['edge_index'].shape[1] for info in sparse_adjacency_info)
        total_nodes = sparse_adjacency_info[0].get('num_nodes', node_features.shape[-2])
        
        dense_memory_mb = (total_nodes * total_nodes * len(sparse_adjacency_info) * 4) / (1024 * 1024)
        sparse_memory_mb = (total_edges * 2 * 4) / (1024 * 1024)  # edge_index memory
        memory_saved = dense_memory_mb - sparse_memory_mb
        
        self.stats['memory_saved_mb'] += memory_saved
        
        
        # Directly pass sparse graph info to base model that now supports sparse input
        return self.base_model(temporal_sequences, sparse_adjacency_info, node_features)
    
    def _handle_scipy_sparse_input(self, temporal_sequences: torch.Tensor,
                                 scipy_adjacency_list: List[sp.csr_matrix],
                                 node_features: torch.Tensor) -> Dict[str, Any]:
        """
        Handle scipy sparse matrix input - convert to PyG format but keep sparse
        """
        pyg_sparse_info = []
        
        for adj_matrix in scipy_adjacency_list:
            # Convert to PyG format using sparse processor
            edge_index, edge_attr = self.sparse_processor.sparse_to_pyg_format(adj_matrix)
            
            pyg_sparse_info.append({
                'edge_index': edge_index.to(self.device),
                'edge_attr': edge_attr.to(self.device) if edge_attr is not None else None,
                'num_nodes': adj_matrix.shape[0]
            })
        
        # Recursively call to handle PyG format
        return self._handle_pure_sparse_input(temporal_sequences, pyg_sparse_info, node_features)
    
    def _emergency_dense_fallback(self, temporal_sequences: torch.Tensor,
                                sparse_adjacency_info: List[Dict],
                                node_features: torch.Tensor) -> Dict[str, Any]:
        """
        Emergency dense fallback

        """
        self.stats['dense_fallbacks'] += 1
        warnings.warn(
            "Please modify AnomalyGFMModel to support sparse input as soon as possible."
        )
        
        dense_adjacency_matrices = []
        
        for sparse_info in sparse_adjacency_info:
            edge_index = sparse_info['edge_index']
            edge_attr = sparse_info.get('edge_attr', None)
            num_nodes = sparse_info.get('num_nodes', node_features.shape[-2])
            
            # Estimate memory risk
            dense_memory_mb = (num_nodes * num_nodes * 4) / (1024 * 1024)
            if dense_memory_mb > self.memory_threshold_mb:
                warnings.warn(f" Memory risk: Converting {dense_memory_mb:.1f}MB graph to dense format!")
            
            # Forced to convert to dense format
            dense_adj = self._sparse_to_dense_safe(edge_index, edge_attr, num_nodes)
            dense_adjacency_matrices.append(dense_adj)
        
        return self.base_model(temporal_sequences, dense_adjacency_matrices, node_features)
    
    def _sparse_to_dense(self, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor], 
                        num_nodes: int) -> torch.Tensor:
        """Convert sparse format to dense adjacency matrix"""
        device = edge_index.device
        
        # Create dense matrix
        dense_adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float32)
        
        if edge_index.numel() > 0:
            row, col = edge_index[0], edge_index[1]
            
            # Safe index check, prevent out-of-bounds
            valid_mask = (row < num_nodes) & (col < num_nodes) & (row >= 0) & (col >= 0)
            row = row[valid_mask]
            col = col[valid_mask]
            
            if len(row) > 0:
                if edge_attr is not None:
                    edge_attr = edge_attr[valid_mask]
                    dense_adj[row, col] = edge_attr
                else:
                    dense_adj[row, col] = 1.0
        
        return dense_adj
    
    def _sparse_to_dense_safe(self, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor], 
                             num_nodes: int) -> torch.Tensor:
        """Safe sparse-to-dense conversion (handle large graphs)"""
        try:
            return self._sparse_to_dense(edge_index, edge_attr, num_nodes)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f" Insufficient memory, trying CPU processing...")
                return self._sparse_to_dense_cpu(edge_index, edge_attr, num_nodes)
            else:
                raise e
    
    def _sparse_to_dense_cpu(self, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor], 
                            num_nodes: int) -> torch.Tensor:
        """Sparse-to-dense conversion on CPU"""
        edge_index_cpu = edge_index.cpu()
        edge_attr_cpu = edge_attr.cpu() if edge_attr is not None else None
        
        dense_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        
        if edge_index_cpu.numel() > 0:
            row, col = edge_index_cpu[0], edge_index_cpu[1]
            if edge_attr_cpu is not None:
                dense_adj[row, col] = edge_attr_cpu
            else:
                dense_adj[row, col] = 1.0
        
        return dense_adj.to(self.device)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        total_calls = self.stats['sparse_calls'] + self.stats['dense_fallbacks']
        
        return {
            **self.stats,
            'memory_efficiency': f"{self.stats['memory_saved_mb']:.1f}MB saved",
            'sparse_ratio': self.stats['sparse_calls'] / max(1, total_calls),
            'fallback_ratio': self.stats['dense_fallbacks'] / max(1, total_calls),
            'efficiency_score': f"{(self.stats['sparse_calls'] / max(1, total_calls)) * 100:.1f}% sparse"
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'sparse_calls': 0,
            'dense_fallbacks': 0,
            'memory_saved_mb': 0.0,
            'total_graphs_processed': 0
        }
    
    def print_performance_summary(self):
        """Print performance summary"""
        stats = self.get_stats()
        print(f"\nSparse Graph Adapter Performance Summary:")
        print(f"  Total graphs processed: {stats['total_graphs_processed']}")
        print(f"  Sparse processing calls: {stats['sparse_calls']}")
        print(f"  Dense fallback calls: {stats['dense_fallbacks']}")
        print(f"  Sparse ratio: {stats['efficiency_score']}")
        print(f"  Memory saved: {stats['memory_efficiency']}")
        
        if stats['dense_fallbacks'] > 0:
            print(f"   Warning: {stats['dense_fallbacks']} dense fallbacks occurred")
            print(f"     Recommendation: Modify AnomalyGFMModel to support sparse input!")


def create_sparse_model_adapter(base_model: nn.Module, device: str = 'cpu', 
                               memory_threshold_mb: float = 100.0) -> SparseGraphModelAdapter:
    """
    Convenience function to create sparse graph model adapter
    
    Args:
        base_model: Original AnomalyGFM model
        device: Computing device
        memory_threshold_mb: Memory threshold
        
    Returns:
        Configured sparse graph adapter
    """
    adapter = SparseGraphModelAdapter(
        base_model=base_model,
        device=device, 
        memory_threshold_mb=memory_threshold_mb
    )
    
    print(f" Sparse graph adapter created (memory threshold: {memory_threshold_mb:.1f}MB)")
    return adapter


# Example usage and testing
if __name__ == "__main__":
    print(" Sparse Graph Model Adapter Demo - True Sparse Processing")
    
    # Simulate model that supports sparse input
    class MockSparseAnomalyGFM(nn.Module):
        """Simulate AnomalyGFM model that already supports sparse input"""
        def forward(self, temporal_seq, adj_info, node_features):
            print(f" Model received sparse graph info: {type(adj_info)} (length: {len(adj_info)})")
            if isinstance(adj_info[0], dict):
                print(f"   Sparse format: edge_index shape = {adj_info[0]['edge_index'].shape}")
            return {'anomaly_scores': torch.randn(node_features.shape[-2])}
    
    # Simulate old model that doesn't support sparse input  
    class MockDenseAnomalyGFM(nn.Module):
        """Simulate old AnomalyGFM model that only supports dense input"""
        def forward(self, temporal_seq, adj_matrices, node_features):
            if isinstance(adj_matrices[0], dict):
                raise TypeError("Model does not support sparse input format")
            print(f" Model received dense matrix: {adj_matrices[0].shape}")
            return {'anomaly_scores': torch.randn(node_features.shape[-2])}
    
    print("\n Test 1: Model that supports sparse input")
    sparse_model = MockSparseAnomalyGFM()
    adapter1 = create_sparse_model_adapter(sparse_model, device='cpu')
    
    # Simulate sparse input
    sparse_input = [{
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        'edge_attr': torch.tensor([1.0, 0.5, 0.8]),
        'num_nodes': 10
    }]
    
    temporal_seq = torch.randn(1, 5, 10, 64)
    node_features = torch.randn(1, 10, 64)
    
    output1 = adapter1(temporal_seq, sparse_input, node_features)
    print(f" Output: {output1['anomaly_scores'].shape}")
    adapter1.print_performance_summary()
    
    print("\n Test 2: Old model that doesn't support sparse input")
    dense_model = MockDenseAnomalyGFM()
    adapter2 = create_sparse_model_adapter(dense_model, device='cpu', memory_threshold_mb=0.1)
    
    try:
        output2 = adapter2(temporal_seq, sparse_input, node_features)
        print(f" Fallback output: {output2['anomaly_scores'].shape}")
        adapter2.print_performance_summary()
    except Exception as e:
        print(f" Expected error: {e}")
