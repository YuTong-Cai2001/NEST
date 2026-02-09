import torch
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Optional, Union
from torch import Tensor
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F


class SparseGraphProcessor:
    """
    Sparse Graph Processor - Efficiently process large-scale graph structures
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def sparse_to_pyg_format(self, adj_matrix: sp.csr_matrix) -> Tuple[Tensor, Tensor]:
        """
        Convert scipy sparse matrix to PyG format edge_index and edge_attr
        
        Args:
            adj_matrix: scipy sparse adjacency matrix
            
        Returns:
            edge_index: (2, num_edges) edge indices
            edge_attr: (num_edges,) edge weights
        """
        # Convert to COO format for easier processing
        coo_adj = adj_matrix.tocoo()
        
        # Create edge_index
        edge_index = torch.LongTensor(np.vstack((coo_adj.row, coo_adj.col)))
        
        # Create edge_attr
        edge_attr = torch.FloatTensor(coo_adj.data)
        
        return edge_index.to(self.device), edge_attr.to(self.device)
    
    def create_sparse_tensor(self, adj_matrix: sp.csr_matrix) -> torch.sparse.FloatTensor:
        """
        Create PyTorch sparse tensor, avoid dense conversion
        
        Args:
            adj_matrix: scipy sparse matrix
            
        Returns:
            PyTorch sparse tensor
        """
        coo_adj = adj_matrix.tocoo()
        indices = torch.LongTensor(np.vstack((coo_adj.row, coo_adj.col)))
        values = torch.FloatTensor(coo_adj.data)
        shape = coo_adj.shape
        
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
        return sparse_tensor.to(self.device)
    
    def batch_sparse_matrices(self, adj_matrices: List[sp.csr_matrix]) -> Tuple[Tensor, Tensor]:
        """
        Batch multiple sparse adjacency matrices into giant graph format
        
        Args:
            adj_matrices: List of sparse adjacency matrices
            
        Returns:
            combined_edge_index: Combined edge indices
            batch_info: Batch info for subsequent splitting
        """
        combined_edge_indices = []
        batch_info = []
        node_offset = 0
        
        for i, adj_matrix in enumerate(adj_matrices):
            edge_index, _ = self.sparse_to_pyg_format(adj_matrix)
            
            # Apply node index offset
            offset_edge_index = edge_index + node_offset
            combined_edge_indices.append(offset_edge_index)
            
            # Record batch info
            num_nodes = adj_matrix.shape[0]
            batch_info.append({
                'start_node': node_offset,
                'end_node': node_offset + num_nodes,
                'num_nodes': num_nodes
            })
            
            node_offset += num_nodes
        
        # Merge all edge indices
        combined_edge_index = torch.cat(combined_edge_indices, dim=1)
        
        return combined_edge_index, batch_info
    
    def zero_shot_graph_processing(self, 
                                 full_adj_matrix: sp.csr_matrix,
                                 batch_size: int = 32) -> dict:
        """
        Correct graph processing method for zero-shot inference
        
        Args:
            full_adj_matrix: Complete target domain graph structure
            batch_size: Batch processing size
            
        Returns:
            Processing result dictionary
        """
        # 1. Convert to sparse tensor format
        sparse_tensor = self.create_sparse_tensor(full_adj_matrix)
        
        # 2. Extract PyG format edge information
        edge_index, edge_attr = self.sparse_to_pyg_format(full_adj_matrix)
        
        # 3. Calculate graph statistics
        num_nodes = full_adj_matrix.shape[0]
        num_edges = full_adj_matrix.nnz
        density = num_edges / (num_nodes ** 2)
        
        print(f"    Graph Processing Statistics:")
        print(f"   - Number of nodes: {num_nodes:,}")
        print(f"   - Number of edges: {num_edges:,}")
        print(f"   - Density: {density:.6f}")
        print(f"   - Memory saved: {self._calculate_memory_savings(num_nodes, num_edges)}")
        
        return {
            'sparse_tensor': sparse_tensor,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density
        }
    
    def _calculate_memory_savings(self, num_nodes: int, num_edges: int) -> str:
        """Calculate memory savings using sparse format"""
        dense_memory = num_nodes * num_nodes * 4  # float32
        sparse_memory = num_edges * 12  # 2 indices + 1 value, each 4 bytes
        
        savings_ratio = (dense_memory - sparse_memory) / dense_memory
        savings_mb = (dense_memory - sparse_memory) / (1024 * 1024)
        
        return f"{savings_ratio:.1%} ({savings_mb:.1f}MB)"
    
    def batch_graphs(self, adj_matrices: List[sp.csr_matrix]) -> Tuple[Tensor, Tensor]:
        """
        Batch multiple sparse adjacency matrices into giant graph format
        
        Args:
            adj_matrices: List of sparse adjacency matrices
            
        Returns:
            tuple: (edge_index, edge_attr) PyG format giant graph
        """
        if not adj_matrices:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), torch.empty((0,), dtype=torch.float, device=self.device)
        
        combined_edge_indices = []
        combined_edge_attrs = []
        node_offset = 0
        
        # Record batch processing information
        graph_sizes = []
        
        for i, adj_matrix in enumerate(adj_matrices):
            # Convert to PyG format
            edge_index, edge_attr = self.sparse_to_pyg_format(adj_matrix)
            
            # Add node offset
            if edge_index.numel() > 0:  # Only process when there are edges
                offset_edge_index = edge_index + node_offset
                combined_edge_indices.append(offset_edge_index)
                if edge_attr is not None:
                    combined_edge_attrs.append(edge_attr)
            
            num_nodes = adj_matrix.shape[0]
            graph_sizes.append(num_nodes)
            node_offset += num_nodes
        
        # Merge all edges
        if combined_edge_indices:
            combined_edge_index = torch.cat(combined_edge_indices, dim=1)
            if combined_edge_attrs:
                combined_edge_attr = torch.cat(combined_edge_attrs, dim=0)
            else:
                combined_edge_attr = torch.ones(combined_edge_index.shape[1], device=self.device)
        else:
            # Handle case with no edges
            combined_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            combined_edge_attr = torch.empty((0,), dtype=torch.float, device=self.device)
        
        # Calculate memory savings
        total_nodes = sum(graph_sizes)
        dense_size = total_nodes * total_nodes * 4  # assume float32
        sparse_size = combined_edge_index.numel() * 4 + combined_edge_attr.numel() * 4
        
        if total_nodes > 0:
            savings = max(0, (dense_size - sparse_size) / dense_size * 100)
            print(f"   Batched {len(adj_matrices)} graphs: {total_nodes} nodes, {combined_edge_index.shape[1]} edges, memory saved {savings:.1f}%")
        
        return combined_edge_index, combined_edge_attr
    
    def temporal_graph_aggregation(self, 
                                 temporal_adj_matrices: List[sp.csr_matrix],
                                 method: str = 'average') -> sp.csr_matrix:
        """
        Temporal graph structure aggregation
        
        Args:
            temporal_adj_matrices: List of temporal adjacency matrices
            method: Aggregation method ('average', 'union', 'intersection')
            
        Returns:
            Aggregated adjacency matrix
        """
        if not temporal_adj_matrices:
            raise ValueError("Temporal adjacency matrix list cannot be empty")
        
        if len(temporal_adj_matrices) == 1:
            return temporal_adj_matrices[0]
        
        # Ensure all matrices have the same shape
        base_shape = temporal_adj_matrices[0].shape
        for adj in temporal_adj_matrices[1:]:
            if adj.shape != base_shape:
                raise ValueError(f"Adjacency matrix shape mismatch: {adj.shape} != {base_shape}")
        
        if method == 'average':
            # Average aggregation: edge weights are frequency of appearance in timesteps
            sum_adj = sum(temporal_adj_matrices)
            avg_adj = sum_adj / len(temporal_adj_matrices)
            return avg_adj.tocsr()
            
        elif method == 'union':
            # Union: keep edges that exist in any timestep
            union_adj = temporal_adj_matrices[0].copy()
            for adj in temporal_adj_matrices[1:]:
                union_adj = union_adj + adj
                union_adj.data[union_adj.data > 0] = 1  # Binarize
            return union_adj.tocsr()
            
        elif method == 'intersection':
            # Intersection: keep edges that exist in all timesteps
            intersect_adj = temporal_adj_matrices[0].copy()
            intersect_adj.data[:] = 1  # Binarize
            
            for adj in temporal_adj_matrices[1:]:
                binary_adj = adj.copy()
                binary_adj.data[:] = 1
                intersect_adj = intersect_adj.multiply(binary_adj)
                
            return intersect_adj.tocsr()
        
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")


def convert_dataset_to_sparse(dataset) -> None:
    """
    Convert TemporalGraphDataset to use sparse format
    """
    original_getitem = dataset.__getitem__
    processor = SparseGraphProcessor(dataset.device if hasattr(dataset, 'device') else 'cpu')
    
    def sparse_getitem(idx):
        temporal_sequence = dataset.temporal_sequences[idx]
        adjacency_sequence = dataset.temporal_adjacencies[idx]
        timestamp = dataset.valid_timestamps[idx]
        
        # Convert to torch tensor
        temporal_tensor = torch.stack([
            torch.FloatTensor(feat) for feat in temporal_sequence
        ])
        
        adjacency_sparse_list = []
        for adj in adjacency_sequence:
            # Ensure scipy sparse format
            if hasattr(adj, 'toarray'):  # Already scipy sparse matrix
                adjacency_sparse_list.append(adj)
            elif hasattr(adj, 'cpu'):  # torch tensor
                adj_np = adj.cpu().numpy()
                adjacency_sparse_list.append(sp.csr_matrix(adj_np))
            else:  # numpy dense matrix
                adjacency_sparse_list.append(sp.csr_matrix(adj))
        
        # Get corresponding labels
        if dataset.labels is not None:
            num_nodes = temporal_tensor.shape[1]
            node_labels = torch.FloatTensor(dataset.labels[:num_nodes])
            return temporal_tensor, adjacency_sparse_list, timestamp, node_labels
        else:
            return temporal_tensor, adjacency_sparse_list, timestamp, None
    
    # Replace original method
    dataset.__getitem__ = sparse_getitem


# Example usage and testing functions
def test_sparse_efficiency():
    """Test efficiency improvement of sparse processing"""
    
    # Create test graph
    num_nodes = 1000
    num_edges = 5000  # Sparse graph
    
    # Generate random sparse adjacency matrix
    row = np.random.randint(0, num_nodes, num_edges)
    col = np.random.randint(0, num_nodes, num_edges)
    data = np.ones(num_edges)
    
    sparse_adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    processor = SparseGraphProcessor()
    
    # Test memory usage
    dense_memory = num_nodes * num_nodes * 4
    sparse_memory = sparse_adj.nnz * 12
    
    # Test zero-shot processing
    result = processor.zero_shot_graph_processing(sparse_adj)
    
    return result


if __name__ == "__main__":
    test_sparse_efficiency()
