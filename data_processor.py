import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.sparse as sp
import networkx as nx
from datetime import datetime, timedelta
from collections import defaultdict
import os
from typing import Tuple, Dict, List, Optional
import pickle
import hashlib
import psutil
import gc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

class AnomalyGFMDataProcessor:
    """
    AnomalyGFM Data Processor - Implements temporal residual computation and cross-domain feature alignment
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data processor
        
        Args:
            config: Configuration dictionary containing data paths, parameters, etc.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.temporal_window = config.get('temporal_window', 7)  
        self.residual_dim = config.get('residual_dim', 128)     
        self.sequence_length = config.get('sequence_length', 10)  
        
        self.target_dim = config.get('target_dim', 128)         
        self.use_svd = config.get('use_svd', True)              
        self.alignment_mode = config.get('alignment_mode', 'invariant')
        
        self.dgraphfin_path = config.get('dgraphfin_path', 'data/DGraphFin2/')
        self.figraph_path = config.get('figraph_path', 'data/FiGraph-main/data/')
        
        self.cache_dir = config.get('cache_dir', 'cache/')
        self.enable_cache = config.get('enable_cache', True)
        self.random_state = int(config.get('seed', 42))

        # Fitted once and reused for every target snapshot.  Re-fitting SVD for
        # each year creates unrelated coordinate systems and destroys temporal
        # comparability.
        self.source_reducer = None
        self.target_reducer = None
        self.source_scaler = None
        self.target_scaler = None
        self.source_descriptor_scaler = None
        self.target_descriptor_scaler = None
        self.feature_aligner_fitted = False
        
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            cache_status = f"{len(cache_files)} cache files exist" if cache_files else "No cache files"
        else:
            cache_status = "Cache disabled"
        

    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """
        Generate cache key
        
        Args:
            operation: Operation name
            **kwargs: Operation parameters
            
        Returns:
            Cache key string
        """
        cache_data = {
            'cache_version': 2,
            'operation': operation,
            'temporal_window': self.temporal_window,
            'sequence_length': self.sequence_length,
            **kwargs
        }
        
        cache_str = str(sorted(cache_data.items()))
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:12]
        
        return f"{operation}_{cache_hash}"
    def _monitor_memory(self, stage: str = "") -> str:
        """
        Monitor memory usage
        
        Args:
            stage: Current stage description
            
        Returns:
            Memory usage string
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            system_memory = psutil.virtual_memory()
            
            gc.collect()
            
            memory_status = (
                f"Memory usage {stage}: "
                f"Process {memory_info.rss / 1024 / 1024 / 1024:.1f}GB "
                f"({memory_percent:.1f}%), "
                f"System available {system_memory.available / 1024 / 1024 / 1024:.1f}GB "
                f"({system_memory.percent:.1f}% used)"
            )
            return memory_status
        except Exception as e:
            return f"Memory monitoring failed: {e}"



    def _save_cache(self, cache_key: str, data: any) -> bool:
        """
        Save cache data
        
        Args:
            cache_key: Cache key
            data: Data to cache
            
        Returns:
            Whether save was successful
        """
        if not self.enable_cache:
            return False
            
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"    Cache saved: {cache_key} ({size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            print(f"    Cache save failed: {e}")
            return False
    
    def _load_cache(self, cache_key: str) -> Tuple[bool, any]:
        """
        Load cache data
        
        Args:
            cache_key: Cache key
            
        Returns:
            (Whether successful, Data)
        """
        if not self.enable_cache:
            return False, None
            
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if not os.path.exists(cache_path):
                return False, None
                
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Calculate file size
            size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"    Cache loaded: {cache_key} ({size_mb:.1f}MB)")
            return True, data
            
        except Exception as e:
            print(f"    Cache load failed: {e}")
            return False, None

    def load_dgraphfin_data(self) -> Dict:
        """
        Load DGraphFin data and perform temporal preprocessing
        
        Returns:
            Dictionary containing node features, edges, labels, and temporal information
        """
        
        data_path = os.path.join(self.dgraphfin_path, 'dgraphfin.npz')
        data = np.load(data_path)
        
        # Extract basic data
        node_features = data['x']          
        labels_raw = data['y']             
        edge_index = data['edge_index']    
        edge_type = data['edge_type']      
        edge_timestamp = data['edge_timestamp']  
        
        if labels_raw.ndim == 2:
            if labels_raw.shape[1] == 2:
                labels = labels_raw[:, 1].astype(np.int8)  
            else:
                labels = labels_raw.argmax(axis=1).astype(np.int8)
        else:
            labels = labels_raw.astype(np.int8)

        
        edge_ts_path = os.path.join(self.dgraphfin_path, 'dgraphfinv2_edge_timestamp.npy')
        node_ts_path = os.path.join(self.dgraphfin_path, 'dgraphfinv2_node_timestamp.npy')
        
        edge_timestamp_v2 = None
        node_timestamp = None
        
        try:
            edge_timestamp_v2 = np.load(edge_ts_path)
            print(f"    Additional edge timestamps: shape={edge_timestamp_v2.shape}")
        except:
            print(f"    Additional edge timestamp file not found")
            
        try:
            node_timestamp = np.load(node_ts_path)
            print(f"    Node timestamps: shape={node_timestamp.shape}")
        except:
            print(f"    Node timestamp file not found")
        
        if edge_timestamp_v2 is not None:
            edge_timestamp = edge_timestamp_v2
        
        data.close()
        
        return {
            'node_features': node_features,
            'labels': labels,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'edge_timestamp': edge_timestamp,
            'node_timestamp': node_timestamp,
            'num_nodes': node_features.shape[0],
            'num_edges': edge_index.shape[0],
            'feature_dim': node_features.shape[1]
        }

    def load_figraph_data(self, year: int = 2014) -> Dict:
        """
        Load FiGraph data for specified year
        
        Args:
            year: Year (2014-2022)
            
        Returns:
            Dictionary containing node features, edges, and labels
        """
        print(f"\n Loading FiGraph {year} data...")
        
        year_path = os.path.join(self.figraph_path, str(year))
        
        # Load edge data
        edges_file = os.path.join(year_path, f'edges{year}.csv')
        edges_df = pd.read_csv(edges_file, header=None)
        edges_df.columns = ['source', 'target', 'edge_type']

        # FiGraph uses valid alphanumeric IDs such as U222540 and L300347.
        # The old digit-only check mistook the first real edge for a header and
        # silently dropped it in every year.
        first_row_values = [
            str(value).strip().lower() for value in edges_df.iloc[0].values
        ]
        source_headers = {'source', 'src', 'from', 'source_id', 'sourceid'}
        target_headers = {'target', 'dst', 'to', 'target_id', 'targetid'}
        if (
            first_row_values[0] in source_headers
            and first_row_values[1] in target_headers
        ):
            edges_df = pd.read_csv(edges_file, skiprows=1, header=None)
            edges_df.columns = ['source', 'target', 'edge_type']

        all_node_ids = set(edges_df['source'].tolist() + edges_df['target'].tolist())
        node_to_int = {node_id: idx for idx, node_id in enumerate(sorted(all_node_ids))}
        
        edges_df['source'] = edges_df['source'].map(node_to_int)
        edges_df['target'] = edges_df['target'].map(node_to_int)
        
        unmapped_source = edges_df['source'].isna().sum()
        unmapped_target = edges_df['target'].isna().sum()
        
        if unmapped_source > 0 or unmapped_target > 0:
            print(f"    Nodes with failed mapping: source={unmapped_source}, target={unmapped_target}")
            edges_df = edges_df.dropna(subset=['source', 'target'])
        
        edges_df['source'] = edges_df['source'].astype(int)
        edges_df['target'] = edges_df['target'].astype(int)
        

        # Index 0 is a valid zero-based node index.  The previous code removed
        # every edge incident to that node.
        edges_df = edges_df[(edges_df['source'] >= 0) & (edges_df['target'] >= 0)]

        if len(edges_df) == 0:
            raise ValueError("Edge file is empty after cleaning. Node ID format may have issues. Please check the original CSV.")
        
        features_file = os.path.join(year_path, f'ListedCompanyFeatures772_{year}.csv')
        features_df = pd.read_csv(features_file)
        
        edge_type_counts = edges_df['edge_type'].value_counts()
        
        if 'nodeID' in features_df.columns:
            
            feature_node_ids = features_df['nodeID'].tolist()
            feature_row_by_id = {
                feature_id: idx for idx, feature_id in enumerate(feature_node_ids)
            }
            
            max_node_id = max(node_to_int.values()) if node_to_int else -1
            aligned_features = []
            aligned_labels = []
            
            # Create reverse mapping: from integer index to original ID
            int_to_node = {idx: node_id for node_id, idx in node_to_int.items()}
            
            for node_idx in range(max_node_id + 1):
                original_node_id = int_to_node[node_idx]
                
                # Find corresponding node in feature file
                feature_row_idx = feature_row_by_id.get(original_node_id)
                
                if feature_row_idx is not None:
                    # Found corresponding features
                    aligned_features.append(features_df.iloc[feature_row_idx])
                    if 'Label' in features_df.columns:
                        aligned_labels.append(features_df.iloc[feature_row_idx]['Label'])
                    else:
                        aligned_labels.append(0)
                else:
                    # Not found, create zero features
                    zero_row = pd.Series([0] * len(features_df.columns), index=features_df.columns)
                    zero_row['nodeID'] = original_node_id
                    zero_row['Year'] = year
                    zero_row['Label'] = -1
                    aligned_features.append(zero_row)
                    aligned_labels.append(-1)
            
            # Reconstruct feature DataFrame
            features_df = pd.DataFrame(aligned_features)
            labels = np.array(aligned_labels)
            
        else:
            labels = features_df['Label'].values if 'Label' in features_df.columns else None
        
        
        meta_columns = ['nodeID', 'Year', 'Label']
        feature_columns = [col for col in features_df.columns if col not in meta_columns]
        
        feature_data = features_df[feature_columns].copy()
        
        feature_data = feature_data.apply(pd.to_numeric, errors='coerce')
        
        feature_data = feature_data.fillna(0)
        
        node_features = feature_data.values.astype(np.float32)
        
        return {
            'node_features': node_features,
            'labels': labels,
            'edges': edges_df,
            'edge_types': edge_type_counts.index.tolist(),
            'num_nodes': len(features_df),
            'num_edges': len(edges_df),
            'feature_dim': node_features.shape[1],
            'year': year,
            'node_id_mapping': node_to_int  # Save node ID mapping for later use
        }

    def load_figraph_temporal_sequence(self, start_year: int = 2018, end_year: int = 2022) -> Dict:
        """
        Load multi-year dynamic graph sequence from FiGraph
        
        Args:
            start_year: Start year
            end_year: End year
            
        Returns:
            Dictionary containing temporal graph sequences for dynamic graph learning
        """
        print(f"\n Building FiGraph dynamic graph sequence ({start_year}-{end_year})...")
        
        available_years = []
        yearly_data = {}
        
        for year in range(start_year, end_year + 1):
            try:
                year_data = self.load_figraph_data(year)
                available_years.append(year)
                yearly_data[year] = year_data
            except Exception as e:
                print(f"    Skip year {year}: {e}")
                continue
        
        if len(available_years) < 2:
            raise ValueError(f"Dynamic graph sequence requires at least 2 years of data, currently only {len(available_years)} years available")
        

        all_nodes = set()
        all_edge_types = set()
        
        for year_data in yearly_data.values():
            all_nodes.update(year_data['node_id_mapping'].keys())
            all_edge_types.update(year_data['edge_types'])
        
        # A single global mapping is essential: local index i from two
        # independently loaded years does not necessarily identify the same
        # company.
        global_node_ids = sorted(all_nodes, key=lambda value: str(value))
        global_node_to_int = {
            node_id: idx for idx, node_id in enumerate(global_node_ids)
        }
        total_nodes = len(global_node_ids)
        feature_dim = next(iter(yearly_data.values()))['feature_dim']
        
        
        temporal_sequence = []       
        temporal_adjacencies = []    
        temporal_labels = []         
        temporal_presence_masks = []
        valid_timestamps = []
        
        for year in available_years:
            year_data = yearly_data[year]
            
            local_features = year_data['node_features']
            local_labels = (
                year_data['labels'] if year_data['labels'] is not None
                else np.full(year_data['num_nodes'], -1, dtype=np.float32)
            )
            local_to_raw = {
                local_idx: raw_id
                for raw_id, local_idx in year_data['node_id_mapping'].items()
            }

            year_features = np.zeros(
                (total_nodes, feature_dim), dtype=np.float32
            )
            # -1 marks nodes without a valid label in that snapshot.  They are
            # excluded from evaluation instead of being counted as normal.
            year_labels = np.full(total_nodes, -1, dtype=np.float32)
            presence_mask = np.zeros(total_nodes, dtype=bool)

            for local_idx, raw_id in local_to_raw.items():
                if local_idx >= len(local_features):
                    continue
                global_idx = global_node_to_int[raw_id]
                year_features[global_idx] = local_features[local_idx]
                if local_idx < len(local_labels):
                    year_labels[global_idx] = local_labels[local_idx]
                presence_mask[global_idx] = True
            
            # Build adjacency matrix
            edges_df = year_data['edges']
            
            adj_matrix = sp.lil_matrix((total_nodes, total_nodes), dtype=np.float32)
            
            edge_type_weights = {edge_type: 1.0 for edge_type in all_edge_types}  # Adjustable weights
            
            for _, row in edges_df.iterrows():
                src_local = int(row['source'])
                tgt_local = int(row['target'])
                if src_local not in local_to_raw or tgt_local not in local_to_raw:
                    continue
                src = global_node_to_int[local_to_raw[src_local]]
                tgt = global_node_to_int[local_to_raw[tgt_local]]
                edge_type = row['edge_type']
                
                if 0 <= src < total_nodes and 0 <= tgt < total_nodes:
                    weight = edge_type_weights.get(edge_type, 1.0)
                    adj_matrix[src, tgt] = weight
                    adj_matrix[tgt, src] = weight  # Undirected graph
            
            adj_matrix = adj_matrix.tocsr()
            
            adj_matrix = adj_matrix + sp.eye(total_nodes, format='csr')

            row_sum = np.array(adj_matrix.sum(1)).flatten()
            row_sum[row_sum == 0] = 1  
            d_inv_sqrt = sp.diags(np.power(row_sum, -0.5), format='csr')
            adj_normalized = d_inv_sqrt @ adj_matrix @ d_inv_sqrt
            
            temporal_sequence.append(year_features)
            temporal_adjacencies.append(adj_normalized)
            temporal_labels.append(year_labels)
            temporal_presence_masks.append(presence_mask)
            valid_timestamps.append(f"FiGraph-{year}")
            
        edge_counts = [adj.nnz for adj in temporal_adjacencies]
        
        return {
            'static_fallback': False,
            'temporal_sequences': [temporal_sequence],     
            'temporal_adjacencies': [temporal_adjacencies],
            'temporal_labels': [temporal_labels],          
            'temporal_presence_masks': [temporal_presence_masks],
            'valid_timestamps': [f"FiGraph-dynamic-{start_year}-{end_year}"],
            'available_years': available_years,
            'sequence_length': len(available_years),
            'num_nodes': total_nodes,
            'feature_dim': feature_dim,
            'global_node_ids': global_node_ids,
            'global_node_to_int': global_node_to_int,
            'edge_evolution_stats': {
                'min_edges': min(edge_counts),
                'max_edges': max(edge_counts), 
                'avg_edges': np.mean(edge_counts),
                'edge_volatility': np.std(edge_counts) / np.mean(edge_counts)
            }
        }

    def _select_important_nodes(self, dgraphfin_data: Dict, max_nodes: int) -> np.ndarray:
        """
        Select important nodes
        
        Args:
            dgraphfin_data: Graph data
            max_nodes: Maximum number of nodes
            
        Returns:
            Array of important node indices
        """
        
        labels = dgraphfin_data['labels']
        anomaly_nodes = np.where(labels == 1)[0]
        num_anomaly = len(anomaly_nodes)
        
        print(f"      - Anomaly nodes: {num_anomaly:,} (all preserved)")
        
        edge_index = dgraphfin_data['edge_index']
        degrees = np.bincount(edge_index.flatten(), minlength=dgraphfin_data['num_nodes'])
        
        normal_nodes = np.where(labels == 0)[0]
        normal_degrees = degrees[normal_nodes]
        
        remaining_slots = max_nodes - num_anomaly
        if remaining_slots > 0 and len(normal_nodes) > 0:
            top_normal_idx = np.argsort(normal_degrees)[-remaining_slots:]
            selected_normal = normal_nodes[top_normal_idx]
            print(f"      - High-connectivity normal nodes: {len(selected_normal):,}")
        else:
            selected_normal = np.array([], dtype=int)
            print(f"      - High-connectivity normal nodes: 0 (anomaly nodes already filled the quota)")
        
        # 4. Merge important nodes
        important_nodes = np.concatenate([anomaly_nodes, selected_normal])
        important_nodes = np.sort(important_nodes)  # Sort for consistency
        
        
        return important_nodes
    
    def _subsample_graph(self, dgraphfin_data: Dict, selected_nodes: np.ndarray) -> Dict:
        """
        Subsample graph based on selected nodes
        
        Args:
            dgraphfin_data: Original graph data
            selected_nodes: Selected node indices
            
        Returns:
            Subsampled graph data
        """
        
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(selected_nodes)}
        
        # 1. Subsample node features
        node_features = dgraphfin_data['node_features'][selected_nodes]
        labels = dgraphfin_data['labels'][selected_nodes]
        
        if 'node_timestamps' in dgraphfin_data:
            node_timestamps = dgraphfin_data['node_timestamps'][selected_nodes]
        else:
            node_timestamps = None
        
        # 2. Subsample edges
        edge_index = dgraphfin_data['edge_index']
        if edge_index.shape[0] == 2 and edge_index.shape[1] > 2:
            edge_index = edge_index.T
        edge_mask = np.isin(edge_index[:, 0], selected_nodes) & np.isin(edge_index[:, 1], selected_nodes)
        
        if edge_mask.sum() > 0:
            filtered_edges = edge_index[edge_mask]
            
            # Remap edge indices
            new_edges = np.array([
                [node_mapping[edge[0]], node_mapping[edge[1]]]
                for edge in filtered_edges
            ], dtype=int)
            
            # Subsample edge features
            if 'edge_timestamps' in dgraphfin_data:
                edge_timestamps = dgraphfin_data['edge_timestamps'][edge_mask]
            else:
                edge_timestamps = None
        else:
            new_edges = np.array([]).reshape(0, 2)
            edge_timestamps = None
        
        
        edge_index_out = new_edges.T  # (2, E)
        subsampled_data = {
            'num_nodes': len(selected_nodes),
            'node_features': node_features,
            'edge_index': edge_index_out,
            'labels': labels,
            'original_node_ids': selected_nodes  
        }
        
        if node_timestamps is not None:
            subsampled_data['node_timestamps'] = node_timestamps
        
        if edge_timestamps is not None:
            subsampled_data['edge_timestamps'] = edge_timestamps
        
        return subsampled_data

    def create_temporal_snapshots(self, dgraphfin_data: Dict, 
                                window_size: int = None) -> List[Dict]:
        """
        Split DGraphFin data into temporal snapshots
        
        Args:
            dgraphfin_data: DGraphFin data dictionary
            window_size: Time window size (days)
            
        Returns:
            List of temporal snapshots
        """
        if window_size is None:
            window_size = self.temporal_window
            
        
        cache_key = self._get_cache_key(
            'temporal_snapshots',
            window_size=window_size,
            num_edges=len(dgraphfin_data['edge_index']),
            num_nodes=len(dgraphfin_data['node_features'])
        )
        
        # Try to load from cache
        cache_loaded, cached_snapshots = self._load_cache(cache_key)
        if cache_loaded:
            print(f"    Cache hit! Skip {window_size}-day window temporal snapshot computation (saves 2-3 hours)")
            print(f"    Cache file: {cache_key}")
            return cached_snapshots
        
        print(f"    Cache miss, computing temporal snapshots... (will save as: {cache_key})")
        
        edge_timestamp = dgraphfin_data['edge_timestamp']
        edge_index = dgraphfin_data['edge_index']
        edge_type = dgraphfin_data['edge_type']
        
        # Timestamp range
        min_ts, max_ts = edge_timestamp.min(), edge_timestamp.max()
        time_span = max_ts - min_ts + 1
        
        print(f"   - Timestamp range: {min_ts} - {max_ts} (span: {time_span})")
        
        # Create time windows
        snapshots = []
        num_windows = int(np.ceil(time_span / window_size))
        
        for i in range(num_windows):
            start_time = min_ts + i * window_size
            end_time = min(start_time + window_size - 1, max_ts)
            
            # Filter edges in current time window
            mask = (edge_timestamp >= start_time) & (edge_timestamp <= end_time)
            window_edges = edge_index[mask]
            window_edge_types = edge_type[mask]
            window_timestamps = edge_timestamp[mask]
            
            if len(window_edges) > 0:  # Only keep time windows with edges
                snapshot = {
                    'window_id': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'edge_index': window_edges,
                    'edge_type': window_edge_types,
                    'edge_timestamp': window_timestamps,
                    'num_edges': len(window_edges)
                }
                snapshots.append(snapshot)
        
        print(f"    Created {len(snapshots)} valid temporal snapshots")
        print(f"    Average edges per snapshot: {np.mean([s['num_edges'] for s in snapshots]):.0f}")
        
        print(f"\n Save temporal snapshots to cache...")
        if self._save_cache(cache_key, snapshots):
            print(f"    Temporal snapshots cached, next run will skip computation")
        else:
            print(f"    Cache save failed, next run will recompute")
        
        return snapshots

    def prepare_temporal_sequences(self, dgraphfin_data: Dict, 
                                 snapshots: List[Dict],
                                 memory_efficient: bool = True,
                                 max_nodes: int = 100000,
                                 max_windows: int = 20) -> Dict:
        """
        Prepare temporal feature sequences for the model
        
        Args:
            dgraphfin_data: DGraphFin raw data
            snapshots: Temporal snapshot list
            memory_efficient: Enable memory optimization mode
            max_nodes: Maximum number of nodes limit (prevent memory explosion)
            max_windows: Maximum number of time windows limit
            
        Returns:
            Dictionary containing temporal feature sequences for TCN processing
        """
        num_nodes = dgraphfin_data['num_nodes']
        num_windows = len(snapshots)

        original_num_nodes = num_nodes  
        node_mapping = None  
        selected_node_ids = np.arange(num_nodes, dtype=np.int64)
        
        if num_nodes > max_nodes:
            print(f"\n  Number of nodes ({num_nodes:,}) exceeds safety limit ({max_nodes:,})")
            
            important_nodes = self._select_important_nodes(dgraphfin_data, max_nodes)
            selected_node_ids = np.asarray(important_nodes, dtype=np.int64)
            
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(important_nodes)}
            
            dgraphfin_data = self._subsample_graph(dgraphfin_data, important_nodes)
            dgraphfin_data['original_num_nodes'] = original_num_nodes  # Save original node count
            num_nodes = len(important_nodes)
            
            print(f"    Downsampled to {num_nodes:,} important nodes")
        
        if num_windows > max_windows:
            print(f"\n  Number of time windows ({num_windows}) exceeds safety limit ({max_windows})")
            print(f"    Enable time window sampling...")
            
            # Uniformly sample time windows
            step = num_windows // max_windows
            snapshots = snapshots[::step][:max_windows]
            num_windows = len(snapshots)
            
            print(f"    Downsampled to {num_windows} time windows")
        
        if node_mapping is not None:
            print(f"    Update snapshot edge indices to match downsampled nodes...")
            updated_snapshots = []
            selected_nodes_set = set(node_mapping.keys())  # Selected node set
            
            for snapshot in snapshots:
                original_edges = snapshot['edge_index']
                
                if len(original_edges) > 0:
                    edge_mask = (
                        np.isin(original_edges[:, 0], list(selected_nodes_set)) & 
                        np.isin(original_edges[:, 1], list(selected_nodes_set))
                    )
                    
                    if edge_mask.sum() > 0:
                        filtered_edges = original_edges[edge_mask]
                        # Remap edge indices
                        remapped_edges = np.array([
                            [node_mapping[edge[0]], node_mapping[edge[1]]] 
                            for edge in filtered_edges
                        ])
                        snapshot['edge_index'] = remapped_edges
                    else:
                        snapshot['edge_index'] = np.array([]).reshape(0, 2)
                else:
                    snapshot['edge_index'] = np.array([]).reshape(0, 2)
                    
                updated_snapshots.append(snapshot)
            
            snapshots = updated_snapshots
            print(f"    Snapshot edge indices updated")
        
        # Cache check - avoid redundant temporal sequence computation!
        cache_key = self._get_cache_key(
            'temporal_sequences_safe',
            num_snapshots=num_windows,
            num_nodes=num_nodes,
            feature_dim=dgraphfin_data['node_features'].shape[1],
            max_nodes=max_nodes,
            max_windows=max_windows
        )
        
        # Try to load from cache
        cache_loaded, cached_sequences = self._load_cache(cache_key)
        if cache_loaded:
            print(f"    Cache hit!")
            print(f"    Cache file: {cache_key}")
            return cached_sequences
        
        print(f"    Cache miss, computing temporal sequences... (will save as: {cache_key})")
        
        # Use processed data (may already be downsampled)
        num_nodes = dgraphfin_data['num_nodes']  
        node_features = dgraphfin_data['node_features']
        
        
        #  Memory-safe: process time windows step by step, release memory promptly
        temporal_sequence = []
        
        print(f"    Processing {len(snapshots)} time windows step by step...")
        
        for i, snapshot in enumerate(snapshots):
            if i % 5 == 0:  # More frequent memory monitoring
                print(f"    {self._monitor_memory(f'Processing window {i+1}')}")
            
            print(f"   Processing time window {i+1}/{len(snapshots)}...", end='\r')
            
            edges = snapshot['edge_index']
            
            if len(edges) == 0:
                # Handle empty edges case
                adj_normalized = sp.csr_matrix((num_nodes, num_nodes))
            else:
                # More efficient adjacency matrix construction
                data = np.ones(len(edges), dtype=np.float32)  # Use float32 to save memory
                adj_matrix = sp.csr_matrix(
                    (data, (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes),
                    dtype=np.float32
                )
                
                adj_matrix.setdiag(1.0)
                
                # Simplified symmetric normalization (save memory)
                row_sum = np.array(adj_matrix.sum(1), dtype=np.float32).flatten()
                row_sum[row_sum == 0] = 1.0
                d_inv_sqrt = np.power(row_sum, -0.5, dtype=np.float32)
                
                adj_matrix = adj_matrix.tocoo()  # Convert to COO format for efficiency
                adj_matrix.data = adj_matrix.data * d_inv_sqrt[adj_matrix.row] * d_inv_sqrt[adj_matrix.col]
                adj_normalized = adj_matrix.tocsr()  # Convert back to CSR
            
            temporal_sequence.append({
                'window_id': i,  # Simplified window ID
                'adj_matrix': adj_normalized,
                'num_edges': len(edges)
            })
            
            del edges
            if 'adj_matrix' in locals():
                del adj_matrix
            
            if (i + 1) % 10 == 0:
                import gc
                gc.collect()
        
        
        sequence_features = [node_features] * len(temporal_sequence)  
        sequence_adjacencies = [snap['adj_matrix'] for snap in temporal_sequence]
        
        temporal_sequences = [sequence_features]
        temporal_adjacencies = [sequence_adjacencies]
        
        result = {
            'temporal_sequence': temporal_sequence,  
            'temporal_sequences': temporal_sequences,  
            'temporal_adjacencies': temporal_adjacencies,  
            'shared_node_features': node_features,  
            'num_windows': len(temporal_sequence),
            'node_feature_dim': node_features.shape[1],
            'num_nodes': num_nodes,
            'sequence_length': len(temporal_sequence),
            'valid_timestamps': [[snap['window_id'] for snap in temporal_sequence]],
            'memory_optimized': True,  # Mark as memory-optimized version
            'subsampled': num_nodes < dgraphfin_data.get('original_num_nodes', num_nodes),
            'selected_node_ids': selected_node_ids,
        }
        
        print(f"\n Save lightweight temporal sequences to cache...")
        if self._save_cache(cache_key, result):
            print(f"    Temporal sequences cached, next run will skip computation of {len(snapshots)} time windows")
        else:
            print(f"    Cache save failed, next run will recompute")
        
        return result

    @staticmethod
    def _pad_features(features: np.ndarray, target_dim: int) -> np.ndarray:
        if features.shape[1] >= target_dim:
            return features
        padding = np.zeros(
            (features.shape[0], target_dim - features.shape[1]),
            dtype=features.dtype,
        )
        return np.hstack([features, padding])

    @staticmethod
    def _valid_feature_rows(features: np.ndarray) -> np.ndarray:
        finite = np.isfinite(features)
        return finite.any(axis=1) & np.any(
            np.where(finite, features, 0.0) != 0.0, axis=1
        )

    @staticmethod
    def _fit_nonzero_scaler(features: np.ndarray) -> StandardScaler:
        valid_rows = AnomalyGFMDataProcessor._valid_feature_rows(features)
        if not valid_rows.any():
            raise ValueError('Cannot fit invariant descriptors on all-zero features')
        clean = np.nan_to_num(
            features[valid_rows], nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
        return StandardScaler().fit(clean)

    @staticmethod
    def _distribution_descriptors(
        features: np.ndarray, raw_scaler: StandardScaler,
        descriptor_scaler: Optional[StandardScaler] = None,
        fit_descriptor_scaler: bool = False,
        chunk_size: int = 20000,
    ) -> Tuple[np.ndarray, StandardScaler]:
        """Convert arbitrary-dimensional attributes into semantic statistics.

        Each output coordinate has the same meaning in every domain (mean
        standardized activation, energy, tail mass, sparsity, etc.), unlike
        independently fitted PCA/SVD coordinates.
        """
        valid_rows = AnomalyGFMDataProcessor._valid_feature_rows(features)
        descriptor_dim = 16
        descriptors = np.zeros(
            (features.shape[0], descriptor_dim), dtype=np.float32
        )

        valid_indices = np.flatnonzero(valid_rows)
        for start in range(0, len(valid_indices), chunk_size):
            indices = valid_indices[start:start + chunk_size]
            raw = np.nan_to_num(
                features[indices], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32, copy=False)
            z = raw_scaler.transform(raw).astype(np.float32, copy=False)
            z = np.clip(z, -8.0, 8.0)
            abs_z = np.abs(z)
            width = max(z.shape[1], 1)
            top_count = max(1, int(np.ceil(width * 0.05)))
            top_abs = np.partition(
                abs_z, kth=width - top_count, axis=1
            )[:, -top_count:].mean(axis=1)
            nonzero_fraction = (raw != 0).mean(axis=1)

            descriptors[indices] = np.column_stack([
                z.mean(axis=1),
                z.std(axis=1),
                abs_z.mean(axis=1),
                np.sqrt(np.mean(z * z, axis=1) + 1e-8),
                z.min(axis=1),
                z.max(axis=1),
                abs_z.max(axis=1),
                (z > 0).mean(axis=1),
                (z < 0).mean(axis=1),
                (abs_z > 1).mean(axis=1),
                (abs_z > 2).mean(axis=1),
                (abs_z > 3).mean(axis=1),
                nonzero_fraction,
                np.mean(np.clip(z, -4, 4) ** 3, axis=1),
                np.mean(np.clip(z, -4, 4) ** 4, axis=1),
                top_abs,
            ]).astype(np.float32)

        if fit_descriptor_scaler:
            descriptor_scaler = StandardScaler().fit(descriptors[valid_rows])
        if descriptor_scaler is None:
            raise RuntimeError('Descriptor scaler is not fitted')
        descriptors[valid_rows] = np.clip(
            descriptor_scaler.transform(descriptors[valid_rows]), -8.0, 8.0
        ).astype(np.float32)
        return descriptors, descriptor_scaler

    @staticmethod
    def structural_temporal_descriptors(
        adjacency_sequence: List[sp.csr_matrix],
    ) -> List[np.ndarray]:
        """Create eight domain-invariant topology/evolution coordinates."""
        binary_sequence = []
        degrees = []
        for adjacency in adjacency_sequence:
            binary = adjacency.tocsr(copy=True)
            binary.data = np.ones_like(binary.data, dtype=np.float32)
            binary.setdiag(0)
            binary.eliminate_zeros()
            binary_sequence.append(binary)
            degrees.append(np.asarray(binary.sum(axis=1)).ravel().astype(np.float32))

        outputs = []
        previous_degree = np.zeros_like(degrees[0])
        previous_binary = None
        previous_delta = np.zeros_like(degrees[0])
        for binary, degree in zip(binary_sequence, degrees):
            log_degree = np.log1p(degree)
            neighbor_sum = np.asarray(binary.dot(log_degree)).ravel()
            neighbor_mean = neighbor_sum / np.maximum(degree, 1.0)
            delta = log_degree - np.log1p(previous_degree)

            if previous_binary is None:
                churn = np.zeros_like(degree)
                new_ratio = (degree > 0).astype(np.float32)
                lost_ratio = np.zeros_like(degree)
            else:
                intersection = np.asarray(
                    previous_binary.multiply(binary).sum(axis=1)
                ).ravel().astype(np.float32)
                union = previous_degree + degree - intersection
                churn = 1.0 - intersection / np.maximum(union, 1.0)
                new_ratio = (degree - intersection) / np.maximum(degree, 1.0)
                lost_ratio = (
                    (previous_degree - intersection)
                    / np.maximum(previous_degree, 1.0)
                )

            raw = np.column_stack([
                log_degree,
                neighbor_mean,
                log_degree - neighbor_mean,
                delta,
                np.abs(delta),
                churn,
                new_ratio,
                lost_ratio + np.abs(delta - previous_delta),
            ]).astype(np.float32)
            active = (degree > 0) | (previous_degree > 0)
            normalized = np.zeros_like(raw)
            if active.any():
                normalized[active] = np.clip(
                    StandardScaler().fit_transform(raw[active]), -8.0, 8.0
                ).astype(np.float32)
            outputs.append(normalized)
            previous_degree = degree
            previous_binary = binary
            previous_delta = delta
        return outputs

    def append_structural_temporal_features(
        self, feature_sequence: List[np.ndarray],
        adjacency_sequence: List[sp.csr_matrix],
    ) -> List[np.ndarray]:
        structural = self.structural_temporal_descriptors(adjacency_sequence)
        if len(feature_sequence) != len(structural):
            raise ValueError('Feature and adjacency sequence lengths do not match')
        return [
            np.concatenate([features, structure], axis=1).astype(np.float32)
            for features, structure in zip(feature_sequence, structural)
        ]

    def fit_feature_aligner(self, source_features: np.ndarray,
                            target_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit deterministic source/target transforms once."""
        if self.alignment_mode == 'invariant':
            self.source_scaler = self._fit_nonzero_scaler(source_features)
            self.target_scaler = self._fit_nonzero_scaler(target_features)
            source_reduced, self.source_descriptor_scaler = (
                self._distribution_descriptors(
                    source_features,
                    self.source_scaler,
                    fit_descriptor_scaler=True,
                )
            )
            target_reduced, self.target_descriptor_scaler = (
                self._distribution_descriptors(
                    target_features,
                    self.target_scaler,
                    fit_descriptor_scaler=True,
                )
            )
            self.target_dim = source_reduced.shape[1]
            self.feature_aligner_fitted = True
            return source_reduced, target_reduced

        source_dim = source_features.shape[1]
        target_dim = target_features.shape[1]

        if source_dim > self.target_dim:
            self.source_reducer = TruncatedSVD(
                n_components=self.target_dim,
                random_state=self.random_state,
            )
            source_reduced = self.source_reducer.fit_transform(source_features)
        else:
            source_reduced = self._pad_features(source_features, self.target_dim)

        if target_dim > self.target_dim:
            self.target_reducer = TruncatedSVD(
                n_components=self.target_dim,
                random_state=self.random_state,
            )
            target_reduced = self.target_reducer.fit_transform(target_features)
        else:
            target_reduced = self._pad_features(target_features, self.target_dim)

        self.source_scaler = StandardScaler().fit(source_reduced)
        self.target_scaler = StandardScaler().fit(target_reduced)
        self.feature_aligner_fitted = True
        return (
            self.source_scaler.transform(source_reduced).astype(np.float32),
            self.target_scaler.transform(target_reduced).astype(np.float32),
        )

    def transform_source_features(self, features: np.ndarray) -> np.ndarray:
        if not self.feature_aligner_fitted:
            raise RuntimeError('Feature aligner has not been fitted')
        if self.alignment_mode == 'invariant':
            transformed, _ = self._distribution_descriptors(
                features, self.source_scaler, self.source_descriptor_scaler
            )
            return transformed
        reduced = (
            self.source_reducer.transform(features)
            if self.source_reducer is not None
            else self._pad_features(features, self.target_dim)
        )
        return self.source_scaler.transform(reduced).astype(np.float32)

    def transform_target_features(self, features: np.ndarray) -> np.ndarray:
        if not self.feature_aligner_fitted:
            raise RuntimeError('Feature aligner has not been fitted')
        if self.alignment_mode == 'invariant':
            transformed, _ = self._distribution_descriptors(
                features, self.target_scaler, self.target_descriptor_scaler
            )
            return transformed
        reduced = (
            self.target_reducer.transform(features)
            if self.target_reducer is not None
            else self._pad_features(features, self.target_dim)
        )
        return self.target_scaler.transform(reduced).astype(np.float32)

    def save_feature_aligner(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump({
                'target_dim': self.target_dim,
                'alignment_mode': self.alignment_mode,
                'source_reducer': self.source_reducer,
                'target_reducer': self.target_reducer,
                'source_scaler': self.source_scaler,
                'target_scaler': self.target_scaler,
                'source_descriptor_scaler': self.source_descriptor_scaler,
                'target_descriptor_scaler': self.target_descriptor_scaler,
                'random_state': self.random_state,
            }, handle)

    def load_feature_aligner(self, path: str):
        with open(path, 'rb') as handle:
            state = pickle.load(handle)
        self.target_dim = state['target_dim']
        self.alignment_mode = state.get('alignment_mode', 'svd')
        self.source_reducer = state['source_reducer']
        self.target_reducer = state['target_reducer']
        self.source_scaler = state['source_scaler']
        self.target_scaler = state['target_scaler']
        self.source_descriptor_scaler = state.get('source_descriptor_scaler')
        self.target_descriptor_scaler = state.get('target_descriptor_scaler')
        self.random_state = state.get('random_state', self.random_state)
        self.feature_aligner_fitted = True

    def align_features(self, source_features: np.ndarray,
                      target_features: np.ndarray,
                      method: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
        """
        Feature dimension alignment
        
        Args:
            source_features: Source domain features (DGraphFin: 17 dims)
            target_features: Target domain features (FiGraph: 772 dims)
            method: Alignment method ('linear', 'svd', 'mlp')
            
        Returns:
            Aligned (source features, target features)
        """
        print(f"\n Feature dimension alignment ({method} method)...")
        print(f"   - Source features: {source_features.shape}")
        print(f"   - Target features: {target_features.shape}")
        
        source_aligned, target_aligned = self.fit_feature_aligner(
            source_features, target_features
        )
        
        print(f"    Aligned dimensions: Source {source_aligned.shape}, Target {target_aligned.shape}")
        
        return source_aligned, target_aligned

    def prepare_zero_shot_data(self, source_year: int = 2014, 
                             target_data: str = 'dgraphfin') -> Dict:
        """
        Prepare zero-shot learning data
        
        Args:
            source_year: FiGraph source domain year
            target_data: Target dataset ('dgraphfin')
            
        Returns:
            Processed zero-shot data dictionary
        """
        
        figraph_data = self.load_figraph_data(source_year)
        
        if target_data == 'dgraphfin':
            dgraphfin_data = self.load_dgraphfin_data()
            
            snapshots = self.create_temporal_snapshots(dgraphfin_data)
            
            temporal_data = self.prepare_temporal_sequences(dgraphfin_data, snapshots)
            
            source_features = figraph_data['node_features']
            target_features = dgraphfin_data['node_features']
            
            source_aligned, target_aligned = self.align_features(source_features, target_features)
            
            source_adj = self._build_adjacency_matrix(figraph_data)
            target_adj = self._build_adjacency_matrix_from_edges(
                dgraphfin_data['edge_index'], dgraphfin_data['num_nodes']
            )
            
            return {
                'source': {
                    'features': source_aligned,
                    'labels': figraph_data['labels'],
                    'adj': source_adj,
                    'dataset': f'FiGraph-{source_year}'
                },
                'target': {
                    'features': target_aligned,
                    'labels': dgraphfin_data['labels'],
                    'adj': target_adj,
                    'temporal_data': temporal_data,  # Temporal data for TCN
                    'snapshots': snapshots,
                    'dataset': 'DGraphFin'
                },
                'config': {
                    'source_dim': source_features.shape[1],
                    'target_dim': target_aligned.shape[1],
                    'temporal_window': self.temporal_window,
                    'residual_dim': self.residual_dim
                }
            }

    def _build_adjacency_matrix(self, figraph_data: Dict) -> sp.csr_matrix:
        """Build FiGraph adjacency matrix"""
        edges_df = figraph_data['edges']
        num_nodes = figraph_data['num_nodes']
        
        # Create node ID mapping
        unique_nodes = set(edges_df['source'].unique()) | set(edges_df['target'].unique())
        node_mapping = {node: i for i, node in enumerate(unique_nodes)}
        
        # Map edge indices
        source_mapped = [node_mapping.get(node, 0) for node in edges_df['source']]
        target_mapped = [node_mapping.get(node, 0) for node in edges_df['target']]
        
        # Build adjacency matrix
        adj_matrix = sp.csr_matrix(
            (np.ones(len(edges_df)), (source_mapped, target_mapped)),
            shape=(len(unique_nodes), len(unique_nodes))
        )
        
        return adj_matrix
    
    def _build_adjacency_matrix_from_edges(self, edge_index: np.ndarray, 
                                          num_nodes: int) -> sp.csr_matrix:
        """Build adjacency matrix from edge index"""
        adj_matrix = sp.csr_matrix(
            (np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
            shape=(num_nodes, num_nodes)
        )
        return adj_matrix

    def save_processed_data(self, data: Dict, save_path: str):
        """Save processed data"""
        print(f"\n Save processed data to: {save_path}")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"    Data save completed")

    def load_processed_data(self, save_path: str) -> Dict:
        """Load processed data"""
        print(f"\n Load data from file: {save_path}")
        
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"    Data load completed")
        return data

    @staticmethod
    def calculate_precision_recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, 
                                       k_values: list = [100, 500, 1000, 5000]) -> dict:
        """
        Calculate Precision@k and Recall@k metrics
        
        Args:
            y_true: Ground truth labels (0/1)
            y_scores: Prediction scores (continuous values)
            k_values: List of k values to compute
            
        Returns:
            Dictionary containing precision and recall for each k
        """
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_labels = y_true[sorted_indices]
        
        total_positives = np.sum(y_true)
        
        results = {}
        
        for k in k_values:
            if k > len(y_true):
                print(f"    k={k} exceeds total samples {len(y_true)}, skipping")
                continue
                
            top_k_positives = np.sum(sorted_labels[:k])
            
            precision_k = top_k_positives / k
            
            recall_k = top_k_positives / total_positives if total_positives > 0 else 0
            
            results[f'precision@{k}'] = precision_k
            results[f'recall@{k}'] = recall_k
            
            print(f"    Precision@{k}: {precision_k:.4f} | Recall@{k}: {recall_k:.4f}")
            
        return results
    
    @staticmethod
    def evaluate_ranking_performance(y_true: np.ndarray, y_scores: np.ndarray, 
                                   dataset_name: str = "Unknown") -> dict:
        """
        Comprehensive ranking performance evaluation, including traditional metrics and business-critical metrics
        
        Args:
            y_true: Ground truth labels
            y_scores: Prediction scores
            dataset_name: Dataset name
            
        Returns:
            Complete evaluation results dictionary
        """
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
        
        print(f"\n {dataset_name} Ranking Performance Evaluation:")
        print(f"    Total samples: {len(y_true):,}")
        print(f"    Positive samples: {np.sum(y_true):,}")
        print(f"    Positive ratio: {np.mean(y_true):.6f} ({np.mean(y_true)*100:.4f}%)")
        
        # Basic metrics
        try:
            auc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            print(f"    ROC-AUC: {auc:.4f}")
            print(f"    Average Precision (AP): {ap:.4f}")
        except Exception as e:
            print(f"    Basic metrics computation failed: {e}")
            auc, ap = 0, 0
        
        # Random baseline comparison
        random_ap_baseline = np.mean(y_true)
        ap_improvement = ap / random_ap_baseline if random_ap_baseline > 0 else 0
        print(f"    AP improvement: {ap_improvement:.2f}x (vs random baseline {random_ap_baseline:.6f})")
        
        # Precision@k and Recall@k
        print(f"\n    Precision@k and Recall@k Evaluation:")
        precision_recall_results = AnomalyGFMDataProcessor.calculate_precision_recall_at_k(
            y_true, y_scores, k_values=[100, 500, 1000, 5000, 10000]
        )
        
        # Summarize results
        results = {
            'dataset': dataset_name,
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'positive_ratio': float(np.mean(y_true)),
            'roc_auc': float(auc),
            'average_precision': float(ap),
            'random_baseline_ap': float(random_ap_baseline),
            'ap_improvement': float(ap_improvement),
            **precision_recall_results
        }
        
        return results


def test_data_processor():
    """Test data processor"""
    config = {
        'temporal_window': 7,
        'residual_dim': 64,
        'target_dim': 128,
        'use_svd': True,
        'dgraphfin_path': 'data/DGraphFin2/',
        'figraph_path': 'data/FiGraph-main/data/'
    }
    
    processor = AnomalyGFMDataProcessor(config)
    
    # Prepare zero-shot data
    zero_shot_data = processor.prepare_zero_shot_data(
        source_year=2014, 
        target_data='dgraphfin'
    )
    
    # Save data
    processor.save_processed_data(
        zero_shot_data, 
        'processed_data/anomalygfm_zero_shot_2014.pkl'
    )
    


if __name__ == "__main__":
    test_data_processor()
