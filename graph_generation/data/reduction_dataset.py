from abc import ABC

import numpy as np
import scipy as sp
import torch as th
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor

from ..reduction import ReductionFactory

class RandRedDataset(IterableDataset, ABC):
    def __init__(self, hypergraphs, red_factory: ReductionFactory, spectrum_extractor):
        super().__init__()

        self.red_factory = red_factory
        self.hypergraphs = hypergraphs
        self.spectrum_extractor = spectrum_extractor

    def get_random_reduction_sequence(self, reduction_level_zero, rng):
        data = []
        red = reduction_level_zero
        red_next = red.get_reduced_hypergraph(rng)
        
        while True:
            reduced_hypergraph = red_next.get_reduced_hypergraph(rng)
            
            node_type = np.zeros(red.bipartite_adj.shape[0])
            node_type[:red.n] = 1
            
            node_type_reduced = np.zeros(red_next.bipartite_adj.shape[0])
            node_type_reduced[:red_next.n] = 1
            
            diff_node_features = None
            diff_hyperedge_features = None
            if reduced_hypergraph.node_features is not None or reduced_hypergraph.hyperedge_features is not None:
                if reduced_hypergraph.node_features is not None:
                    # Reshape features to 2D for sparse matrix multiplication
                    original_shape = reduced_hypergraph.node_features.shape
                    reshaped_features = reduced_hypergraph.node_features.reshape((original_shape[0], -1))  # (n, d1*d2*...)
                    
                    node_features_reduced = reduced_hypergraph.expansion_matrix_nodes @ reshaped_features
                    
                    # Reshape back to the original feature dimensions
                    node_features_reduced = node_features_reduced.reshape((reduced_hypergraph.expansion_matrix_nodes.shape[0],) + original_shape[1:])  # (k, d1, d2, ...)
                    
                    diff_node_features = np.abs(node_features_reduced - red_next.node_features)
                
                if reduced_hypergraph.hyperedge_features is not None:
                    # Reshape features to 2D for sparse matrix multiplication
                    original_shape = reduced_hypergraph.hyperedge_features.shape
                    reshaped_features = reduced_hypergraph.hyperedge_features.reshape((original_shape[0], -1))  # (n, d1*d2*...)
                    
                    hyperedge_features_reduced = reduced_hypergraph.expansion_matrix_hyperedges @ reshaped_features
                    
                    # Reshape back to the original feature dimensions
                    hyperedge_features_reduced = hyperedge_features_reduced.reshape((reduced_hypergraph.expansion_matrix_hyperedges.shape[0],) + original_shape[1:])  # (k, d1, d2, ...)
                    
                    diff_hyperedge_features = np.abs(hyperedge_features_reduced - red_next.hyperedge_features)

            data.append(
                ReducedGraphData(
                    target_size=reduction_level_zero.n,
                    reduction_level=red.level,
                    adj=red.bipartite_adj.astype(bool).astype(np.float32),
                    node_type=node_type,
                    node_features=red.node_features,
                    node_cluster_size=red.node_cluster_size,
                    hyperedge_features=red.hyperedge_features,
                    node_expansion=red.node_expansion,
                    edge_expansion=red.edge_expansion,
                    adj_reduced=red_next.bipartite_adj.astype(bool).astype(np.float32),
                    node_type_reduced=node_type_reduced,
                    node_features_reduced=red_next.node_features,
                    node_features_diff=diff_node_features,
                    node_cluster_size_reduced=red_next.node_cluster_size,
                    hyperedge_features_reduced=red_next.hyperedge_features,
                    hyperedge_features_diff=diff_hyperedge_features,
                    expansion_matrix=red_next.expansion_matrix,
                    expansion_matrix_nodes=red_next.expansion_matrix_nodes,
                    expansion_matrix_hyperedges=red_next.expansion_matrix_hyperedges,
                    spectral_features_reduced=self.spectrum_extractor(red_next.bipartite_adj)
                    if self.spectrum_extractor is not None
                    else None,
                )
            )
            
            if red.n <= 1:
                break
            
            red = red_next
            red_next = reduced_hypergraph

        return data


class FiniteRandRedDataset(RandRedDataset):
    def __init__(
        self, hypergraphs, red_factory: ReductionFactory, spectrum_extractor, num_red_seqs
    ):
        super().__init__(hypergraphs, red_factory, spectrum_extractor)
        self.num_red_seqs = num_red_seqs

        self.rng = np.random.default_rng(seed=0)
        self.hypergraphs_reduced_data = {i: [] for i in range(len(hypergraphs))}
        for i, hypergraph in enumerate(hypergraphs):
            red = red_factory(hypergraph)
            for _ in range(num_red_seqs):
                self.hypergraphs_reduced_data[i] += self.get_random_reduction_sequence(
                    red, self.rng
                )

    def __iter__(self):
        while True:
            i = self.rng.integers(len(self.hypergraphs))
            j = self.rng.integers(len(self.hypergraphs_reduced_data[i]))
            yield self.hypergraphs_reduced_data[i][j]

    @property
    def max_node_expansion(self):
        return max(
            [
                rgd.node_expansion.max().item()
                for seq in self.hypergraphs_reduced_data
                for rgd in seq
            ]
        )


class InfiniteRandRedDataset(RandRedDataset):
    def __iter__(self):
        reds = [self.red_factory(hypergraph) for hypergraph in self.hypergraphs]
        
        # get process id
        worker_info = th.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(worker_id)

        # initialize graph_reduced_data
        hypergraphs_reduced_data = {
            i: self.get_random_reduction_sequence(hypergraph, rng)
            for i, hypergraph in enumerate(reds)
        }

        # yield random reduced graph data
        while True:
            i = rng.integers(len(self.hypergraphs))
            if len(hypergraphs_reduced_data[i]) == 0:
                hypergraphs_reduced_data[i] = self.get_random_reduction_sequence(
                    reds[i], rng
                )
                rng.shuffle(hypergraphs_reduced_data[i])
            yield hypergraphs_reduced_data[i].pop()

    @property
    def max_node_expansion(self):
        raise NotImplementedError


class ReducedGraphData(Data):
    def __init__(self, **kwargs):
        if not kwargs:
            super().__init__()
            return

        super().__init__(x=th.zeros(kwargs["adj"].shape[0]))
        for key, value in kwargs.items():
            if value is None:
                continue
            elif isinstance(value, int):
                value = th.tensor(value).type(th.long)
            elif isinstance(value, np.ndarray):
                value = th.from_numpy(value).type(
                    th.float32 if np.issubdtype(value.dtype, np.floating) else th.long
                )
            elif isinstance(value, sp.sparse.sparray):
                value = SparseTensor.from_scipy(value).type(
                    th.float32 if np.issubdtype(value.dtype, np.floating) else th.long
                )
            else:
                raise ValueError(f"Unsupported type {type(value)} for key {key}")

            setattr(self, key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)  # concatenate along diagonal
        return super().__cat_dim__(key, value, *args, **kwargs)
