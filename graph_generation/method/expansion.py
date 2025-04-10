import hypernetx as hnx
import torch as th
from torch.nn import Module
from torch_geometric.utils import to_edge_index
from torch_scatter import scatter
from torch_sparse import SparseTensor
import numpy as np

from .method import Method


class Expansion(Method):
    """Hypergraph generation method generating graphs by local expansion."""  

    def __init__(
        self,
        diffusion,
        spectrum_extractor,
        emb_features,
        augmented_radius=1,
        augmented_dropout=0.0,
        deterministic_expansion=False,
        min_red_frac=0.0,
        max_red_frac=0.5,
        red_threshold=0,
        features_noise_perturbation=0
    ):
        self.diffusion = diffusion
        self.spectrum_extractor = spectrum_extractor
        self.emb_features = emb_features
        self.augmented_radius = augmented_radius
        self.augmented_dropout = augmented_dropout
        self.deterministic_expansion = deterministic_expansion
        self.min_red_frac = min_red_frac
        self.max_red_frac = max_red_frac
        self.red_threshold = red_threshold
        self.features_noise_perturbation = features_noise_perturbation

    def sample_hypergraphs(self, target_size, model: Module, sign_net: Module, node_features_dim: tuple, hyperedge_features_dim: tuple):
        """Samples a batch of hypergraphs."""
        num_hypergraphs = len(target_size)
        
        dense_tensor = th.zeros(num_hypergraphs*2, num_hypergraphs*2, device=self.device)
        dense_tensor[th.arange(num_hypergraphs, device=self.device)*2, 1 + th.arange(num_hypergraphs, device=self.device)*2] = 1
        dense_tensor[1 + th.arange(num_hypergraphs, device=self.device)*2, th.arange(num_hypergraphs, device=self.device)*2] = 1
        
        adj = SparseTensor.from_dense(dense_tensor)

        batch = th.repeat_interleave(th.arange(0, num_hypergraphs, device=self.device), 2*th.ones(num_hypergraphs, dtype = th.int32, device=self.device))
        node_expansion = th.ones(num_hypergraphs*2, dtype=th.long, device=self.device)
        node_type = th.ones(num_hypergraphs*2, dtype=th.int, device=self.device)
        node_cluster_size = target_size
        node_type[1::2] = 0
        
        node_features = None
        hyperedge_features = None
        
        if node_features_dim:
            node_features = th.zeros((num_hypergraphs,) + (node_features_dim if isinstance(node_features_dim, tuple) else (node_features_dim,)), device=self.device, dtype=th.float32)
            
        if hyperedge_features_dim:
            hyperedge_features = th.zeros((num_hypergraphs,) + (hyperedge_features_dim if isinstance(hyperedge_features_dim, tuple) else (hyperedge_features_dim,)), device=self.device)
        
        while node_type.sum() < target_size.sum():
            adj, batch, node_expansion, node_type, node_features, hyperedge_features, node_cluster_size = self.expand(
                adj,
                batch,
                node_expansion,
                node_type,
                node_features,
                hyperedge_features,
                node_cluster_size,
                target_size,
                model=model,
                sign_net=sign_net,
            )
            
            if node_expansion[node_type == 1].max() <= 1:
                break
                
        # return hypergraphs
        if node_features is not None:
            node_features = node_features.cpu().numpy()
            
        if hyperedge_features is not None:
            hyperedge_features = hyperedge_features.cpu().numpy()
        
        adjs, num_nodes, node_features_unbatched, hyperedge_features_unbatched = unbatch_adj(adj, batch, node_type, node_features, hyperedge_features)
        hypergraphs = []
        
        for i in range(len(adjs)):
            adj = adjs[i]
            n = num_nodes[i]
            
            # remind that we have :
            # Adj_bipartite = ( 0  H)
            #                 (H^T 0)
            adj = adj.to_dense().cpu().numpy()
            
            if np.max(adj) > 0.:
                num_hyperedges = adj.shape[0] - n
    
                incidence_matrix = adj[:n, n:n + num_hyperedges]
                
                H = hnx.Hypergraph.from_incidence_matrix(incidence_matrix)
                hypergraphs.append(H)
            else:
                H = hnx.Hypergraph(np.ones((2,2)))
                hypergraphs.append(H)
                
            if node_features_unbatched is not None:
                node_features = node_features_unbatched[i]
                H.add_nodes_from([ (node, {"feature": node_features[i]}) for (i, node) in enumerate(H.nodes)])
            
            if hyperedge_features_unbatched is not None:
                hyperedge_features = hyperedge_features_unbatched[i]
                H.add_edges_from([ (edge, {"feature": hyperedge_features[i]}) for (i, edge) in enumerate(H.edges)])
        
        return hypergraphs

    @th.no_grad()
    def expand(
        self,
        adj_reduced,
        batch_reduced,
        node_expansion,
        node_type,
        node_features,
        hyperedge_features,
        node_cluster_size,
        target_size,
        model: Module,
        sign_net: Module,
    ):
        """Expands a hypergraph by a single level."""
        # expand
        # don't expand nodes that doesn't have any budget left
        all_node_map = th.repeat_interleave(
            th.arange(0, adj_reduced.size(0), device=self.device), node_expansion
        )
        
        node_map = th.repeat_interleave(
            th.arange(0, th.sum(node_type), device=self.device), node_expansion[node_type == 1]
        )
        
        edge_node_map = th.repeat_interleave(
            th.arange(0, node_type.size(0) - th.sum(node_type), device=self.device), node_expansion[node_type == 0]
        )
        
        expanded_node_type = node_type[all_node_map]
        expanded_node_cluster_size = node_cluster_size[node_map]
        batch = batch_reduced[all_node_map]
        
        expanded_node_features = None
        expanded_hyperedge_features = None
        
        if node_features is not None:
            expanded_node_features = node_features[node_map]
        
        if hyperedge_features is not None:
            expanded_hyperedge_features = hyperedge_features[edge_node_map]
        
        size = scatter(expanded_node_type, batch)
        expansion_matrix = SparseTensor(
            row=th.arange(all_node_map.size(0), device=self.device),
            col=all_node_map,
            value=th.ones(all_node_map.size(0), device=self.device),
        )
        adj_augmented = self.get_augmented_hypergraph(adj_reduced, expansion_matrix)
        augmented_edge_index = th.stack(adj_augmented.coo()[:2], dim=0)
        
        # get node embeddings
        if self.spectrum_extractor is not None:
            spectral_features = th.cat(
                [
                    th.tensor(
                        self.spectrum_extractor(adj.to("cpu").to_scipy(layout="coo")),
                        dtype=th.float32,
                        device=self.device,
                    )
                    for adj in unbatch_adj(adj_reduced, batch_reduced, node_type, node_features, hyperedge_features)[0]
                ]
            )
            both_type_node_emb_reduced = sign_net(
                spectral_features=spectral_features, edge_index=adj_reduced
            )
        else:
            both_type_node_emb_reduced = th.randn(
                node_type.size(0), self.emb_features, device=self.device
            )
            
        both_type_node_emb = both_type_node_emb_reduced[all_node_map]
        node_emb = both_type_node_emb[expanded_node_type == 1]
        edge_node_emb = both_type_node_emb[expanded_node_type == 0]
        
        # compute number of nodes in expanded hypergraph
        random_reduction_fraction = (
            th.rand(len(target_size), device=self.device)
            * (self.max_red_frac - self.min_red_frac)
            + self.min_red_frac
        )

        # if expanded number of nodes is less than threshold, use max_red_frac
        max_reduction_mask = (
            th.ceil(size / (1 - self.max_red_frac)) <= self.red_threshold
        ).float()
        random_reduction_fraction = (
            1 - max_reduction_mask
        ) * random_reduction_fraction + max_reduction_mask * self.max_red_frac

        # expanded number of nodes is ⌈n / (1-r)⌉ and at least n+1 and at most target_size
        expanded_size = th.minimum(
            th.maximum(
                th.ceil(size / (1 - random_reduction_fraction)).long(),
                size + 1,
            ),
            target_size,
        )
        
        # make predictions
        node_pred, edge_node_pred, augmented_edge_pred, node_features_pred, hyperedge_features_pred = self.diffusion.sample(
            edge_index=augmented_edge_index,
            batch=batch,
            node_type=expanded_node_type,
            node_features=expanded_node_features,
            hyperedge_features=expanded_hyperedge_features,
            node_cluster_size=expanded_node_cluster_size,
            expansion_matrix_nodes=expansion_matrix[expanded_node_type == 1, :][:, node_type == 1],
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "edge_node_emb": edge_node_emb,
                "red_frac": 1 - size / expanded_size,
                "target_size": target_size.float(),
            },
        )
        
        # rescale
        node_cluster_size_pred = (node_pred[:, 1] + 1)/2
              
        print(f"expanded_node_cluster_size : {expanded_node_cluster_size}")
        print(f"node_cluster_size_pred : {node_cluster_size_pred}")
        print(f"node_features_pred : {node_features_pred}")
        
        node_pred = (node_pred[:, 0] + 1)/2
        edge_node_pred = edge_node_pred + 1
        augmented_edge_pred = (augmented_edge_pred + 1)/2

        # get node attributes
        if self.deterministic_expansion:
            # Compute the predicted share of the parent's budget for each child
            node_cluster_size_frac = th.clamp(node_cluster_size_pred, min=0) / scatter(th.clamp(node_cluster_size_pred, min=0), node_map)[node_map]
            
            # Budget is at least 1
            node_cluster_size = th.clamp((expanded_node_cluster_size * node_cluster_size_frac).round().long(), min=1)
            
            # Handle tie case (frac predicted is exactly 0.5)
            tie_mask = (node_cluster_size_frac == 0.5)
            if tie_mask.any():
                # Get the parent cluster size for the tied nodes
                parent_cluster_size = expanded_node_cluster_size[tie_mask]
                
                # Split the parent cluster size into two halves
                half_size = parent_cluster_size // 2
                remaining_size = parent_cluster_size - half_size
                
                # Assign the halves to the tied nodes
                node_cluster_size[tie_mask] = th.where(
                    th.arange(tie_mask.sum(), device=self.device) % 2 == 0,  # Alternate between nodes
                    half_size,
                    remaining_size
                )
            
            # If a node comes from an expanded cluster, it can have at most the parent's budget - 1
            node_counts = th.bincount(node_map)  # Count occurrences of each node ID in node_map
            is_expanded = node_counts[node_map] > 1  # True for nodes from expanded clusters
            
            node_cluster_size_max = th.where(
                is_expanded,
                expanded_node_cluster_size - 1,  # Subtract 1 if expanded
                expanded_node_cluster_size       # Keep as is otherwise
            )
            
            node_cluster_size = th.minimum(node_cluster_size, node_cluster_size_max)            
            
            # Don't expand nodes having no budget
            node_pred[node_cluster_size == 1] = -1 # not zero to be sure that they are not selected during the topk
            
            node_attr = th.ones_like(node_pred, dtype=th.long)
            num_new_nodes = expanded_size - size
            
            node_range_end = size.cumsum(0)
            node_range_start = node_range_end - size
            # get top-k nodes per graph
            for i in range(len(target_size)):
                new_node_idx = (
                    th.topk(
                        node_pred[node_range_start[i] : node_range_end[i]],
                        num_new_nodes[i],
                        largest=True,
                    )[1]
                    + node_range_start[i]
                )
                node_attr[new_node_idx] = 2
            node_attr[node_cluster_size == 1] = 1
        else:
            node_attr = (node_pred > 0.5).long() + 1
            node_cluster_size = th.maximum(node_cluster_size_pred.round().long(), th.ones_like(node_cluster_size_pred).long())
            
        edge_node_attr = (edge_node_pred > 0.66).long() + (edge_node_pred > 1.33).long() + 1
        
        # construct new hypergraph
        adj = SparseTensor.from_edge_index(
            augmented_edge_index[:, augmented_edge_pred > 0.5],
            sparse_sizes=adj_augmented.sizes(),
            edge_attr = th.ones(th.sum(augmented_edge_pred > 0.5), device=augmented_edge_pred.device)
        )
        
        all_node_attr = th.zeros(batch.size(0), dtype=th.long, device = node_attr.device)
        all_node_attr[expanded_node_type == 0] = edge_node_attr
        all_node_attr[expanded_node_type == 1] = node_attr
        
        return adj, batch, all_node_attr, expanded_node_type, node_features_pred, hyperedge_features_pred, node_cluster_size

    def get_loss(self, batch, model: Module, sign_net: Module):
        """Returns a weighted sum of the node and edge expansion loss and the augmented edge loss."""
        # get augmented hypergraph
        adj_augmented = self.get_augmented_hypergraph(
            batch.adj_reduced, batch.expansion_matrix
        )

        # construct labels
        node_attr = (batch.node_expansion.float() - 1) * 2 - 1
        edge_node_attr = batch.edge_expansion.float() - 2
        augmented_edge_index, edge_val = to_edge_index(adj_augmented + batch.adj)
        augmented_edge_attr = (edge_val.float() - 1) * 2 - 1
        
        node_cluster_size_expanded = (batch.expansion_matrix_nodes @ th.unsqueeze(batch.node_cluster_size_reduced,-1)).squeeze(1).float()
        
        # get node embeddings
        if sign_net is not None:
            both_type_node_emb_reduced = sign_net(
                spectral_features=batch.spectral_features_reduced,
                edge_index=batch.adj_reduced,
            )
        else:
            both_type_node_emb_reduced = th.randn(
                batch.node_type.size(0), self.emb_features, device=self.device
            )
                
        both_type_node_emb = batch.expansion_matrix @ both_type_node_emb_reduced
        node_emb = both_type_node_emb[batch.node_type == 1]
        edge_node_emb = both_type_node_emb[batch.node_type == 0]
        
        node_features = None
        if hasattr(batch, 'node_features'):
            if self.features_noise_perturbation > 0:
                node_features_reduced_noised = batch.node_features_reduced + 0.3*th.randn_like(batch.node_features_reduced) #self.features_noise_perturbation*batch.node_features_diff*
            else:
                node_features_reduced_noised = batch.node_features_reduced
                
            # Reshape features to 2D for sparse matrix multiplication
            original_shape = node_features_reduced_noised.size()
            reshaped_features = node_features_reduced_noised.reshape((original_shape[0], -1))  # (n, d1*d2*...)
            
            node_features = batch.expansion_matrix_nodes @ reshaped_features
            
            # Reshape back to the original feature dimensions
            node_features = node_features.reshape((batch.expansion_matrix_nodes.size(0),) + original_shape[1:])  # (k, d1, d2, ...)
            
        hyperedge_features = None
        if hasattr(batch, 'hyperedge_features'):
            if self.features_noise_perturbation > 0:
                hyperedge_features_reduced_noised = batch.hyperedge_features_reduced + self.features_noise_perturbation*batch.hyperedge_features_diff*th.randn_like(batch.hyperedge_features_reduced)
            else:
                hyperedge_features_reduced_noised = batch.hyperedge_features_reduced
                
            # Reshape features to 2D for sparse matrix multiplication
            original_shape = hyperedge_features_reduced_noised.size()
            reshaped_features = hyperedge_features_reduced_noised.reshape((original_shape[0], -1))  # (n, d1*d2*...)
            
            hyperedge_features = batch.expansion_matrix_hyperedges @ reshaped_features
            
            # Reshape back to the original feature dimensions
            hyperedge_features = hyperedge_features.reshape((batch.expansion_matrix_hyperedges.size(0),) + original_shape[1:])  # (k, d1, d2, ...)
            
            
        # reduction fraction
        size = scatter(batch.node_type, batch.batch)
        expanded_size = scatter(batch.node_expansion, batch.batch[batch.node_type == 1])
        red_frac = 1 - size / expanded_size
        
        # loss
        node_loss, edge_node_loss, edge_loss, node_features_loss, hyperedge_features_loss = self.diffusion.get_loss(
            edge_index=augmented_edge_index,
            batch=batch.batch,
            node_type=batch.node_type,
            expansion_matrix_nodes=batch.expansion_matrix_nodes,
            expansion_matrix_hyperedges=batch.expansion_matrix_hyperedges,
            node_attr=node_attr,
            edge_node_attr=edge_node_attr,
            edge_attr=augmented_edge_attr,
            node_features_expanded=node_features,
            node_features_real=batch.node_features if node_features is not None else None,
            node_cluster_size_expanded=node_cluster_size_expanded,
            node_cluster_size_real=batch.node_cluster_size,
            hyperedge_features_expanded=hyperedge_features,
            hyperedge_features_real=batch.hyperedge_features if hyperedge_features is not None else None,
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "edge_node_emb": edge_node_emb,
                "red_frac": red_frac,
                "target_size": batch.target_size.float(),
            },
        )

        loss = node_loss + edge_node_loss + edge_loss
        
        res = {
            "node_expansion_loss": node_loss.item(),
            "edge_node_expansion_loss": edge_node_loss.item(),
            "augmented_edge_loss": edge_loss.item(),
        }
        
        if hasattr(batch, 'node_features'):
            node_features_loss = node_features_loss.mean()
            loss += node_features_loss
            res["node_features_loss"] = node_features_loss.item()
        
        if hasattr(batch, 'hyperedge_features'):
            hyperedge_features_loss = hyperedge_features_loss.mean()
            loss += hyperedge_features_loss
            res["hyperedge_features_loss"] = hyperedge_features_loss.item()
            
        res["loss"] = loss.item()
            
        return loss, res

    def get_augmented_hypergraph(self, adj_reduced, expansion_matrix):
        """Returns the expanded bipartite adjacency matrix with additional augmented edges.

        All edge weights are set to 1.
        """
        # construct augmented adjacency matrix
        adj_reduced_augmented = adj_reduced.copy()
        
        if self.augmented_radius > 1 and self.augmented_dropout < 1:
            adj_reduced_square = (adj_reduced @ adj_reduced).set_diag(1)
            for _ in range(1, self.augmented_radius):
                adj_reduced_augmented = adj_reduced_augmented @ adj_reduced_square

            adj_reduced_augmented = adj_reduced_augmented.set_value(
                th.ones(adj_reduced_augmented.nnz(), device=self.device), layout="coo"
            )
            
            adj_reduced_augmented = adj_reduced_augmented + adj_reduced
            
        adj_augmented = (
            expansion_matrix @ adj_reduced_augmented @ expansion_matrix.t()
        )
        
        # drop out edges
        if self.augmented_radius > 1 and self.augmented_dropout < 1:
            if self.augmented_dropout > 0.0:
                row, col, val = adj_augmented.coo()
                edge_mask = th.rand_like(val) >= self.augmented_dropout
                edge_mask = edge_mask | (val > 1)  # keep required edges
                
                # make undirected
                edge_mask = edge_mask & (row < col)
                edge_index = th.stack([row[edge_mask], col[edge_mask]], dim=0)
                edge_index = th.cat([edge_index, edge_index.flip(0)], dim=1)
                adj_augmented = SparseTensor.from_edge_index(
                    edge_index,
                    edge_attr=th.ones(edge_index.shape[1], device=self.device),
                    sparse_sizes=adj_augmented.sizes(),
                )
            else:
                adj_augmented.set_value(
                    th.ones(adj_augmented.nnz(), device=self.device), layout="coo"
                )
                
        return adj_augmented


def unbatch_adj(adj, batch, node_type, node_features, hyperedge_features) -> tuple:
    size = scatter(th.ones_like(batch), batch)
    
    hypergraph_end_idx = size.cumsum(0)
    hypergraph_start_idx = hypergraph_end_idx - size
    
    n_nodes = [th.sum(node_type[hypergraph_start_idx[i] : hypergraph_end_idx[i]]).item() for i in range(len(size))]
    
    node_features_unbatched = None
    hyperedge_features_unbatched = None
    
    if node_features is not None:
        node_features_end_idx = np.cumsum(n_nodes)
        node_features_start_idx = node_features_end_idx - n_nodes
        node_features_unbatched = [node_features[node_features_start_idx[i] : node_features_end_idx[i]] for i in range(len(n_nodes))]
    
    if hyperedge_features is not None:
        n_hyperedges = [th.sum(1 - node_type[hypergraph_start_idx[i] : hypergraph_end_idx[i]]).item() for i in range(len(size))]
        
        hyperedge_features_end_idx = np.cumsum(n_hyperedges)
        hyperedge_features_start_idx = hyperedge_features_end_idx - n_hyperedges
        hyperedge_features_unbatched = [hyperedge_features[hyperedge_features_start_idx[i] : hyperedge_features_end_idx[i]] for i in range(len(n_hyperedges))]
    
    return [ adj[hypergraph_start_idx[i] : hypergraph_end_idx[i], :][:, hypergraph_start_idx[i] : hypergraph_end_idx[i] ]
    for i in range(len(size)) ], n_nodes, node_features_unbatched, hyperedge_features_unbatched
