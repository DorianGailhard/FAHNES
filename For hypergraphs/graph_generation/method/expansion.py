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
        emb_dim,
        augmented_radius=1,
        augmented_dropout=0.0,
        deterministic_expansion=False,
        min_red_frac=0.0,
        max_red_frac=0.5,
        red_threshold=0,
        node_features_noise_strength=0,
        node_features_on_simplex=False,
        hyperedge_features_noise_strength=0,
        hyperedge_features_on_simplex=False
    ):
        self.diffusion = diffusion
        self.spectrum_extractor = spectrum_extractor
        self.emb_dim = emb_dim
        self.augmented_radius = augmented_radius
        self.augmented_dropout = augmented_dropout
        self.deterministic_expansion = deterministic_expansion
        self.min_red_frac = min_red_frac
        self.max_red_frac = max_red_frac
        self.red_threshold = red_threshold
        self.node_features_noise_strength = node_features_noise_strength
        self.node_features_on_simplex = node_features_on_simplex
        self.hyperedge_features_noise_strength = hyperedge_features_noise_strength
        self.hyperedge_features_on_simplex = hyperedge_features_on_simplex

    def sample_hypergraphs(self, target_size, model: Module, sign_net: Module, node_features_dim: tuple, hyperedge_features_dim: tuple):
        """Samples a batch of hypergraphs."""
        num_hypergraphs = len(target_size)
        
        idx = th.arange(num_hypergraphs, device=self.device)
        row = th.cat([2 * idx, 2 * idx + 1])
        col = th.cat([2 * idx + 1, 2 * idx])
        adj = SparseTensor(
            row=row,
            col=col,
            value=th.ones_like(row, dtype=th.float),
            sparse_sizes=(2 * num_hypergraphs, 2 * num_hypergraphs)
        )

        batch = th.repeat_interleave(
            th.arange(0, num_hypergraphs, device=self.device),
            2*th.ones(num_hypergraphs, dtype = th.int32, device=self.device)
        )
        node_expansion = th.ones(num_hypergraphs*2, dtype=th.long, device=self.device)
        node_type = th.ones(num_hypergraphs*2, dtype=th.int, device=self.device)
        budget = target_size
        node_type[1::2] = 0
        
        node_features = None
        hyperedge_features = None
        
        if node_features_dim:
            shape = node_features_dim if isinstance(node_features_dim, tuple) else (node_features_dim,)
            node_features = th.zeros((num_hypergraphs,) + shape, device=self.device, dtype=th.float32)
            
        if hyperedge_features_dim:
            shape = hyperedge_features_dim if isinstance(hyperedge_features_dim, tuple) else (hyperedge_features_dim,)
            hyperedge_features = th.zeros((num_hypergraphs,) + shape, device=self.device, dtype=th.float32)
            
        while node_type.sum() < target_size.sum():
            adj, batch, node_expansion, node_type, node_features, hyperedge_features, budget = self.expand(
                adj,
                batch,
                node_expansion,
                node_type,
                node_features,
                hyperedge_features,
                budget,
                target_size,
                model=model,
                sign_net=sign_net,
            )
            
            if node_expansion[node_type == 1].max() <= 1:
                break
                
        # return hypergraphs
        adjs, num_nodes, node_feats, hyperedge_feats = unbatch_adj(
            adj, batch, node_type, node_features, hyperedge_features
        )
        hypergraphs = []
        
        for i in range(len(adjs)):
            adj = adjs[i]
            n = num_nodes[i]
            
            # remember that we have :
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
                
            if node_feats is not None:
                node_features = node_feats[i].cpu().numpy()
                H.add_nodes_from([
                    (node, {"feature": node_features[i]})
                    for (i, node) in enumerate(H.nodes)
                ])
            
            if hyperedge_feats is not None:
                hyperedge_features = hyperedge_feats[i].cpu().numpy()
                H.add_edges_from([
                    (edge, {"feature": hyperedge_features[i]})
                    for (i, edge) in enumerate(H.edges)
                ])
        
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
        budget,
        target_size,
        model: Module,
        sign_net: Module,
    ):
        """Expands a hypergraph by a single level."""
        # expand
        all_node_map = th.repeat_interleave(
            th.arange(0, adj_reduced.size(0), device=self.device),
            node_expansion
        )
        
        node_map = th.repeat_interleave(
            th.arange(0, th.sum(node_type), device=self.device),
            node_expansion[node_type == 1]
        )
        
        hyperedge_map = th.repeat_interleave(
            th.arange(0, node_type.size(0) - th.sum(node_type), device=self.device),
            node_expansion[node_type == 0]
        )
        
        expanded_node_type = node_type[all_node_map]
        expanded_budget = budget[node_map]
        batch = batch_reduced[all_node_map]
        
        expanded_node_features = None
        expanded_hyperedge_features = None
        
        if node_features is not None:
            expanded_node_features = node_features[node_map]
        
        if hyperedge_features is not None:
            expanded_hyperedge_features = hyperedge_features[hyperedge_map]
        
        size = scatter(expanded_node_type, batch)
        expansion_matrix = SparseTensor(
            row=th.arange(all_node_map.size(0), device=self.device),
            col=all_node_map,
            value=th.ones(all_node_map.size(0), device=self.device),
        )
        adj_augmented = self.get_augmented_hypergraph(adj_reduced, expansion_matrix)
        augmented_incidence_index = th.stack(adj_augmented.coo()[:2], dim=0)
        
        # get node embeddings
        if self.spectrum_extractor is not None:
            spectral_features = th.cat(
                [
                    th.tensor(
                        self.spectrum_extractor(adj.to("cpu").to_scipy(layout="coo")),
                        dtype=th.float32,
                        device=self.device,
                    )
                    for adj in unbatch_adj(
                            adj_reduced, batch_reduced, node_type,
                            node_features, hyperedge_features
                        )[0]
                ]
            )
            both_type_node_emb_reduced = sign_net(
                spectral_features=spectral_features, edge_index=adj_reduced
            )
        else:
            both_type_node_emb_reduced = th.randn(
                node_type.size(0), self.emb_dim, device=self.device
            )
            
        both_type_node_emb = both_type_node_emb_reduced[all_node_map]
        node_emb = both_type_node_emb[expanded_node_type == 1]
        hyperedge_emb = both_type_node_emb[expanded_node_type == 0]
        
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
        ( node_pred, hyperedge_pred, augmented_incidence_pred,
          node_features_pred, hyperedge_features_pred ) = self.diffusion.sample(
            incidence_index=augmented_incidence_index,
            batch=batch,
            node_type=expanded_node_type,
            node_features=expanded_node_features,
            hyperedge_features=expanded_hyperedge_features,
            budget=expanded_budget,
            expansion_matrix_nodes=expansion_matrix[expanded_node_type == 1, :][:, node_type == 1],
            expansion_matrix_hyperedges=expansion_matrix[expanded_node_type == 0, :][:, node_type == 0],
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "hyperedge_emb": hyperedge_emb,
                "red_frac": 1 - size / expanded_size,
                "target_size": target_size.float(),
            },
        )
        
        # rescale
        budget_pred = (node_pred[:, 1] + 1)/2
        
        node_pred = (node_pred[:, 0] + 1)/2
        hyperedge_pred = hyperedge_pred + 1
        augmented_incidence_pred = (augmented_incidence_pred + 1)/2

        # get node attributes
        if self.deterministic_expansion:
            budget = split_budget(budget_pred, node_map, expanded_budget)
            
            # Don't expand nodes having no budget
            node_pred[budget == 1] = -th.inf
            
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
            node_attr[budget == 1] = 1
        else:
            node_attr = (node_pred > 0.5).long() + 1
            budget = th.maximum(budget_pred.round().long(), th.ones_like(budget_pred).long())
            
        hyperedge_attr = (hyperedge_pred > 0.66).long() + (hyperedge_pred > 1.33).long() + 1
        
        # construct new hypergraph
        adj = SparseTensor.from_edge_index(
            augmented_incidence_index[:, augmented_incidence_pred > 0.5],
            sparse_sizes=adj_augmented.sizes(),
            edge_attr = th.ones(th.sum(augmented_incidence_pred > 0.5), device=augmented_incidence_pred.device)
        )
        
        all_node_attr = th.zeros(batch.size(0), dtype=th.long, device = node_attr.device)
        all_node_attr[expanded_node_type == 0] = hyperedge_attr
        all_node_attr[expanded_node_type == 1] = node_attr
        
        return adj, batch, all_node_attr, expanded_node_type, node_features_pred, hyperedge_features_pred, budget

    def get_loss(self, batch, model: Module, sign_net: Module):
        """Returns a weighted sum of the node and edge expansion loss and the augmented edge loss."""
        # get augmented hypergraph
        adj_augmented = self.get_augmented_hypergraph(
            batch.adj_reduced, batch.expansion_matrix
        )

        # construct labels
        node_attr = (batch.node_expansion.float() - 1) * 2 - 1
        hyperedge_attr = batch.hyperedge_expansion.float() - 2
        augmented_incidence_index, incidence_val = to_edge_index(adj_augmented + batch.adj)
        augmented_incidence_attr = (incidence_val.float() - 1) * 2 - 1
        
        budget_expanded = (batch.expansion_matrix_nodes @ th.unsqueeze(batch.budget_reduced,-1)).squeeze(1).float()

        # get node embeddings
        if sign_net is not None:
            both_type_node_emb_reduced = sign_net(
                spectral_features=batch.spectral_features_reduced,
                edge_index=batch.adj_reduced,
            )
        else:
            both_type_node_emb_reduced = th.randn(
                batch.node_type.size(0), self.emb_dim, device=self.device
            )
                
        both_type_node_emb = batch.expansion_matrix @ both_type_node_emb_reduced
        node_emb = both_type_node_emb[batch.node_type == 1]
        hyperedge_emb = both_type_node_emb[batch.node_type == 0]
        
        # Expand features and perturbate them a bit to make the model more robust
        node_features_expanded = None
        if hasattr(batch, 'node_features'):
            node_features_expanded = noise_and_expand_features(
                batch.node_features_reduced,
                batch.expansion_matrix_nodes,
                self.node_features_noise_strength,
                self.node_features_on_simplex
            )
        
        hyperedge_features_expanded = None
        if hasattr(batch, 'hyperedge_features'):
            hyperedge_features_expanded = noise_and_expand_features(
                batch.hyperedge_features_reduced,
                batch.expansion_matrix_hyperedges,
                self.hyperedge_features_noise_strength,
                self.hyperedge_features_on_simplex
            )
            
        # reduction fraction
        size = scatter(batch.node_type, batch.batch)
        expanded_size = scatter(batch.node_expansion, batch.batch[batch.node_type == 1])
        red_frac = 1 - size / expanded_size
        
        # loss
        ( node_loss, hyperedge_loss, incidence_loss,
         node_features_loss, hyperedge_features_loss ) = self.diffusion.get_loss(
            incidence_index=augmented_incidence_index,
            batch=batch.batch,
            node_type=batch.node_type,
            expansion_matrix_nodes=batch.expansion_matrix_nodes,
            expansion_matrix_hyperedges=batch.expansion_matrix_hyperedges,
            node_attr=node_attr,
            hyperedge_attr=hyperedge_attr,
            incidence_attr=augmented_incidence_attr,
            node_features_expanded=node_features_expanded,
            node_features_real=batch.node_features if node_features_expanded is not None else None,
            budget_expanded=budget_expanded,
            budget_real=batch.budget,
            hyperedge_features_expanded=hyperedge_features_expanded,
            hyperedge_features_real=batch.hyperedge_features if hyperedge_features_expanded is not None else None,
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "hyperedge_emb": hyperedge_emb,
                "red_frac": red_frac,
                "target_size": batch.target_size.float(),
            },
        )

        loss = node_loss + hyperedge_loss + incidence_loss
        
        res = {
            "node_expansion_loss": node_loss.item(),
            "hyperedge_expansion_loss": hyperedge_loss.item(),
            "augmented_incidence_loss": incidence_loss.item(),
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
        """Returns the expanded bipartite adjacency matrix with additional augmented incidences.

        All incidence weights are set to 1.
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
        
        # drop out incidences
        if self.augmented_radius > 1 and self.augmented_dropout < 1:
            if self.augmented_dropout > 0.0:
                row, col, val = adj_augmented.coo()
                incidence_mask = th.rand_like(val) >= self.augmented_dropout
                incidence_mask = incidence_mask | (val > 1)  # keep required incidences
                
                # make undirected
                incidence_mask = incidence_mask & (row < col)
                incidence_index = th.stack([row[incidence_mask], col[incidence_mask]], dim=0)
                incidence_index = th.cat([incidence_index, incidence_index.flip(0)], dim=1)
                adj_augmented = SparseTensor.from_edge_index(
                    incidence_index,
                    edge_attr=th.ones(incidence_index.shape[1], device=self.device),
                    sparse_sizes=adj_augmented.sizes(),
                )
            else:
                adj_augmented.set_value(
                    th.ones(adj_augmented.nnz(), device=self.device), layout="coo"
                )
                
        return adj_augmented
  

def noise_and_expand_features(features_reduced, expansion_matrix, noise_strength, on_simplex):
    """Apply noise to features and expand them via the expansion matrix."""
    if noise_strength > 0:
        if on_simplex:
            summed = features_reduced.sum(dim=-1)
            # Perturbate only if not at the first scale (with one single node)
            if (summed > 1e-1).any():
                gamma = 10.0 / noise_strength
                alpha = features_reduced[summed > 1e-1] * gamma + 1
                dirichlet_dist = th.distributions.Dirichlet(alpha)
                features_reduced[summed > 1e-1] = dirichlet_dist.sample()
        else:
            features_reduced = features_reduced + noise_strength * th.randn_like(features_reduced)

    # Reshape features to 2D for sparse matrix multiplication
    original_shape = features_reduced.size()
    reshaped_features = features_reduced.reshape((original_shape[0], -1))  # (n, d1*d2*...)

    features = expansion_matrix @ reshaped_features

    # Reshape back to the original feature dimensions
    features = features.reshape((expansion_matrix.size(0),) + original_shape[1:])  # (k, d1, d2, ...)

    return features
    
def split_budget(budget_pred, node_map, expanded_budget):
    """Compute the budget allocated to each child node from its parent's budget."""
    device = budget_pred.device
    
    # Compute the predicted share of the parent's budget for each child
    clamped = th.clamp(budget_pred, min=0)
    budget_frac = clamped / scatter(clamped, node_map)[node_map]
    budget = th.floor(expanded_budget * budget_frac).long()

    # Budgets can't be smaller than 1 or equal to the parent budget if expanded
    budget = th.clamp(budget, min=1)

    node_counts = th.bincount(node_map)
    is_expanded = node_counts[node_map] > 1

    budget_max = th.where(
        is_expanded,
        expanded_budget - node_counts[node_map] + 1,
        expanded_budget
    )
    budget = th.minimum(budget, budget_max)

    # For the remainder
    remainder = expanded_budget - scatter(budget, node_map)[node_map]
    bx = expanded_budget * budget_frac
    frac = bx - th.floor(bx)

    # Compute rank inside cluster
    perm = (-frac + node_map * 2).argsort()
    inv_perm = th.empty_like(perm)
    inv_perm[perm] = th.arange(len(perm), device=device)

    cluster_sizes = scatter(th.ones_like(node_map), node_map)
    cluster_sizes_shifted = th.cat([th.tensor([0], device=device), cluster_sizes[:-1]])
    cluster_cumsum = th.cumsum(cluster_sizes_shifted, dim=0)

    indices = th.cumsum(th.ones_like(node_map), dim=0) - cluster_cumsum[node_map] - 1
    rank = indices[inv_perm]

    # Assign remainder
    extra = (rank < remainder).long()
    budget = budget + extra

    return budget


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
        node_features_unbatched = [
            node_features[node_features_start_idx[i] : node_features_end_idx[i]]
            for i in range(len(n_nodes))
        ]
    
    if hyperedge_features is not None:
        n_hyperedges = [
            th.sum(1 - node_type[hypergraph_start_idx[i] : hypergraph_end_idx[i]]).item()
            for i in range(len(size))
        ]
        
        hyperedge_features_end_idx = np.cumsum(n_hyperedges)
        hyperedge_features_start_idx = hyperedge_features_end_idx - n_hyperedges
        hyperedge_features_unbatched = [
            hyperedge_features[hyperedge_features_start_idx[i] : hyperedge_features_end_idx[i]]
            for i in range(len(n_hyperedges))
        ]
    
    return [ adj[hypergraph_start_idx[i] : hypergraph_end_idx[i], :][:, hypergraph_start_idx[i] : hypergraph_end_idx[i] ]
    for i in range(len(size)) ], n_nodes, node_features_unbatched, hyperedge_features_unbatched