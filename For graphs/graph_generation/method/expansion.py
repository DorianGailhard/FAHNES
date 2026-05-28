import networkx as nx
import torch as th
from torch.nn import Module
from torch_geometric.utils import to_edge_index
from torch_scatter import scatter
from torch_sparse import SparseTensor

from .method import Method

class Expansion(Method):
    """graph generation method generating graphs by local expansion."""  

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
        node_features_noise_strength=0,
        node_features_on_simplex=False,
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
        self.node_features_noise_strength = node_features_noise_strength
        self.node_features_on_simplex = node_features_on_simplex

    def sample_graphs(self, target_size, model: Module, sign_net: Module, node_features_dim: tuple):
        """Samples a batch of graphs."""
        num_graphs = len(target_size)
        adj = SparseTensor.from_dense(
            th.zeros((num_graphs, num_graphs), device=self.device)
        )

        batch = th.arange(num_graphs, device=self.device)
        node_expansion = th.ones(num_graphs, dtype=th.long, device=self.device)
        budget = target_size
        
        node_features = None
        if node_features_dim:
            shape = node_features_dim if isinstance(node_features_dim, tuple) else (node_features_dim,)
            node_features = th.zeros((num_graphs,) + shape, device=self.device, dtype=th.float32)
            
        while adj.size(0) < target_size.sum():
            adj, batch, node_expansion, node_features, budget = self.expand(
                adj,
                batch,
                node_expansion,
                node_features,
                budget,
                target_size,
                model=model,
                sign_net=sign_net,
            )
            
            if node_expansion.max() <= 1:
                break
                
        # return graphs            
        adjs, node_feats = unbatch_adj(adj, batch, node_features)
        graphs = []
        
        for i in range(len(adjs)):
            G = nx.from_scipy_sparse_array(adjs[i].to_scipy(layout="coo").astype(bool))
                
            if node_feats is not None:
                node_features = node_feats[i].cpu().numpy()
                G.add_nodes_from([
                    (node, {"feature": node_features[i]})
                    for (i, node) in enumerate(G.nodes)
                ])
            
            graphs.append(G)
            
        return graphs

    @th.no_grad()
    def expand(
        self,
        adj_reduced,
        batch_reduced,
        node_expansion,
        node_features,
        budget,
        target_size,
        model: Module,
        sign_net: Module,
    ):
        """Expands a graph by a single level."""
        # expand
        node_map = th.repeat_interleave(
            th.arange(0, adj_reduced.size(0), device=self.device), node_expansion
        )
        
        expanded_budget = budget[node_map]
        batch = batch_reduced[node_map]
        
        expanded_node_features = None
        
        if node_features is not None:
            expanded_node_features = node_features[node_map]
        
        size = scatter(th.ones_like(batch), batch)
        expansion_matrix = SparseTensor(
            row=th.arange(node_map.size(0), device=self.device),
            col=node_map,
            value=th.ones(node_map.size(0), device=self.device),
        )
        adj_augmented = self.get_augmented_graph(adj_reduced, expansion_matrix)
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
                    for adj in unbatch_adj(adj_reduced, batch_reduced, node_features)[0]
                ]
            )
            node_emb_reduced = sign_net(
                spectral_features=spectral_features, edge_index=adj_reduced
            )
        else:
            node_emb_reduced = th.randn(
                batch.size(0), self.emb_features, device=self.device
            )
            
        node_emb = node_emb_reduced[node_map]
        
        # compute number of nodes in expanded graph
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
        node_pred, augmented_edge_pred, node_features_pred = self.diffusion.sample(
            edge_index=augmented_edge_index,
            batch=batch,
            node_features=expanded_node_features,
            budget=expanded_budget,
            expansion_matrix=expansion_matrix,
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "red_frac": 1 - size / expanded_size,
                "target_size": target_size.float(),
            },
        )
        
        # rescale
        budget_pred = (node_pred[:, 1] + 1)/2
        
        node_pred = (node_pred[:, 0] + 1)/2
        augmented_edge_pred = (augmented_edge_pred + 1)/2

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
        
        # construct new graph
        adj = SparseTensor.from_edge_index(
            augmented_edge_index[:, augmented_edge_pred > 0.5],
            sparse_sizes=adj_augmented.sizes(),
            edge_attr = th.ones(th.sum(augmented_edge_pred > 0.5), device=augmented_edge_pred.device)
        )
        
        return adj, batch, node_attr, node_features_pred, budget

    def get_loss(self, batch, model: Module, sign_net: Module):
        """Returns a weighted sum of the node and edge expansion loss and the augmented edge loss."""
        # get augmented graph
        adj_augmented = self.get_augmented_graph(
            batch.adj_reduced, batch.expansion_matrix
        )

        # construct labels
        node_attr = (batch.node_expansion.float() - 1) * 2 - 1
        augmented_edge_index, edge_val = to_edge_index(adj_augmented + batch.adj)
        augmented_edge_attr = (edge_val.float() - 1) * 2 - 1
        
        budget_expanded = (batch.expansion_matrix @ th.unsqueeze(batch.budget_reduced,-1)).squeeze(1).float()

        # get node embeddings
        if sign_net is not None:
            node_emb_reduced = sign_net(
                spectral_features=batch.spectral_features_reduced,
                edge_index=batch.adj_reduced,
            )
        else:
            node_emb_reduced = th.randn(
                batch.batch.size(0), self.emb_features, device=self.device
            )
                
        node_emb = batch.expansion_matrix @ node_emb_reduced
        
        # Expand features and perturbate them a bit to make the model more robust
        node_features_expanded = None
        if hasattr(batch, 'node_features'):
            node_features_expanded = noise_and_expand_features(
                batch.node_features_reduced,
                batch.expansion_matrix,
                self.node_features_noise_strength,
                self.node_features_on_simplex
            )
            
        # reduction fraction
        size = scatter(th.ones_like(batch.batch), batch.batch)
        expanded_size = scatter(batch.node_expansion, batch.batch)
        red_frac = 1 - size / expanded_size
        
        # loss
        node_loss, edge_loss, node_features_loss = self.diffusion.get_loss(
            edge_index=augmented_edge_index,
            batch=batch.batch,
            expansion_matrix=batch.expansion_matrix,
            node_attr=node_attr,
            edge_attr=augmented_edge_attr,
            node_features_expanded=node_features_expanded,
            node_features_real=batch.node_features if node_features_expanded is not None else None,
            budget_expanded=budget_expanded,
            budget_real=batch.budget,
            model=model,
            model_kwargs={
                "node_emb": node_emb,
                "red_frac": red_frac,
                "target_size": batch.target_size.float(),
            },
        )

        loss = node_loss + edge_loss
        
        res = {
            "node_expansion_loss": node_loss.item(),
            "augmented_edge_loss": edge_loss.item(),
        }
        
        if hasattr(batch, 'node_features'):
            node_features_loss = node_features_loss.mean()
            loss += node_features_loss
            res["node_features_loss"] = node_features_loss.item()
            
        res["loss"] = loss.item()
            
        return loss, res

    def get_augmented_graph(self, adj_reduced, expansion_matrix):
        """Returns the expanded adjacency matrix with additional augmented edges.

        All edge weights are set to 1.
        """
        # construct augmented adjacency matrix
        adj_reduced = adj_reduced.set_diag(1)
        adj_reduced_augmented = adj_reduced.copy()
        for _ in range(1, self.augmented_radius):
            adj_reduced_augmented = adj_reduced_augmented @ adj_reduced

        adj_reduced_augmented = adj_reduced_augmented.set_value(
            th.ones(adj_reduced_augmented.nnz(), device=self.device), layout="coo"
        )
        adj_augmented = (
            expansion_matrix @ adj_reduced_augmented @ expansion_matrix.t()
        ).remove_diag()
        adj_expanded = (
            expansion_matrix @ adj_reduced @ expansion_matrix.t()
        ).remove_diag()

        # drop out edges
        if self.augmented_dropout > 0.0:
            adj_required = adj_augmented + adj_expanded
            row, col, val = adj_required.coo()
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

def unbatch_adj(adj, batch, node_features) -> tuple:
    size = scatter(th.ones_like(batch), batch)
    
    graph_end_idx = size.cumsum(0)
    graph_start_idx = graph_end_idx - size
    
    node_features_unbatched = None
    if node_features is not None:
        node_features_unbatched = [
            node_features[graph_start_idx[i] : graph_end_idx[i]]
            for i in range(len(size))
        ]
    
    return [ adj[graph_start_idx[i] : graph_end_idx[i], :][:, graph_start_idx[i] : graph_end_idx[i] ]
    for i in range(len(size)) ], node_features_unbatched