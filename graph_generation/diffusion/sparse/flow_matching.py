import torch as th
from torch.nn import Module
from torch_geometric.utils import sort_edge_index
from torch_scatter import scatter
from ot import emd
import numpy as np

class CFMModel(Module):
    """Model for Conditional Flow Matching without preconditioning."""
    
    def __init__(self, self_conditioning):
        super().__init__()
        self.self_conditioning = self_conditioning

    def forward(
        self,
        edge_index,
        batch,
        node_type,
        node_cluster_size,
        node_attr,
        edge_node_attr,
        edge_attr,
        node_features_initial,
        node_features,
        hyperedge_features,
        t,
        model,
        model_kwargs,
        node_attr_self_cond=None,
        edge_node_attr_self_cond=None,
        edge_attr_self_cond=None,
    ):
        model_kwargs = dict(model_kwargs, noise_cond=t.float())
        
        node_attr_in = node_attr
        edge_node_attr_in = edge_node_attr
        edge_attr_in = edge_attr
        
        if node_features is not None:
            node_features_in = node_features
            
        if hyperedge_features is not None:
            hyperedge_features_in = hyperedge_features

        # Self-conditioning (if enabled)
        if self.self_conditioning:
            if model.training and np.random.rand() < 0.5:
                # Self-conditioning pass
                with th.no_grad():
                    node_attr_self_cond, edge_node_attr_self_cond, edge_attr_self_cond, node_features_self_cond, hyperedge_features_self_cond = model(
                        edge_index=edge_index,
                        batch=batch,
                        node_type=node_type,
                        node_cluster_size=node_cluster_size,
                        node_attr=th.cat(
                            [node_attr_in, th.zeros_like(node_attr_in)], dim=-1
                        ).float(),
                        edge_node_attr=th.cat(
                            [edge_node_attr_in, th.zeros_like(edge_node_attr_in)], dim=-1
                        ).float(),
                        edge_attr=th.cat(
                            [edge_attr_in, th.zeros_like(edge_attr_in)], dim=-1
                        ).float(),
                        node_features_initial=node_features_initial,
                        node_features=th.cat(
                            [node_features_in, th.zeros_like(node_features_in)], dim=-1
                        ).float() if node_features is not None else None,
                        hyperedge_features=th.cat(
                            [hyperedge_features_in, th.zeros_like(hyperedge_features_in)], dim=-1
                        ).float() if hyperedge_features is not None else None,
                        **model_kwargs,
                    )
                    node_attr_self_cond = node_attr_self_cond.detach()
                    edge_node_attr_self_cond = edge_node_attr_self_cond.detach()
                    edge_attr_self_cond = edge_attr_self_cond.detach()
                    
                    if node_features is not None:
                        node_features_self_cond = node_features_self_cond.detach()
                        
                    if hyperedge_features is not None:
                        hyperedge_features_self_cond = hyperedge_features_self_cond.detach()
            else:
                node_attr_self_cond = th.zeros_like(node_attr)
                edge_node_attr_self_cond = th.zeros_like(edge_node_attr)
                edge_attr_self_cond = th.zeros_like(edge_attr)
                
                if node_features is not None:
                    node_features_self_cond = th.zeros_like(node_features)
                    
                if hyperedge_features is not None:
                    hyperedge_features_self_cond = th.zeros_like(hyperedge_features)

            # Concatenate with input
            node_attr_in = th.cat([node_attr_in, node_attr_self_cond], dim=-1)
            edge_node_attr_in = th.cat([edge_node_attr_in, edge_node_attr_self_cond], dim=-1)
            edge_attr_in = th.cat([edge_attr_in, edge_attr_self_cond], dim=-1)
            
            if node_features is not None:
                node_features_in = th.cat([node_features_in, node_features_self_cond], dim=-1)
                
            if hyperedge_features is not None:
                hyperedge_features_in = th.cat([hyperedge_features_in, hyperedge_features_self_cond], dim=-1)

        # Compute model output
        node_attr_pred, edge_node_attr_pred, edge_attr_pred, node_features_pred, hyperedge_features_pred = model(
            edge_index=edge_index,
            batch=batch,
            node_type=node_type,
            node_cluster_size=node_cluster_size,
            node_attr=node_attr_in.float(),
            edge_node_attr=edge_node_attr_in.float(),
            edge_attr=edge_attr_in.float(),
            node_features_initial=node_features_initial,
            node_features=node_features_in.float() if node_features is not None else None,
            hyperedge_features=hyperedge_features_in.float() if hyperedge_features is not None else None,
            **model_kwargs,
        )
        
        return node_attr_pred, edge_node_attr_pred, edge_attr_pred, node_features_pred, hyperedge_features_pred


class CFM:
    """Conditional Flow Matching Framework"""
    alpha = 1.5
    
    def __init__(self, self_conditioning, num_steps):
        self.model_wrapper = CFMModel(self_conditioning)
        self.num_steps = num_steps
    
    @property
    def device(self):
        assert hasattr(self, "_device")
        return self._device

    def to(self, device):
        self._device = device
        self.model_wrapper.to(device)
        return self
    
    def get_loss(self, edge_index, batch, node_type, expansion_matrix_nodes, expansion_matrix_hyperedges, node_attr, edge_node_attr, edge_attr, node_features_expanded, node_features_real, node_cluster_size_expanded, node_cluster_size_real, hyperedge_features_expanded, hyperedge_features_real, model, model_kwargs):
        """Loss function for conditional flow matching."""
        # sample noise level
        num_graphs = batch.max().item() + 1
        t = th.randn((num_graphs,), device=self.device) # sample gaussian
        t = th.sigmoid(t) # map through logistic function
        
        # Compute clusters
        row, col, _ = expansion_matrix_nodes.coo()
        clusters_nodes = col[th.argsort(row)]
        
        unique_clusters, cluster_counts = th.unique(clusters_nodes, return_counts=True)
        expanded_clusters = unique_clusters[cluster_counts != 1]
        nodes_expanded = row[th.isin(col, expanded_clusters)]
        nodes_for_budget_loss = nodes_expanded[node_cluster_size_expanded[nodes_expanded] > 2]
        
        # Compute start and end points
        edge_batch = batch[edge_index[0]]
        
        # compute start and end points for node attributes
        ratio_node_cluster_size = (node_cluster_size_real/node_cluster_size_expanded)*2 - 1 # Rescale between -1 and 1
        node_end = th.stack((node_attr, ratio_node_cluster_size), dim=1)
        
        node_attr_start = th.randn_like(node_attr)
        
        # Force initial point for node budgets to be on the simplex (transformed into [-1, 1])
        node_budget_start = th.distributions.Gamma(self.alpha*th.ones_like(ratio_node_cluster_size), th.ones_like(ratio_node_cluster_size)).sample().clamp(min=1e-10)
        node_budget_start = (node_budget_start / scatter(node_budget_start, clusters_nodes)[clusters_nodes])*2 - 1
        
        node_start = th.stack((node_attr_start, node_budget_start), dim=1)
        
        
        edge_node_start = th.randn_like(edge_node_attr)
        
        edge_start = self.edge_randn(edge_index, dtype=th.float32)
            
        node_features_start = None
        node_features_end = None
        if node_features_expanded is not None:
            # Compute target
            node_features_end = node_features_real - node_features_expanded
            
            # Compute starting point
            node_features_start = th.randn_like(node_features_expanded)
            
            # Expand t to allow multipliying
            n_dims = node_features_end.dim()
            t_expanded_node_features = t[batch].view(-1, *([1] * (n_dims - 1)))
            
        hyperedge_features_start = None
        hyperedge_features_end = None
        if hyperedge_features_expanded is not None:
            # Compute target
            hyperedge_features_end = hyperedge_features_real
            
            # Compute starting point
            hyperedge_features_start = hyperedge_features_expanded
            
            # Expand t to allow multipliying
            n_dims = hyperedge_features_end.dim()
            t_expanded_hyperedge_features = t[batch].view(-1, *([1] * (n_dims - 1)))
        
        
        # Minibatch OT-coupling
        # In the following we only consider permuting duplicates of nodes before refinement
        # Nodes
        perm = self.minibatch_ot_coupling(expansion_matrix_nodes, node_start, node_end, node_features_start, node_features_end, device=self.device)
        node_start = node_start[perm]
        
        if node_features_expanded is not None:
            node_features_start = node_features_start[perm]
            
        # Edge nodes            
        perm = self.minibatch_ot_coupling(expansion_matrix_hyperedges, edge_node_start, edge_node_attr, hyperedge_features_start, hyperedge_features_end, device=self.device)
        
        edge_node_start = edge_node_start[perm]

        if hyperedge_features_expanded is not None:
            hyperedge_features_start = hyperedge_features_start[perm]
            
        # Forward model to predict flow
        node_attr_pred, edge_node_attr_pred, edge_attr_pred, node_features_pred, hyperedge_features_pred = self.model_wrapper(
            edge_index=edge_index,
            batch=batch,
            node_type=node_type,
            node_cluster_size=node_cluster_size_expanded,
            node_attr=t[batch[node_type==1]].unsqueeze(1) * node_end + (1-t[batch[node_type==1]].unsqueeze(1)) * node_start,
            edge_node_attr=(t[batch[node_type==0]] * edge_node_attr + (1-t[batch[node_type==0]]) * edge_node_start).unsqueeze(1),
            edge_attr=(t[edge_batch] * edge_attr + (1-t[edge_batch]) * edge_start).unsqueeze(1),
            node_features_initial=node_features_expanded,
            node_features=t_expanded_node_features[node_type == 1] * node_features_end + (1-t_expanded_node_features[node_type == 1]) * node_features_start if node_features_expanded is not None else None,
            hyperedge_features=t_expanded_hyperedge_features[node_type == 0] * hyperedge_features_end + (1-t_expanded_hyperedge_features[node_type == 0]) * hyperedge_features_start if hyperedge_features_expanded is not None else None,
            t=t,
            model=model,
            model_kwargs=model_kwargs,
        )

        # Compute loss
        node_attr_pred = node_attr_pred.float()
        edge_node_attr_pred = edge_node_attr_pred.float().squeeze(1)
        edge_attr_pred = edge_attr_pred.float().squeeze(1)
        
        node_loss = (node_attr_pred - node_end) ** 2
        edge_node_loss = (edge_node_attr_pred - edge_node_attr)**2
        edge_loss = (edge_attr_pred - edge_attr) ** 2
        
        node_features_loss = None
        if node_features_expanded is not None:
            node_features_loss = (node_features_pred - node_features_end)**2
            
        hyperedge_features_loss = None
        if hyperedge_features_expanded is not None:
            hyperedge_features_loss = (hyperedge_features_pred - hyperedge_features_end)**2
        
        node_loss = ( scatter(node_loss[node_cluster_size_expanded != 1, 0], batch[node_type == 1][node_cluster_size_expanded != 1], reduce="mean").mean()
                     + scatter(node_loss[nodes_for_budget_loss, 1], batch[node_type == 1][nodes_for_budget_loss], reduce="mean").mean()
                    )
                    # No need to predict the budget clusters that are not expanded, nor when the budget is equal to two (then necessarily half and a half
                    # between the children), nor is it necessary to predict the expansion for nodes that cannot be expanded
        
        edge_node_loss = scatter(edge_node_loss, batch[node_type == 0], reduce="mean").mean()
        edge_loss = scatter(edge_loss, batch[edge_index[0]], reduce="mean").mean()
        
        if node_features_expanded is not None:
            node_features_loss = scatter(node_features_loss[nodes_expanded].sum(dim=-1), batch[node_type == 1][nodes_expanded], reduce="mean").mean()
            # The model only learns to move features of expanded nodes
            
        if hyperedge_features_expanded is not None:
            hyperedge_features_loss = scatter(hyperedge_features_loss.mean(dim=-1), batch[node_type == 0], reduce="mean").mean()            
           
        return node_loss, edge_node_loss, edge_loss, node_features_loss, hyperedge_features_loss


    @th.no_grad()
    def sample(self, edge_index, node_type, node_features, hyperedge_features, node_cluster_size, expansion_matrix_nodes, batch, model, model_kwargs):
        """Sampling procedure for CFM."""
        num_graphs = batch.max().item() + 1
        
        row, col, _ = expansion_matrix_nodes.coo()
        clusters_nodes = col[th.argsort(row)]

        unique_clusters, cluster_counts = th.unique(clusters_nodes, return_counts=True)
        expanded_clusters = unique_clusters[cluster_counts != 1]
        nodes_not_expanded = row[~th.isin(col, expanded_clusters)]
        nodes_expanded = row[th.isin(col, expanded_clusters)]
        nodes_expanded_cluster_size_two = nodes_expanded[node_cluster_size[nodes_expanded] == 2]

        # Sample random initialization
        node_attr_start = th.randn(batch[node_type == 1].shape[0], dtype=th.float32, device=self.device)
        
        # Force initial point for node budgets to be on the simplex (transformed into [-1, 1])
        node_budget_start = th.distributions.Gamma(self.alpha*th.ones(batch[node_type == 1].shape[0], dtype=th.float32, device=self.device), th.ones(batch[node_type == 1].shape[0], dtype=th.float32, device=self.device)).sample().clamp(min=1e-10)
        node_budget_start = (node_budget_start / scatter(node_budget_start, clusters_nodes)[clusters_nodes])*2 - 1

        node_attr_next = th.stack((node_attr_start, node_budget_start), dim=1)
        
        edge_node_attr_next = th.randn(batch[node_type == 0].shape[0], dtype=th.float32, device=self.device)[:, None]
        edge_attr_next = self.edge_randn(edge_index, dtype=th.float32)[:, None]
        
        node_features_next = None
        if node_features is not None:
            node_features_next = th.randn_like(node_features)
            
        hyperedge_features_next = None
        if hyperedge_features is not None:
            hyperedge_features_next = hyperedge_features
            
        # For self-conditioning
        k1_node_attr = th.zeros_like(node_attr_next)
        k1_edge_node_attr = th.zeros_like(edge_node_attr_next)
        k1_edge_attr = th.zeros_like(edge_attr_next) 
        
        # Flow matching sampling: Integrate learned flow model over time (0 to 1) with Heun's method
        t_steps = th.linspace(0, 1, steps=self.num_steps, device=self.device)
        dt = t_steps[1] - t_steps[0]
        
        for i in range(len(t_steps[:-1])):            
            # First step: Calculate initial derivatives (k1)
            k1_node_attr, k1_edge_node_attr, k1_edge_attr, k1_node_features, k1_hyperedge_features = self.model_wrapper(
                edge_index=edge_index,
                batch=batch,
                node_type=node_type,
                node_cluster_size=node_cluster_size,
                node_attr=node_attr_next,
                edge_node_attr=edge_node_attr_next,
                edge_attr=edge_attr_next,
                node_features_initial=node_features,
                node_features=node_features_next,
                hyperedge_features=hyperedge_features_next,
                t=t_steps[i].repeat(num_graphs),
                model=model,
                model_kwargs=model_kwargs,
                node_attr_self_cond=k1_node_attr,
                edge_node_attr_self_cond=k1_edge_node_attr,
                edge_attr_self_cond=k1_edge_attr,
            )
            
            # Project budgets on the simplex (renormalized to [-1, 1]) using softmax
            k1_node_attr[:, 1] = th.clamp(k1_node_attr[:, 1] + 1, min=1e-10)/2
            k1_node_attr[:, 1] = (k1_node_attr[:, 1] / scatter(k1_node_attr[:, 1], clusters_nodes, reduce="sum")[clusters_nodes]) * 2 - 1
            k1_node_attr[node_cluster_size == 1, 0] = -1 # Nodes without budget cannot be further expanded
            k1_node_attr[nodes_expanded_cluster_size_two, 1] = 0 # When the cluster is expanded and its budget is 2, the fraction is necessarily half and half
            
            # Heun's method update: Average of k1 and k2
            node_attr_next = node_attr_next + dt * (k1_node_attr - node_attr_next)/(1-t_steps[i])
            edge_node_attr_next = edge_node_attr_next + dt * (k1_edge_node_attr - edge_node_attr_next)/(1-t_steps[i])
            edge_attr_next = edge_attr_next + dt * (k1_edge_attr - edge_attr_next)/(1-t_steps[i])
            
            if node_features is not None:
                k1_node_features[nodes_not_expanded] = 0 # Only expanded nodes move their features
                node_features_next = node_features_next + dt * (k1_node_features - node_features_next)/(1-t_steps[i])
                
            if hyperedge_features is not None:
                hyperedge_features_next = hyperedge_features_next + dt * 0.5 * (k1_hyperedge_features - hyperedge_features_next)/(1-t_steps[i])
        
        return node_attr_next, edge_node_attr_next.squeeze(1), edge_attr_next.squeeze(1), node_features + node_features_next if node_features_next is not None else None, hyperedge_features_next if hyperedge_features_next is not None else None


    @staticmethod
    def edge_randn(edge_index, dtype=th.float32) -> th.Tensor:
        """Sample symmetric Gaussian noise for edges in edge_index."""
        # sample noise for upper triangle
        edge_index_u = edge_index[:, edge_index[0] <= edge_index[1]]
        edge_noise_u = th.randn_like(edge_index_u[0], dtype=dtype)
    
        # make symmetric
        new_edge_index, edge_noise = sort_edge_index(
            edge_index=th.cat([edge_index_u, edge_index_u.flip(0)], dim=1),
            edge_attr=th.cat([edge_noise_u, edge_noise_u], dim=0),
        )
        assert (edge_index == new_edge_index).all()
    
        return edge_noise
    
    @staticmethod
    def minibatch_ot_coupling(expansion_matrix, attr_start, attr_end, features_start=None, features_end=None, device: str = 'cuda'):
        """
        Perform minibatch optimal transport coupling for nodes. Only pairs of nodes coming from an expanded cluster can be swapped.
    
        Parameters:
        - expansion_matrix: The expansion matrix.
        - attr_start: The starting noise for the attributes.
        - attr_end: The ending points for the attributes.
        - features_start: Optional, the starting features.
        - features_end: Optional, the ending features.
    
        Returns:
        - perm: The final permutation for OT-coupling.
        """
        # Code is ugly due to heavy broadcasting but in fact this is simple: two nodes from the same expanded cluster are equivalent except
        # for their starting gaussian noise, so we permutate the noise to minimize the distance between starting noise and ending point.
        # Same for hyperedges but with clusters of size 2 and 3.
        # This is similar to minibatch OT coupling for images.
        
        # Find which cluster contains which node
        row, col, _ = expansion_matrix.coo()
        clusters_nodes = col[th.argsort(row)]
    
        # Identify expanded clusters
        unique_clusters, cluster_counts = th.unique(clusters_nodes, return_counts=True)
        perm = th.arange(attr_start.size(0), dtype=th.int64, device=device)
        
        
        # For clusters of two nodes
        two_node_clusters = unique_clusters[cluster_counts == 2]
        
        # Filter nodes that belong to clusters with two nodes
        mask = th.isin(clusters_nodes, two_node_clusters)
        relevant_nodes = th.nonzero(mask).squeeze()
    
        if len(relevant_nodes) > 0:
            # Extract clusters of two nodes (clusters of just one node will not be reordered, we can skip computations)
            relevant_clusters = clusters_nodes[relevant_nodes]
            
            # Sort the relevant nodes by their cluster IDs
            sorted_indices = th.arange(len(relevant_nodes), dtype=th.int64, device=device)[th.argsort(relevant_clusters)]
    
            # Create a permutation where nodes in the same cluster swap their starting point
            perm_nodes_relevant = th.arange(len(relevant_nodes), dtype=th.int64, device=device)
            perm_nodes_relevant[sorted_indices[::2]], perm_nodes_relevant[sorted_indices[1::2]] = perm_nodes_relevant[sorted_indices[1::2]], perm_nodes_relevant[sorted_indices[::2]]
            
            # Compute the distance between starting noise and ending point for the relevant nodes
            perm_nodes_relevant = th.stack([th.arange(len(relevant_nodes), dtype=th.int64, device=device), perm_nodes_relevant], dim=0)
            perm_nodes_expanded = perm_nodes_relevant.view(*perm_nodes_relevant.shape, *([1] * (len(attr_start.shape) - len(perm_nodes_relevant.shape) + 1)))
            perm_nodes_expanded = perm_nodes_expanded.expand(2, *attr_start[relevant_nodes].shape)
            attr_start_relevant = th.gather(attr_start[relevant_nodes].unsqueeze(0).expand(perm_nodes_expanded.size(0), *attr_start[relevant_nodes].shape), 1, perm_nodes_expanded)
    
            node_distance_relevant = (attr_end[relevant_nodes] - attr_start_relevant) ** 2
            
            # Sum the distances for the relevant clusters (summing distance across all dimensions so each node gets a scalar)
            scatter_indices = th.searchsorted(two_node_clusters, relevant_clusters) # Renumber cluster id so instead of [15, 15, 19, ...] they become [0, 0, 1, ...]
            expanded_scatter_indices = scatter_indices.unsqueeze(0).expand(node_distance_relevant.size(0), -1)  # Shape: (2, num_relevant_nodes)
    
            summed_node_distance_relevant = scatter(src=node_distance_relevant.sum(dim=tuple(range(2, node_distance_relevant.ndim))) if node_distance_relevant.ndim > 2 else node_distance_relevant, index=expanded_scatter_indices, dim=1, reduce="sum")
    
            if features_start is not None:
                perm_nodes_expanded = perm_nodes_relevant.view(*perm_nodes_relevant.shape, *([1] * (len(features_start.shape) - len(perm_nodes_relevant.shape) + 1)))
                perm_nodes_expanded = perm_nodes_expanded.expand(2, *features_start[relevant_nodes].shape)
    
                features_start_relevant = th.gather(features_start[relevant_nodes].unsqueeze(0).expand(perm_nodes_expanded.size(0), *features_start[relevant_nodes].shape), 1, perm_nodes_expanded)
    
                nodes_features_distance_relevant = (features_end[relevant_nodes] - features_start_relevant)**2
                summed_nodes_features_distance_relevant = scatter(src=nodes_features_distance_relevant.sum(dim=tuple(range(2, nodes_features_distance_relevant.ndim))) if nodes_features_distance_relevant.ndim > 2 else nodes_features_distance_relevant, index=expanded_scatter_indices, dim=1, reduce="sum")
    
                summed_node_distance_relevant += summed_nodes_features_distance_relevant
                
            # Select best ordering for the relevant nodes
            best_perm_relevant = th.argmin(summed_node_distance_relevant, dim=0)
            final_perm_choice_relevant = best_perm_relevant[scatter_indices]
            perm[relevant_nodes] = relevant_nodes[perm_nodes_relevant[final_perm_choice_relevant, th.arange(len(final_perm_choice_relevant), device=device)]]

            # For clusters of three nodes
            three_node_clusters = unique_clusters[cluster_counts == 3]
        
            # Filter nodes that belong to clusters with two nodes
            mask = th.isin(clusters_nodes, three_node_clusters)
            relevant_nodes = th.nonzero(mask).squeeze()
        
            if len(relevant_nodes) > 0:
                # Extract clusters of two nodes (clusters of just one node will not be reordered, we can skip computations)
                relevant_clusters = clusters_nodes[relevant_nodes]
                
                # Sort the relevant nodes by their cluster IDs
                sorted_indices = th.arange(len(relevant_nodes), dtype=th.int64, device=device)[th.argsort(relevant_clusters)]
        
                # Create permutations for the relevant nodes
                perm_nodes_relevant1 = th.arange(len(relevant_nodes), dtype=th.int64, device=device)
                perm_nodes_relevant1[sorted_indices[::3]], perm_nodes_relevant1[sorted_indices[1::3]], perm_nodes_relevant1[sorted_indices[2::3]] = perm_nodes_relevant1[sorted_indices[::3]], perm_nodes_relevant1[sorted_indices[2::3]], perm_nodes_relevant1[sorted_indices[1::3]]
                
                perm_nodes_relevant2 = th.arange(len(relevant_nodes), dtype=th.int64, device=device)
                perm_nodes_relevant2[sorted_indices[::3]], perm_nodes_relevant2[sorted_indices[1::3]], perm_nodes_relevant2[sorted_indices[2::3]] = perm_nodes_relevant2[sorted_indices[1::3]], perm_nodes_relevant2[sorted_indices[::3]], perm_nodes_relevant2[sorted_indices[2::3]]
                
                perm_nodes_relevant3 = th.arange(len(relevant_nodes), dtype=th.int64, device=device)
                perm_nodes_relevant3[sorted_indices[::3]], perm_nodes_relevant3[sorted_indices[1::3]], perm_nodes_relevant3[sorted_indices[2::3]] = perm_nodes_relevant3[sorted_indices[1::3]], perm_nodes_relevant3[sorted_indices[2::3]], perm_nodes_relevant3[sorted_indices[::3]]
                
                perm_nodes_relevant4 = th.arange(len(relevant_nodes), dtype=th.int64, device=device)
                perm_nodes_relevant4[sorted_indices[::3]], perm_nodes_relevant4[sorted_indices[1::3]], perm_nodes_relevant4[sorted_indices[2::3]] = perm_nodes_relevant4[sorted_indices[2::3]], perm_nodes_relevant4[sorted_indices[::3]], perm_nodes_relevant4[sorted_indices[1::3]]
        
                perm_nodes_relevant5 = th.arange(len(relevant_nodes), dtype=th.int64, device=device)
                perm_nodes_relevant5[sorted_indices[::3]], perm_nodes_relevant5[sorted_indices[1::3]], perm_nodes_relevant5[sorted_indices[2::3]] = perm_nodes_relevant5[sorted_indices[2::3]], perm_nodes_relevant5[sorted_indices[1::3]], perm_nodes_relevant5[sorted_indices[::3]]
                
                # Compute the distance between starting noise and ending point for the relevant nodes
                perm_nodes_relevant = th.stack([th.arange(len(relevant_nodes), dtype=th.int64, device=device), perm_nodes_relevant1, perm_nodes_relevant2, perm_nodes_relevant3, perm_nodes_relevant4, perm_nodes_relevant5], dim=0)
                perm_nodes_expanded = perm_nodes_relevant.view(*perm_nodes_relevant.shape, *([1] * (len(attr_start.shape) - len(perm_nodes_relevant.shape) + 1)))
                perm_nodes_expanded = perm_nodes_expanded.expand(6, *attr_start[relevant_nodes].shape)
                attr_start_relevant = th.gather(attr_start[relevant_nodes].unsqueeze(0).expand(perm_nodes_expanded.size(0), *attr_start[relevant_nodes].shape), 1, perm_nodes_expanded)
        
                node_distance_relevant = (attr_end[relevant_nodes] - attr_start_relevant) ** 2
        
                # Sum the distances for the relevant clusters (summing distance across all dimensions so each node gets a scalar)
                scatter_indices = th.searchsorted(three_node_clusters, relevant_clusters) # Renumber cluster id so instead of [15, 15, 19, ...] they become [0, 0, 1, ...]
                expanded_scatter_indices = scatter_indices.unsqueeze(0).expand(node_distance_relevant.size(0), -1)  # Shape: (2, num_relevant_nodes)
        
                summed_node_distance_relevant = scatter(src=node_distance_relevant.sum(dim=tuple(range(2, node_distance_relevant.ndim))) if node_distance_relevant.ndim > 2 else node_distance_relevant, index=expanded_scatter_indices, dim=1, reduce="sum")
        
                if features_start is not None:
                    perm_nodes_expanded = perm_nodes_relevant.view(*perm_nodes_relevant.shape, *([1] * (len(features_start.shape) - len(perm_nodes_relevant.shape) + 1)))
                    perm_nodes_expanded = perm_nodes_expanded.expand(6, *features_start[relevant_nodes].shape)
        
                    features_start_relevant = th.gather(features_start[relevant_nodes].unsqueeze(0).expand(perm_nodes_expanded.size(0), *features_start[relevant_nodes].shape), 1, perm_nodes_expanded)
        
                    nodes_features_distance_relevant = (features_end[relevant_nodes] - features_start_relevant)**2
                    summed_nodes_features_distance_relevant = scatter(src=nodes_features_distance_relevant.sum(dim=tuple(range(2, nodes_features_distance_relevant.ndim))) if nodes_features_distance_relevant.ndim > 2 else nodes_features_distance_relevant, index=expanded_scatter_indices, dim=1, reduce="sum")
        
                    summed_node_distance_relevant += summed_nodes_features_distance_relevant
                    
                # Select best ordering for the relevant nodes
                best_perm_relevant = th.argmin(summed_node_distance_relevant, dim=0)
                final_perm_choice_relevant = best_perm_relevant[scatter_indices]
                perm[relevant_nodes] = relevant_nodes[perm_nodes_relevant[final_perm_choice_relevant, th.arange(len(final_perm_choice_relevant), device=device)]]

        return perm
        

    @staticmethod
    def compute_node_averages(
        node_type: th.Tensor,
        edge_index: th.Tensor,
        node_attributes: th.Tensor,
        edge_node_attributes: th.Tensor,
        device: str = 'cuda'
    ) -> th.Tensor:
        """
        Compute average edge node attributes based on connected nodes.

        Args:
            node_type: Tensor indicating type of each node (0 for hyperedges, 1 for regular nodes)
            edge_index: Tensor of shape [2, num_edges] containing source and destination node indices
            node_attributes: Tensor containing attributes for regular nodes
            edge_node_attributes: Tensor containing attributes for nodes representing hyperedges
            device: Device to place tensors on ('cuda' or 'cpu')

        Returns:
            th.Tensor: Average attributes for hyperedge nodes
        """
        # Determine the dimensionality of the attributes
        node_attr_dim = node_attributes.ndimension() if node_attributes is not None else -1
        edge_node_attr_dim = edge_node_attributes.ndimension() if edge_node_attributes is not None else -1

        # Initialize full cluster size tensor for all nodes
        if node_attributes is not None:
            if node_attr_dim == 1:
                node_attributes = node_attributes.unsqueeze(1)
            attributes_full = th.zeros((node_type.shape[0], node_attributes.shape[1]), device=device)
            attributes_full[node_type == 1] = node_attributes

        if edge_node_attributes is not None:
            if edge_node_attr_dim == 1:
                edge_node_attributes = edge_node_attributes.unsqueeze(1)
            attributes_full = th.zeros((node_type.shape[0], edge_node_attributes.shape[1]), device=device)
            attributes_full[node_type == 0] = edge_node_attributes

        # Get source and destination nodes from edge_index
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        # Initialize tensors for accumulating neighbor features and counts
        neighbor_sum = th.zeros_like(attributes_full)
        neighbor_count = th.zeros(node_type.shape[0], device=device)

        # Accumulate neighbor features
        neighbor_sum.index_add_(0, src_nodes, attributes_full[dst_nodes])
        neighbor_sum.index_add_(0, dst_nodes, attributes_full[src_nodes])

        # Count neighbors for each node
        neighbor_count.index_add_(0, src_nodes, th.ones_like(src_nodes, dtype=neighbor_count.dtype))
        neighbor_count.index_add_(0, dst_nodes, th.ones_like(dst_nodes, dtype=neighbor_count.dtype))

        # Prevent division by zero
        new_shape = neighbor_count.shape + (1,) * (neighbor_sum.ndim - neighbor_count.ndim)
        neighbor_count = neighbor_count.clamp(min=1).view(*new_shape)

        # Calculate average of neighbor features
        neighbor_avg = neighbor_sum / neighbor_count

        # Return averages for hyperedge nodes only (where node_type == 0)
        if node_attr_dim == 1 or edge_node_attr_dim == 1:
            neighbor_avg = neighbor_avg.squeeze(1)

        return neighbor_avg
