import torch as th
from torch.nn import Module
from torch_geometric.utils import sort_edge_index
from torch_scatter import scatter
import numpy as np

class CFMModel(Module):
    """Model for Conditional Flow Matching without preconditioning."""
    
    def __init__(self, self_conditioning, node_features_on_simplex, hyperedge_features_on_simplex):
        super().__init__()
        self.self_conditioning = self_conditioning
        self.node_features_on_simplex = node_features_on_simplex
        self.hyperedge_features_on_simplex = hyperedge_features_on_simplex

    def forward(
        self,
        incidence_index,
        batch,
        node_type,
        node_cluster_size,
        node_attr,
        hyperedge_attr,
        incidence_attr,
        node_features_initial,
        node_features,
        hyperedge_features_initial,
        hyperedge_features,
        t,
        model,
        model_kwargs,
        clusters_nodes=None,
        nodes_expanded_cluster_size_two=None,
        nodes_expanded=None,
        nodes_not_expanded=None,
        hyperedges_not_expanded=None,
        node_attr_self_cond=None,
        hyperedge_attr_self_cond=None,
        incidence_attr_self_cond=None,
        node_features_self_cond=None,
        hyperedge_features_self_cond=None
    ):
        model_kwargs = dict(model_kwargs, noise_cond=t.float())
        
        node_attr_in = node_attr
        hyperedge_attr_in = hyperedge_attr
        incidence_attr_in = incidence_attr
        
        if node_features is not None:
            node_features_in = node_features
            
        if hyperedge_features is not None:
            hyperedge_features_in = hyperedge_features

        # Self-conditioning (if enabled)
        if self.self_conditioning:
            if model.training:
                if np.random.rand() < 0.5:
                    # Self-conditioning pass
                    with th.no_grad():
                        node_attr_self_cond, hyperedge_attr_self_cond, incidence_attr_self_cond, node_features_self_cond, hyperedge_features_self_cond = model(
                            incidence_index=incidence_index,
                            batch=batch,
                            node_type=node_type,
                            node_cluster_size=node_cluster_size,
                            node_attr=th.cat(
                                [node_attr_in, th.zeros_like(node_attr_in)], dim=-1
                            ).float(),
                            hyperedge_attr=th.cat(
                                [hyperedge_attr_in, th.zeros_like(hyperedge_attr_in)], dim=-1
                            ).float(),
                            incidence_attr=th.cat(
                                [incidence_attr_in, th.zeros_like(incidence_attr_in)], dim=-1
                            ).float(),
                            node_features_initial=node_features_initial,
                            node_features=th.cat(
                                [node_features_in, th.zeros_like(node_features_in)], dim=-1
                            ).float() if node_features is not None else None,
                            hyperedge_features_initial=hyperedge_features_initial,
                            hyperedge_features=th.cat(
                                [hyperedge_features_in, th.zeros_like(hyperedge_features_in)], dim=-1
                            ).float() if hyperedge_features is not None else None,
                            **model_kwargs,
                        )
                        
                        # Project budgets on the simplex (renormalized to [-1, 1]) using softmax
                        node_attr_self_cond[:, 1] = th.clamp(node_attr_self_cond[:, 1] + 1, min=1e-10)/2
                        node_attr_self_cond[:, 1] = (node_attr_self_cond[:, 1] / scatter(node_attr_self_cond[:, 1], clusters_nodes, reduce="sum")[clusters_nodes]) * 2 - 1
                        node_attr_self_cond[node_cluster_size == 1, 0] = -1 # Nodes without budget cannot be further expanded
                        node_attr_self_cond[nodes_expanded_cluster_size_two, 1] = 0 # When the cluster is expanded and its budget is 2, the fraction is necessarily half and half
                        
                        node_attr_self_cond = node_attr_self_cond.detach()
                        hyperedge_attr_self_cond = hyperedge_attr_self_cond.detach()
                        incidence_attr_self_cond = incidence_attr_self_cond.detach()
                        
                        size = scatter(th.ones_like(batch), batch)
                        if node_features is not None:
                            if self.node_features_on_simplex:
                                node_features_self_cond[size[batch[node_type == 1]] != 0] = project_to_simplex(node_features_self_cond[size[batch[node_type == 1]] != 0])
                                
                            node_features_self_cond[nodes_not_expanded] = node_features_initial[nodes_not_expanded]
                            node_features_self_cond = node_features_self_cond.detach()
                            
                        if hyperedge_features is not None:
                            if self.hyperedge_features_on_simplex:
                                hyperedge_features_self_cond[size[batch[node_type == 0]] != 0] = project_to_simplex(hyperedge_features_self_cond[size[batch[node_type == 0]] != 0])
                                
                            hyperedge_features_self_cond[hyperedges_not_expanded] = hyperedge_features_initial[hyperedges_not_expanded]
                            hyperedge_features_self_cond = hyperedge_features_self_cond.detach()
                else:
                    node_attr_self_cond = th.zeros_like(node_attr)
                    hyperedge_attr_self_cond = th.zeros_like(hyperedge_attr)
                    incidence_attr_self_cond = th.zeros_like(incidence_attr)
                    
                    if node_features is not None:
                        node_features_self_cond = th.zeros_like(node_features)
                        
                    if hyperedge_features is not None:
                        hyperedge_features_self_cond = th.zeros_like(hyperedge_features)

            # Concatenate with input
            node_attr_in = th.cat([node_attr_in, node_attr_self_cond], dim=-1)
            hyperedge_attr_in = th.cat([hyperedge_attr_in, hyperedge_attr_self_cond], dim=-1)
            incidence_attr_in = th.cat([incidence_attr_in, incidence_attr_self_cond], dim=-1)
            
            if node_features is not None:
                node_features_in = th.cat([node_features_in, node_features_self_cond], dim=-1)
                
            if hyperedge_features is not None:
                hyperedge_features_in = th.cat([hyperedge_features_in, hyperedge_features_self_cond], dim=-1)

        # Compute model output
        node_attr_pred, hyperedge_attr_pred, incidence_attr_pred, node_features_pred, hyperedge_features_pred = model(
            incidence_index=incidence_index,
            batch=batch,
            node_type=node_type,
            node_cluster_size=node_cluster_size,
            node_attr=node_attr_in.float(),
            hyperedge_attr=hyperedge_attr_in.float(),
            incidence_attr=incidence_attr_in.float(),
            node_features_initial=node_features_initial,
            node_features=node_features_in.float() if node_features is not None else None,
            hyperedge_features_initial=hyperedge_features_initial,
            hyperedge_features=hyperedge_features_in.float() if hyperedge_features is not None else None,
            **model_kwargs,
        )
        
        return node_attr_pred, hyperedge_attr_pred, incidence_attr_pred, node_features_pred, hyperedge_features_pred


class CFM:
    """Conditional Flow Matching Framework"""
    alpha = 1.5
    
    def __init__(self, self_conditioning, num_steps, node_features_on_simplex=False, hyperedge_features_on_simplex=False):
        self.model_wrapper = CFMModel(self_conditioning, node_features_on_simplex, hyperedge_features_on_simplex)
        self.num_steps = num_steps
        self.node_features_on_simplex = node_features_on_simplex
        self.hyperedge_features_on_simplex = hyperedge_features_on_simplex
    
    @property
    def device(self):
        assert hasattr(self, "_device")
        return self._device

    def to(self, device):
        self._device = device
        self.model_wrapper.to(device)
        return self
    
    def get_loss(
        self,
        incidence_index,
        batch,
        node_type,
        expansion_matrix_nodes,
        expansion_matrix_hyperedges,
        node_attr,
        hyperedge_attr,
        incidence_attr,
        node_features_expanded,
        node_features_real,
        node_cluster_size_expanded,
        node_cluster_size_real,
        hyperedge_features_expanded,
        hyperedge_features_real,
        model,
        model_kwargs
    ):
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
        nodes_not_expanded = row[~th.isin(col, expanded_clusters)]
        nodes_expanded_cluster_size_two = nodes_expanded[node_cluster_size_expanded[nodes_expanded] == 2]
        nodes_for_budget_loss = nodes_expanded[node_cluster_size_expanded[nodes_expanded] > 2]
        
        
        row, col, _ = expansion_matrix_hyperedges.coo()
        clusters_hyperedges = col[th.argsort(row)]
        
        unique_clusters, cluster_counts = th.unique(clusters_hyperedges, return_counts=True)
        expanded_clusters = unique_clusters[cluster_counts != 1]
        hyperedges_not_expanded = row[~th.isin(col, expanded_clusters)]
        hyperedges_expanded = row[th.isin(col, expanded_clusters)]
        
        # Compute start and end points
        incidence_batch = batch[incidence_index[0]]
        
        # compute start and end points for node attributes
        ratio_node_cluster_size = (node_cluster_size_real/node_cluster_size_expanded)*2 - 1 # Rescale between -1 and 1
        node_end = th.stack((node_attr, ratio_node_cluster_size), dim=1)
        
        node_attr_start = th.randn_like(node_attr)
        
        # Force initial point for node budgets to be on the simplex (transformed into [-1, 1])
        node_budget_start = th.distributions.Gamma(self.alpha*th.ones_like(ratio_node_cluster_size), th.ones_like(ratio_node_cluster_size)).sample().clamp(min=1e-10)
        node_budget_start = (node_budget_start / scatter(node_budget_start, clusters_nodes)[clusters_nodes])*2 - 1
        
        node_start = th.stack((node_attr_start, node_budget_start), dim=1)
        
        
        hyperedge_start = th.randn_like(hyperedge_attr)
        
        incidence_start = self.incidence_randn(incidence_index, dtype=th.float32)
            
        node_features_start = None
        node_features_end = None
        if node_features_expanded is not None:
            # Compute target
            node_features_end = node_features_real
            
            # Compute starting point
            if self.node_features_on_simplex:
                node_features_start = th.distributions.Dirichlet(th.full((node_features_expanded.shape[1],), self.alpha, device=node_features_expanded.device)).sample((node_features_expanded.shape[0],))
            else:
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
            if self.hyperedge_features_on_simplex:
                hyperedge_features_start = th.distributions.Dirichlet(th.full((hyperedge_features_expanded.shape[1],), self.alpha, device=hyperedge_features_expanded.device)).sample((hyperedge_features_expanded.shape[0],))
            else:
                hyperedge_features_start = th.randn_like(hyperedge_features_expanded)
            
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
        perm = self.minibatch_ot_coupling(expansion_matrix_hyperedges, hyperedge_start, hyperedge_attr, hyperedge_features_start, hyperedge_features_end, device=self.device)
        
        hyperedge_start = hyperedge_start[perm]

        if hyperedge_features_expanded is not None:
            hyperedge_features_start = hyperedge_features_start[perm]
            
            
        # Forward model to predict flow
        node_attr_pred, hyperedge_attr_pred, incidence_attr_pred, node_features_pred, hyperedge_features_pred = self.model_wrapper(
            incidence_index=incidence_index,
            batch=batch,
            node_type=node_type,
            node_cluster_size=node_cluster_size_expanded,
            node_attr=t[batch[node_type==1]].unsqueeze(1) * node_end + (1-t[batch[node_type==1]].unsqueeze(1)) * node_start,
            hyperedge_attr=(t[batch[node_type==0]] * hyperedge_attr + (1-t[batch[node_type==0]]) * hyperedge_start).unsqueeze(1),
            incidence_attr=(t[incidence_batch] * incidence_attr + (1-t[incidence_batch]) * incidence_start).unsqueeze(1),
            node_features_initial=node_features_expanded,
            node_features=t_expanded_node_features[node_type == 1] * node_features_end + (1-t_expanded_node_features[node_type == 1]) * node_features_start if node_features_expanded is not None else None,
            hyperedge_features_initial=hyperedge_features_expanded,
            hyperedge_features=t_expanded_hyperedge_features[node_type == 0] * hyperedge_features_end + (1-t_expanded_hyperedge_features[node_type == 0]) * hyperedge_features_start if hyperedge_features_expanded is not None else None,
            t=t,
            model=model,
            model_kwargs=model_kwargs,
            clusters_nodes=clusters_nodes,
            nodes_expanded_cluster_size_two=nodes_expanded_cluster_size_two,
            nodes_expanded=nodes_expanded,
            nodes_not_expanded=nodes_not_expanded,
            hyperedges_not_expanded=hyperedges_not_expanded
        )

        # Compute loss
        node_attr_pred = node_attr_pred.float()
        hyperedge_attr_pred = hyperedge_attr_pred.float().squeeze(1)
        incidence_attr_pred = incidence_attr_pred.float().squeeze(1)
        
        node_loss = (node_attr_pred - node_end) ** 2
        hyperedge_loss = (hyperedge_attr_pred - hyperedge_attr)**2
        incidence_loss = (incidence_attr_pred - incidence_attr) ** 2
        
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
        
        hyperedge_loss = scatter(hyperedge_loss, batch[node_type == 0], reduce="mean").mean()
        incidence_loss = scatter(incidence_loss, batch[incidence_index[0]], reduce="mean").mean()
        
        if node_features_expanded is not None:
            node_features_loss = scatter(node_features_loss[nodes_expanded].sum(dim=-1), batch[node_type == 1][nodes_expanded], reduce="mean").mean()
            
        if hyperedge_features_expanded is not None:
            hyperedge_features_loss = scatter(hyperedge_features_loss[hyperedges_expanded].sum(dim=-1), batch[node_type == 0][hyperedges_expanded], reduce="mean").mean()            
           
        return node_loss, hyperedge_loss, incidence_loss, node_features_loss, hyperedge_features_loss


    @th.no_grad()
    def sample(
        self,
        incidence_index,
        node_type,
        node_features,
        hyperedge_features,
        node_cluster_size,
        expansion_matrix_nodes,
        expansion_matrix_hyperedges,
        batch,
        model,
        model_kwargs
    ):
        """Sampling procedure for CFM."""
        num_graphs = batch.max().item() + 1
        size = scatter(th.ones_like(batch), batch)
        
        row, col, _ = expansion_matrix_nodes.coo()
        clusters_nodes = col[th.argsort(row)]

        unique_clusters, cluster_counts = th.unique(clusters_nodes, return_counts=True)
        expanded_clusters = unique_clusters[cluster_counts != 1]
        nodes_expanded = row[th.isin(col, expanded_clusters)]
        nodes_not_expanded = row[~th.isin(col, expanded_clusters)]
        nodes_expanded_cluster_size_two = nodes_expanded[node_cluster_size[nodes_expanded] == 2]


        row, col, _ = expansion_matrix_hyperedges.coo()
        clusters_hyperedges = col[th.argsort(row)]
        
        unique_clusters, cluster_counts = th.unique(clusters_hyperedges, return_counts=True)
        expanded_clusters = unique_clusters[cluster_counts != 1]
        hyperedges_not_expanded = row[~th.isin(col, expanded_clusters)]

        # Sample random initialization
        node_attr_start = th.randn(batch[node_type == 1].shape[0], dtype=th.float32, device=self.device)
        
        # Force initial point for node budgets to be on the simplex (transformed into [-1, 1])
        node_budget_start = th.distributions.Gamma(self.alpha*th.ones(batch[node_type == 1].shape[0], dtype=th.float32, device=self.device), th.ones(batch[node_type == 1].shape[0], dtype=th.float32, device=self.device)).sample().clamp(min=1e-10)
        node_budget_start = (node_budget_start / scatter(node_budget_start, clusters_nodes)[clusters_nodes])*2 - 1

        node_attr_next = th.stack((node_attr_start, node_budget_start), dim=1)
        
        hyperedge_attr_next = th.randn(batch[node_type == 0].shape[0], dtype=th.float32, device=self.device)[:, None]
        incidence_attr_next = self.incidence_randn(incidence_index, dtype=th.float32)[:, None]
        
        node_features_next = None
        if node_features is not None:
            if self.node_features_on_simplex:
                node_features_next = th.distributions.Dirichlet(th.full((node_features.shape[1],), self.alpha, device=node_features.device)).sample((node_features.shape[0],))
            else:
                node_features_next = th.randn_like(node_features)
            
        hyperedge_features_next = None
        if hyperedge_features is not None:
            if self.hyperedge_features_on_simplex:
                hyperedge_features_next = th.distributions.Dirichlet(th.full((hyperedge_features.shape[1],), self.alpha, device=hyperedge_features.device)).sample((hyperedge_features.shape[0],))
            else:
                hyperedge_features_next = th.randn_like(hyperedge_features)
            
        # For self-conditioning
        k2_node_attr = th.zeros_like(node_attr_next)
        k2_hyperedge_attr = th.zeros_like(hyperedge_attr_next)
        k2_incidence_attr = th.zeros_like(incidence_attr_next)
        
        k2_node_features = None
        if node_features is not None:
            k2_node_features = th.zeros_like(node_features)
        
        k2_hyperedge_features = None
        if hyperedge_features is not None:
            k2_hyperedge_features = th.zeros_like(hyperedge_features) 
        
        # Flow matching sampling: Integrate learned flow model over time (0 to 1) with Heun's method
        t_steps = th.linspace(0, 1, steps=self.num_steps, device=self.device)
        dt = t_steps[1] - t_steps[0]
        
        for i in range(len(t_steps[:-1])):            
            # First step: Calculate initial derivatives (k1)
            k1_node_attr, k1_hyperedge_attr, k1_incidence_attr, k1_node_features, k1_hyperedge_features = self.model_wrapper(
                incidence_index=incidence_index,
                batch=batch,
                node_type=node_type,
                node_cluster_size=node_cluster_size,
                node_attr=node_attr_next,
                hyperedge_attr=hyperedge_attr_next,
                incidence_attr=incidence_attr_next,
                node_features_initial=node_features,
                node_features=node_features_next,
                hyperedge_features_initial=hyperedge_features,
                hyperedge_features=hyperedge_features_next,
                t=t_steps[i].repeat(num_graphs),
                model=model,
                model_kwargs=model_kwargs,
                node_attr_self_cond=k2_node_attr,
                hyperedge_attr_self_cond=k2_hyperedge_attr,
                incidence_attr_self_cond=k2_incidence_attr,
                node_features_self_cond=k2_node_features,
                hyperedge_features_self_cond=k2_hyperedge_features
            )
            
            # Project budgets on the simplex (renormalized to [-1, 1]) using softmax
            k1_node_attr[:, 1] = th.clamp(k1_node_attr[:, 1] + 1, min=1e-10)/2
            k1_node_attr[:, 1] = (k1_node_attr[:, 1] / scatter(k1_node_attr[:, 1], clusters_nodes, reduce="sum")[clusters_nodes]) * 2 - 1
            k1_node_attr[node_cluster_size == 1, 0] = -1 # Nodes without budget cannot be further expanded
            k1_node_attr[nodes_expanded_cluster_size_two, 1] = 0 # When the cluster is expanded and its budget is 2, the fraction is necessarily half and half
            
            # 1st order estimation
            node_attr_pred = node_attr_next + dt * (k1_node_attr - node_attr_next)/(1-t_steps[i])
            hyperedge_attr_pred = hyperedge_attr_next + dt * (k1_hyperedge_attr - hyperedge_attr_next)/(1-t_steps[i])
            incidence_attr_pred = incidence_attr_next + dt * (k1_incidence_attr - incidence_attr_next)/(1-t_steps[i])
            
            node_features_pred = None
            if node_features is not None:
                if self.node_features_on_simplex:
                    k1_node_features[size[batch[node_type == 1]] != 0] = project_to_simplex(k1_node_features[size[batch[node_type == 1]] != 0])
                    
                k1_node_features[nodes_not_expanded] = node_features[nodes_not_expanded]
                node_features_pred = node_features_next + dt * (k1_node_features - node_features_next)/(1-t_steps[i])
                
            hyperedge_features_pred = None
            if hyperedge_features is not None:
                if self.hyperedge_features_on_simplex:
                    k1_hyperedge_features[size[batch[node_type == 0]] != 0] = project_to_simplex(k1_hyperedge_features[size[batch[node_type == 0]] != 0])
                    
                k1_hyperedge_features[hyperedges_not_expanded] = hyperedge_features[hyperedges_not_expanded]
                hyperedge_features_pred = hyperedge_features_next + dt * (k1_hyperedge_features - hyperedge_features_next)/(1-t_steps[i])
                
            
            # 2nd order correction
            if i < self.num_steps - 2:
                k2_node_attr, k2_hyperedge_attr, k2_incidence_attr, k2_node_features, k2_hyperedge_features = self.model_wrapper(
                    incidence_index=incidence_index,
                    batch=batch,
                    node_type=node_type,
                    node_cluster_size=node_cluster_size,
                    node_attr=node_attr_pred,
                    hyperedge_attr=hyperedge_attr_pred,
                    incidence_attr=incidence_attr_pred,
                    node_features_initial=node_features,
                    node_features=node_features_pred,
                    hyperedge_features_initial=hyperedge_features,
                    hyperedge_features=hyperedge_features_pred,
                    t=t_steps[i+1].repeat(num_graphs),
                    model=model,
                    model_kwargs=model_kwargs,
                    node_attr_self_cond=k1_node_attr,
                    hyperedge_attr_self_cond=k1_hyperedge_attr,
                    incidence_attr_self_cond=k1_incidence_attr,
                    node_features_self_cond=k1_node_features,
                    hyperedge_features_self_cond=k1_hyperedge_features
                )
                
                # Project budgets on the simplex (renormalized to [-1, 1]) using softmax
                k2_node_attr[:, 1] = th.clamp(k2_node_attr[:, 1] + 1, min=1e-10)/2
                k2_node_attr[:, 1] = (k2_node_attr[:, 1] / scatter(k2_node_attr[:, 1], clusters_nodes, reduce="sum")[clusters_nodes]) * 2 - 1
                k2_node_attr[node_cluster_size == 1, 0] = -1 # Nodes without budget cannot be further expanded
                k2_node_attr[nodes_expanded_cluster_size_two, 1] = 0 # When the cluster is expanded and its budget is 2, the fraction is necessarily half and half
                
                
                # Heun's method update: Average of k1 and k2
                node_attr_next = ( node_attr_next + 0.5 * dt * (
                    (k1_node_attr - node_attr_next)/(1-t_steps[i])
                    + (k2_node_attr - node_attr_pred)/(1-t_steps[i+1])
                    ) )
                hyperedge_attr_next = ( hyperedge_attr_next + 0.5 * dt * (
                    (k1_hyperedge_attr - hyperedge_attr_next)/(1-t_steps[i])
                    + (k2_hyperedge_attr - hyperedge_attr_pred)/(1-t_steps[i+1])
                    ) )
                incidence_attr_next = ( incidence_attr_next + 0.5 * dt * (
                    (k1_incidence_attr - incidence_attr_next)/(1-t_steps[i])
                    + (k2_incidence_attr - incidence_attr_pred)/(1-t_steps[i+1])
                    ) )
                
                if node_features is not None:
                    if self.node_features_on_simplex:
                        k2_node_features[size[batch[node_type == 1]] != 0] = project_to_simplex(k2_node_features[size[batch[node_type == 1]] != 0])
                        
                    k2_node_features[nodes_not_expanded] = node_features[nodes_not_expanded]
                    node_features_next = ( node_features_next + 0.5 * dt * (
                        (k1_node_features - node_features_next)/(1-t_steps[i])
                        + (k2_node_features - node_features_pred)/(1-t_steps[i+1])
                    ) )
                    
                if hyperedge_features is not None:
                    if self.hyperedge_features_on_simplex:
                        k2_hyperedge_features[size[batch[node_type == 0]] != 0] = project_to_simplex(k2_hyperedge_features[size[batch[node_type == 0]] != 0])
                    
                    k2_hyperedge_features[hyperedges_not_expanded] = hyperedge_features[hyperedges_not_expanded]
                    hyperedge_features_next = ( hyperedge_features_next + 0.5 * dt * (
                        (k1_hyperedge_features - hyperedge_features_next)/(1-t_steps[i])
                        + (k2_hyperedge_features - hyperedge_features_pred)/(1-t_steps[i+1])
                    ) )
            else:
                node_attr_next = node_attr_pred
                hyperedge_attr_next = hyperedge_attr_pred
                incidence_attr_next = incidence_attr_pred
                
                if node_features is not None:
                    node_features_next = node_features_pred
                    
                if hyperedge_features is not None:
                    hyperedge_features_next = hyperedge_features_pred
        
        
        return node_attr_next, hyperedge_attr_next.squeeze(1), incidence_attr_next.squeeze(1), node_features_next if node_features_next is not None else None, hyperedge_features_next if hyperedge_features_next is not None else None


    @staticmethod
    def incidence_randn(incidence_index, dtype=th.float32) -> th.Tensor:
        """Sample symmetric Gaussian noise for incidences in incidence_index."""
        # sample noise for upper triangle
        incidence_index_u = incidence_index[:, incidence_index[0] <= incidence_index[1]]
        incidence_noise_u = th.randn_like(incidence_index_u[0], dtype=dtype)
    
        # make symmetric
        new_incidence_index, incidence_noise = sort_edge_index(
            edge_index=th.cat([incidence_index_u, incidence_index_u.flip(0)], dim=1),
            edge_attr=th.cat([incidence_noise_u, incidence_noise_u], dim=0),
        )
        assert (incidence_index == new_incidence_index).all()
    
        return incidence_noise
    
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
        
            # Filter nodes that belong to clusters with three nodes
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

def project_to_simplex(v):
    # Sort the tensor in descending order along the last dimension (for each vector in the batch)
    sorted_v, _ = th.sort(v, descending=True, dim=-1)
    
    # Calculate the cumulative sum of the sorted values along the last dimension
    cssv = th.cumsum(sorted_v, dim=-1)
    
    # Create an index tensor for rho where the condition holds
    # We need the max index where the following condition is satisfied:
    #   sorted_v[i] - (cssv[i] - 1) / (i + 1) > 0
    rho = (sorted_v - (cssv - 1) / th.arange(1, v.size(-1) + 1, device=v.device).float()).gt(0).sum(dim=-1) - 1

    # Calculate the threshold value theta
    theta = (1 - cssv.gather(-1, rho.unsqueeze(-1))) / (rho.float() + 1).unsqueeze(-1)

    # Apply the threshold and ensure the values are non-negative
    projected = th.maximum(v + theta, th.zeros_like(v))
    
    return projected