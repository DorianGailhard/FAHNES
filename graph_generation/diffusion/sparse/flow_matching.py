import torch as th
from torch.nn import Module
from torch_geometric.utils import sort_edge_index
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
        node_attr,
        edge_node_attr,
        edge_attr,
        node_features,
        t,
        model,
        model_kwargs,
        node_attr_self_cond=None,
        edge_node_attr_self_cond=None,
        edge_attr_self_cond=None,
    ):
        # For flow matching, no preconditioning is needed.
        model_kwargs = dict(model_kwargs, noise_cond=t.float())
        
        node_attr_in = node_attr
        edge_node_attr_in = edge_node_attr
        edge_attr_in = edge_attr
        
        if node_features is not None:
            node_features_in = node_features

        # Self-conditioning (if enabled)
        if self.self_conditioning:
            if not model.training:
                assert (
                    node_attr_self_cond is not None and 
                    edge_node_attr_self_cond is not None and 
                    edge_attr_self_cond is not None
                )
            elif np.random.rand() < 0.5:
                # Self-conditioning pass
                with th.no_grad():
                    node_attr_self_cond, edge_node_attr_self_cond, edge_attr_self_cond, node_features_nodes_self_cond, node_features_edge_nodes_self_cond = model(
                        edge_index=edge_index,
                        batch=batch,
                        node_type=node_type,
                        node_attr=th.cat(
                            [node_attr_in, th.zeros_like(node_attr_in)], dim=-1
                        ).float(),
                        edge_node_attr=th.cat(
                            [edge_node_attr_in, th.zeros_like(edge_node_attr_in)], dim=-1
                        ).float(),
                        edge_attr=th.cat(
                            [edge_attr_in, th.zeros_like(edge_attr_in)], dim=-1
                        ).float(),
                        **model_kwargs,
                        node_features=th.cat(
                            [node_features_in, th.zeros_like(node_features_in)], dim=-1
                        ).float() if node_features is not None else None,
                    )
                    node_attr_self_cond = node_attr_self_cond.detach()
                    edge_node_attr_self_cond = edge_node_attr_self_cond.detach()
                    edge_attr_self_cond = edge_attr_self_cond.detach()
                    
                    if node_features is not None:
                        node_features_self_cond = th.zeros_like(node_features)
                        node_features_self_cond[node_type == 1] = node_features_nodes_self_cond
                        node_features_self_cond[node_type == 0] = node_features_edge_nodes_self_cond
                        node_features_self_cond = node_features_self_cond.detach()
            else:
                node_attr_self_cond = th.zeros_like(node_attr)
                edge_node_attr_self_cond = th.zeros_like(edge_node_attr)
                edge_attr_self_cond = th.zeros_like(edge_attr)
                
                if node_features is not None:
                    node_features_self_cond = th.zeros_like(node_features)

            # Concatenate with input
            node_attr_in = th.cat([node_attr_in, node_attr_self_cond], dim=-1)
            edge_node_attr_in = th.cat([edge_node_attr_in, edge_node_attr_self_cond], dim=-1)
            edge_attr_in = th.cat([edge_attr_in, edge_attr_self_cond], dim=-1)
            
            if node_features is not None:
                node_features_in = th.cat([node_features_in, node_features_self_cond], dim=-1)

        # Compute model output
        node_attr_pred, edge_node_attr_pred, edge_attr_pred, node_features_nodes_pred, node_features_edge_nodes_pred = model(
            edge_index=edge_index,
            batch=batch,
            node_type=node_type,
            node_attr=node_attr_in.float(),
            edge_node_attr=edge_node_attr_in.float(),
            edge_attr=edge_attr_in.float(),
            node_features=node_features_in.float() if node_features is not None else None,
            **model_kwargs,
        )
        
        return node_attr_pred, edge_node_attr_pred, edge_attr_pred, node_features_nodes_pred, node_features_edge_nodes_pred


class CFM:
    """Conditional Flow Matching Framework"""
    sigma_min = 0.001
    
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
    
    def get_loss(self, edge_index, batch, node_type, node_attr, edge_node_attr, edge_attr, node_features_expanded, node_features_real, hyperedge_features_expanded, hyperedge_features_real, model, model_kwargs):
        """Loss function for conditional flow matching."""
        # rescale attributes to {-1, 1}
        node_attr = node_attr.float() * 2 - 1
        edge_node_attr = edge_node_attr.float() - 1
        edge_attr = edge_attr.float() * 2 - 1
        
        # sample noise level
        num_graphs = batch.max().item() + 1
        t = th.rand((num_graphs,), device=self.device)

        # sample noise
        edge_batch = batch[edge_index[0]]

        node_noise = th.randn_like(node_attr)
        edge_node_noise = th.randn_like(edge_node_attr)
        edge_noise = self.edge_randn(edge_index)
        
        node_features_start = None
        node_features_full = None
        if node_features_expanded is not None:
            # Compute target
            node_features_full = th.zeros((node_type.shape[0],) + node_features_expanded.shape[1:], device=self.device)
            node_features_full[node_type == 1] = node_features_real
            
            # For hyperedges, set features as average of the nodes it contains
            # Get indices of the nodes forming edges from edge_index
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            
            # Initialize neighbor sum and count to accumulate features and number of neighbors
            neighbor_sum = th.zeros_like(node_features_full)
            neighbor_count = th.zeros(node_type.shape[0], device=self.device)
            
            # Accumulate the features of the neighbors
            neighbor_sum.index_add_(0, src_nodes, node_features_full[dst_nodes])
            neighbor_sum.index_add_(0, dst_nodes, node_features_full[src_nodes])
            
            # Accumulate the count of neighbors
            neighbor_count.index_add_(0, src_nodes, th.ones_like(src_nodes, dtype=neighbor_count.dtype))
            neighbor_count.index_add_(0, dst_nodes, th.ones_like(dst_nodes, dtype=neighbor_count.dtype))
            
            # Avoid division by zero (in case a node has no neighbors)
            neighbor_count = neighbor_count.clamp(min=1)
            
            # Compute the average of neighbor features
            n_dims = node_features_full[node_type == 1].dim()
            neighbor_avg = neighbor_sum / neighbor_count.view(-1, *([1] * (n_dims - 1)))
            node_features_full[node_type == 0] = neighbor_avg[node_type == 0]
            
            # Compute starting point
            node_features_start = th.zeros_like(node_features_full)
            node_features_start[node_type == 1] = node_features_expanded
            node_features_start[node_type == 0] = th.rand_like(node_features_start[node_type == 0])
            
            # Expand t to allow multipliying
            n_dims = node_features_full.dim()
            t_expanded = t[batch].view(-1, *([1] * (n_dims - 1)))
            
        # Forward model to predict flow
        node_attr_pred, edge_node_attr_pred, edge_attr_pred, node_features_nodes_pred, node_features_edge_nodes_pred = self.model_wrapper(
            edge_index=edge_index,
            batch=batch,
            node_type=node_type,
            node_attr=(t[batch[node_type==1]] * node_attr + (1-(1-self.sigma_min)*t[batch[node_type==1]]) * node_noise).unsqueeze(1),
            edge_node_attr=(t[batch[node_type==0]] * edge_node_attr + (1-(1-self.sigma_min)*t[batch[node_type==0]]) * edge_node_noise).unsqueeze(1),
            edge_attr=(t[edge_batch] * edge_attr + (1-(1-self.sigma_min)*t[edge_batch]) * edge_noise).unsqueeze(1),
            node_features=t_expanded * node_features_full + (1-t_expanded) * node_features_start + self.sigma_min*th.randn_like(node_features_full) if node_features_expanded is not None else None,
            t=t,
            model=model,
            model_kwargs=model_kwargs,
        )
        
        node_attr_pred = node_attr_pred.float().squeeze(1)
        edge_node_attr_pred = edge_node_attr_pred.float().squeeze(1)
        edge_attr_pred = edge_attr_pred.float().squeeze(1)

        node_loss = (node_attr_pred - (node_attr - node_noise)) ** 2
        edge_node_loss = (edge_node_attr_pred - (edge_node_attr - edge_node_noise)) ** 2
        edge_loss = (edge_attr_pred - (edge_attr - edge_noise)) ** 2
        
        nodes_node_features_loss = None
        edge_nodes_node_features_loss = None
        
        if node_features_expanded is not None:
            nodes_node_features_loss = (node_features_nodes_pred - (node_features_full[node_type == 1] - node_features_start[node_type == 1]))**2
            edge_nodes_node_features_loss = (node_features_edge_nodes_pred - (node_features_full[node_type == 0] - node_features_start[node_type == 0]))**2
            
        return node_loss, edge_node_loss, edge_loss, nodes_node_features_loss, edge_nodes_node_features_loss, None


    @th.no_grad()
    def sample(self, edge_index, node_type, node_features, hyperedge_features, batch, model, model_kwargs):
        """Sampling procedure for CFM."""
        num_graphs = batch.max().item() + 1
        
        # Start from random initialization
        node_attr_next = th.randn_like(batch[node_type == 1], dtype=th.float32)[:, None]
        edge_node_attr_next = th.randn_like(batch[node_type == 0], dtype=th.float32)[:, None]
        edge_attr_next = self.edge_randn(edge_index, dtype=th.float32)[:, None]

        node_features_next = None
        if node_features is not None:
            node_features_next = th.zeros((node_type.shape[0],) + node_features.shape[1:], device=self.device, dtype=th.float32)
            node_features_next[node_type == 1] = node_features
            node_features_next[node_type == 0] = th.randn_like(node_features_next[node_type == 0])

        # Flow matching sampling: Integrate learned flow model over time (0 to 1)
        t_steps = th.linspace(0, 1, steps=self.num_steps, device=self.device)

        for t_cur in t_steps:
            # Forward pass through the model to get flow direction
            node_attr_pred, edge_node_attr_pred, edge_attr_pred, node_features_nodes_pred, node_features_edge_nodes_pred = self.model_wrapper(
                edge_index=edge_index,
                batch=batch,
                node_type=node_type,
                node_attr=node_attr_next,
                edge_node_attr=edge_node_attr_next,
                edge_attr=edge_attr_next,
                node_features=node_features_next,
                t=t_cur.repeat(num_graphs),
                model=model,
                model_kwargs=model_kwargs,
            )
            
            # Euler step: Integrate the flow over time to update the attributes and features
            dt = t_steps[1] - t_steps[0]
            node_attr_next += dt * node_attr_pred
            edge_node_attr_next += dt * edge_node_attr_pred
            edge_attr_next += dt * edge_attr_pred
            
            if node_features is not None:
                node_features_next[node_type == 1] += dt * node_features_nodes_pred
                node_features_next[node_type == 0] += dt * node_features_edge_nodes_pred

        # rescale attributes to {0, 1} and {0, 2}
        node_attr_out = (node_attr_next + 1) / 2
        edge_node_attr_out = edge_node_attr_next + 1
        edge_attr_out = (edge_attr_next + 1) / 2

        return node_attr_out.squeeze(1), edge_node_attr_out.squeeze(1), edge_attr_out.squeeze(1), node_features_next[node_type == 1] if node_features_next is not None else None, None


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