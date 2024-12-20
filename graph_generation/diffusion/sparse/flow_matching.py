import torch as th
from torch.nn import Module
from torch_geometric.utils import sort_edge_index
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
                        **model_kwargs,
                        node_features=th.cat(
                            [node_features_in, th.zeros_like(node_features_in)], dim=-1
                        ).float() if node_features is not None else None,
                    )
                    node_attr_self_cond = node_attr_self_cond.detach()
                    edge_node_attr_self_cond = edge_node_attr_self_cond.detach()
                    edge_attr_self_cond = edge_attr_self_cond.detach()
                    
                    if node_features is not None:
                        """node_features_nodes_self_cond = th.exp(node_features_nodes_self_cond[:,0::2]) * node_features_nodes_self_cond[:,1::2]
                        node_features_edge_nodes_self_cond = th.exp(node_features_edge_nodes_self_cond[:,0::2]) * node_features_edge_nodes_self_cond[:,1::2]
                        """
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
            node_cluster_size=node_cluster_size,
            node_attr=node_attr_in.float(),
            edge_node_attr=edge_node_attr_in.float(),
            edge_attr=edge_attr_in.float(),
            node_features=node_features_in.float() if node_features is not None else None,
            **model_kwargs,
        )
        
        """# The direction magnitude for features can be both small (precision is needed, refinement) or huge (precision not needed, coarse approximation refined later)
        if node_features_nodes_pred is not None:
            node_features_nodes_pred = th.exp(node_features_nodes_pred[:,0::2]) * node_features_nodes_pred[:,1::2]
            node_features_edge_nodes_pred = th.exp(node_features_edge_nodes_pred[:,0::2]) * node_features_edge_nodes_pred[:,1::2]
        """
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
    
    def get_loss(self, edge_index, batch, node_type, node_attr, edge_node_attr, edge_attr, node_features_expanded, node_features_real, node_cluster_size_expanded, node_cluster_size_real, hyperedge_features_expanded, hyperedge_features_real, model, model_kwargs):
        """Loss function for conditional flow matching."""
        # sample noise level
        num_graphs = batch.max().item() + 1
        t = th.rand((num_graphs,), device=self.device)
        
        # Compute start and end points
        edge_batch = batch[edge_index[0]]

        # compute start and end points for node attributes
        node_attr_start = th.randn_like(node_attr)
        node_start = th.cat((node_attr_start[:, None], node_cluster_size_expanded), dim=1)
        
        node_end = th.cat((node_attr[:, None], node_cluster_size_real), dim=1)
        node_end += th.randn_like(node_end)*self.sigma_min
        
        # For hyperedges, set final cluster_size as average of the nodes it contains
        edge_node_end = self.compute_edge_node_averages(node_type, edge_index, node_cluster_size_expanded, self.device)
        edge_node_end = th.cat((edge_node_attr[:, None], edge_node_end), dim=1)
        
        edge_node_start = th.randn_like(edge_node_end)
        edge_node_end += th.randn_like(edge_node_end)*self.sigma_min
        
        
        edge_start = self.edge_randn(edge_index, dtype=th.float32)
        
        
        node_features_start = None
        node_features_full = None
        if node_features_expanded is not None:
            # Compute target
            node_features_full = th.zeros((node_type.shape[0],) + node_features_expanded.shape[1:], device=self.device)
            node_features_full[node_type == 1] = node_features_real
            node_features_full[node_type == 0] = self.compute_edge_node_averages(node_type, edge_index, node_features_expanded, self.device)
            
            node_features_full += th.randn_like(node_features_full) * self.sigma_min
            
            # Compute starting point
            node_features_start = th.zeros_like(node_features_full)
            node_features_start[node_type == 1] = node_features_expanded + th.randn_like(node_features_expanded) * self.sigma_min
            node_features_start[node_type == 0] = th.randn_like(node_features_start[node_type == 0])
            
            # Expand t to allow multipliying
            n_dims = node_features_full.dim()
            t_expanded = t[batch].view(-1, *([1] * (n_dims - 1)))
        
        t_expanded_attr = t[batch].view(-1, 1)
        
        # Forward model to predict flow
        node_attr_pred, edge_node_attr_pred, edge_attr_pred, node_features_nodes_pred, node_features_edge_nodes_pred = self.model_wrapper(
            edge_index=edge_index,
            batch=batch,
            node_type=node_type,
            node_cluster_size=node_cluster_size_expanded,
            node_attr=t_expanded_attr[batch[node_type==1]] * node_end + (1-t_expanded_attr[batch[node_type==1]]) * node_start,
            edge_node_attr=t_expanded_attr[batch[node_type==0]] * edge_node_end + (1-t_expanded_attr[batch[node_type==0]]) * edge_node_start,
            edge_attr=(t[edge_batch] * edge_attr + (1-(1-self.sigma_min)*t[edge_batch]) * edge_start).unsqueeze(1),
            node_features=t_expanded * node_features_full + (1-t_expanded) * node_features_start if node_features_expanded is not None else None,
            t=t,
            model=model,
            model_kwargs=model_kwargs,
        )
        
        # Compute loss ignoring node_loss for first level
        node_attr_pred = node_attr_pred.float().squeeze(1)
        edge_node_attr_pred = edge_node_attr_pred.float().squeeze(1)
        edge_attr_pred = edge_attr_pred.float().squeeze(1)
        
        node_loss = (node_attr_pred - (node_end - node_start)) ** 2
        edge_node_loss = (edge_node_attr_pred - (edge_node_end - edge_node_start)) ** 2
        edge_loss = (edge_attr_pred - (edge_attr - (1-self.sigma_min)*edge_start)) ** 2
        
        nodes_node_features_loss = None
        edge_nodes_node_features_loss = None
        
        if node_features_expanded is not None:
            nodes_node_features_loss = (node_features_nodes_pred - (node_features_full[node_type == 1] - node_features_start[node_type == 1]))**2
            edge_nodes_node_features_loss = (node_features_edge_nodes_pred - (node_features_full[node_type == 0] - node_features_start[node_type == 0]))**2
            
        """node_pred, edge_node_pred, augmented_edge_pred, node_features_pred, hyperedge_features_pred = self.sample(edge_index, node_type, node_features_expanded, None, batch, model, model_kwargs)
        
        diff = node_features_pred - node_features_real

        # Flatten the diff tensor if it's multi-dimensional
        flat_diff = diff.flatten()
        
        idx = th.argmax(flat_diff**2)
        
        print(diff)
        print(f"real : {node_features_real.flatten()[idx]}")
        print(f"pred : {node_features_pred.flatten()[idx]}")"""
        
        return node_loss, edge_node_loss, edge_loss, nodes_node_features_loss, edge_nodes_node_features_loss, None


    @th.no_grad()
    def sample(self, edge_index, node_type, node_features, hyperedge_features, node_cluster_size, batch, model, model_kwargs):
        """Sampling procedure for CFM."""
        num_graphs = batch.max().item() + 1
        
        # Start from random initialization
        node_attr_next = th.randn_like(batch[node_type == 1], dtype=th.float32)
        edge_attr_next = self.edge_randn(edge_index, dtype=th.float32)[:, None]

        node_attr_next = th.cat((node_attr_next[:, None], node_cluster_size), dim=1)
        edge_node_attr_next = th.randn((batch[node_type == 0].shape[0], 3), dtype=th.float32, device=self.device)
        
        node_features_next = None
        if node_features is not None:
            node_features_next = th.zeros((node_type.shape[0],) + node_features.shape[1:], device=self.device, dtype=th.float32)
            node_features_next[node_type == 1] = node_features
            node_features_next[node_type == 0] = th.randn_like(node_features_next[node_type == 0])

        # Flow matching sampling: Integrate learned flow model over time (0 to 1)
        t_steps = th.linspace(0, 1, steps=self.num_steps, device=self.device)

        for i in range(len(t_steps[:-1])):
            # Forward pass through the model to get flow direction
            node_attr_pred, edge_node_attr_pred, edge_attr_pred, node_features_nodes_pred, node_features_edge_nodes_pred = self.model_wrapper(
                edge_index=edge_index,
                batch=batch,
                node_type=node_type,
                node_cluster_size=node_cluster_size,
                node_attr=node_attr_next,
                edge_node_attr=edge_node_attr_next,
                edge_attr=edge_attr_next,
                node_features=node_features_next,
                t=t_steps[i].repeat(num_graphs),
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

        return node_attr_next, edge_node_attr_next[:, 0], edge_attr_next.squeeze(1), node_features_next[node_type == 1] if node_features_next is not None else None, None


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
    def minibatch_ot_coupling(noise, target):
        """Reorder noise samples for OT coupling."""
        # Broadcast
        if noise.dim() == 1:
            noise = noise.view(-1, 1)
        if target.dim() == 1:
            target = target.view(-1, 1)
        
        with th.no_grad():
            # Calculate the distance matrix between each pair of points in noise and target
            cost_matrix = th.cdist(noise, target)**2
            
            # Compute transport plan
            ones = th.ones(noise.shape[0], device=noise.device)
            pi = emd(ones, ones, cost_matrix.detach())
          
            p = pi.flatten()
            p = p / p.sum()
            
            # Sample from the OT coupling
            choices = th.multinomial(p, num_samples=pi.shape[0], replacement=False)
            
        return th.div(choices, pi.shape[1], rounding_mode='floor'), choices % pi.shape[1]
        

    @staticmethod
    def compute_edge_node_averages(
        node_type: th.Tensor,
        edge_index: th.Tensor,
        attributes: th.Tensor,
        device: str = 'cuda'
    ) -> th.Tensor:
        """
        Compute average edge node attributes based on connected nodes.
        
        Args:
            node_type: Tensor indicating type of each node (0 for hyperedges, 1 for regular nodes)
            edge_index: Tensor of shape [2, num_edges] containing source and destination node indices
            attribute: Tensor containing attributes for regular nodes
            device: Device to place tensors on ('cuda' or 'cpu')
        
        Returns:
            th.Tensor: Average attributes for hyperedge nodes
        """
        # Initialize full cluster size tensor for all nodes
        attributes_full = th.zeros((node_type.shape[0], attributes.shape[1]), device=device)
        attributes_full[node_type == 1] = attributes
        
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
        return neighbor_avg[node_type == 0]