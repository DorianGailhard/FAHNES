import torch as th
from torch.nn import Dropout, Linear, Module, ModuleList
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.utils import coalesce

from .mlp import MLP


class GINE(Module):
    """Graph Isomorphism Network (GIN) model with edge features.

    Operates on a sparse graph representation.
    """

    def __init__(
        self,
        node_features_in_dim,
        node_features_out_dim,
        node_features_emb_dim,
        hyperedge_features_in_dim,
        hyperedge_features_out_dim,
        hyperedge_features_emb_dim,
        expansion_node_in_dim: int,
        expansion_hyperedge_in_dim: int,
        expansion_edge_in_dim: int,
        expansion_node_out_dim: int,
        expansion_hyperedge_out_dim: int,
        expansion_edge_out_dim: int,
        expansion_emb_dim: int,
        hidden_dim: int,
        ppgn_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        self_conditioning: bool = False
    ):
        super().__init__()

        # Feature type
        if node_features_in_dim != 0 and isinstance(node_features_in_dim, int):
            self.node_dim = 0 # coding for vector feature
            self.node_features_dim = node_features_in_dim
        else:
            self.node_dim = -1 # coding for no features
            self.node_features_dim = 0
            node_emb_dim = 0
            
        if hyperedge_features_in_dim != 0 and isinstance(hyperedge_features_in_dim, int):
            self.hyperedge_dim = 0 # coding for vector feature
            self.hyperedge_features_dim = hyperedge_features_in_dim
        else:
            self.hyperedge_dim = -1 # coding for no features
            self.hyperedge_features_dim = 0
            hyperedge_emb_dim = 0

        # Embedding layers
        self.node_emb_layer = Linear((expansion_node_in_dim + self.node_features_dim)*(1+self_conditioning) + self.node_features_dim, expansion_emb_dim + node_emb_dim)
        self.hyperedge_emb_layer = Linear((expansion_hyperedge_in_dim + self.hyperedge_features_dim)*(1+self_conditioning) + self.hyperedge_features_dim, expansion_emb_dim + hyperedge_emb_dim)
        self.edge_emb_layer = Linear(expansion_edge_in_dim*(1+self_conditioning), expansion_emb_dim + node_emb_dim + hyperedge_emb_dim)
        self.noise_cond_emb_layer = Linear(1, expansion_emb_dim)
        self.red_frac_emb_layer = Linear(1, expansion_emb_dim)
        self.node_budget_emb_layer = PositionalEncoding(expansion_emb_dim, base_freq=1e-4)

        # In layers
        self.node_in_mlp = MLP(5 * expansion_emb_dim + node_emb_dim, [hidden_dim, hidden_dim])
        self.hyperedge_in_mlp = MLP(4 * expansion_emb_dim + hyperedge_emb_dim, [hidden_dim, hidden_dim])
        self.edge_in_mlp = MLP(5 * expansion_emb_dim + node_emb_dim + hyperedge_emb_dim, [hidden_dim, hidden_dim])

        # GNN layers
        self.gine_layers = ModuleList(
            [
                GINEConv(MLP(hidden_dim, [hidden_dim, hidden_dim]))
                for _ in range(num_layers)
            ]
        )
        self.edge_layers = ModuleList(
            [
                MLP(3 * hidden_dim, [hidden_dim, hidden_dim])
                for _ in range(num_layers)
            ]
        )

        # Out layers        
        self.node_out_layer = Linear((num_layers + 1) * hidden_dim, expansion_node_out_dim + node_features_out_dim)
        self.hyperedge_out_layer = Linear((num_layers + 1) * hidden_dim, expansion_hyperedge_out_dim + hyperedge_features_out_dim)
        self.edge_out_layer = Linear((num_layers + 1) * hidden_dim, expansion_edge_out_dim)

        # Dropout
        self.dropout = Dropout(dropout)

    def forward(
        self,
        edge_index,
        batch,
        node_type,
        node_cluster_size,
        node_attr,
        hyperedge_attr,
        edge_attr,
        node_features_initial,
        node_dim,
        hyperedge_features_initial,
        hyperedge_dim,
        node_emb,
        hyperedge_emb,
        noise_cond,
        red_frac,
        target_size,
    ):
        # Embedding
        edge_attr_emb = self.edge_emb_layer(edge_attr)
        noise_cond_emb = self.noise_cond_emb_layer(noise_cond[..., None])
        red_frac_emb = self.red_frac_emb_layer(red_frac[..., None])
        node_budget_emb = self.node_budget_emb_layer(node_cluster_size[..., None])
        
        if self.node_dim == 0: # vector features
            node_attr_emb = self.node_emb_layer(th.cat((node_attr, node_dim, node_features_initial), dim=-1))
        else:
            node_attr_emb = self.node_emb_layer(node_attr)
        
        if self.hyperedge_dim == 0: # vector features
            hyperedge_attr_emb = self.hyperedge_emb_layer(th.cat((hyperedge_attr, hyperedge_dim, hyperedge_features_initial), dim=-1))
        else:
            hyperedge_attr_emb = self.hyperedge_emb_layer(hyperedge_attr)
        
        # Input
        # Nodes
        x_node = [
            node_attr_emb,
            node_emb,
            noise_cond_emb[batch[node_type == 1]],
            red_frac_emb[batch[node_type == 1]],
            node_budget_emb,
        ]
        
        x_node = th.cat(x_node, dim=-1)
            
        x_node = self.dropout(x_node)
        x_node = self.node_in_mlp(x_node)
        
        # Edge nodes
        x_hyperedge = [
            hyperedge_attr_emb,
            hyperedge_emb,
            noise_cond_emb[batch[node_type == 0]],
            red_frac_emb[batch[node_type == 0]],
        ]
        
        x_hyperedge = th.cat(x_hyperedge, dim=-1)
        
        x_hyperedge = self.dropout(x_hyperedge)
        x_hyperedge = self.hyperedge_in_mlp(x_hyperedge)
        
        # Combine them
        x_all_nodes = th.zeros(x_node.size(0) + x_hyperedge.size(0), x_node.size(1), device=x_node.device)
        x_all_nodes[node_type == 0] = x_hyperedge
        x_all_nodes[node_type == 1] = x_node
        
        # Edges
        all_nodes_emb = th.zeros(node_emb.size(0) + hyperedge_emb.size(0), node_emb.size(1), device=node_emb.device)
        all_nodes_emb[node_type == 0] = hyperedge_emb
        all_nodes_emb[node_type == 1] = node_emb
        
        sorted_edge_min = th.minimum(edge_index[0], edge_index[1])
        sorted_edge_max = th.maximum(edge_index[0], edge_index[1])
        
        x_edge = [
            edge_attr_emb,
            all_nodes_emb[sorted_edge_min],
            all_nodes_emb[sorted_edge_max],
            noise_cond_emb[batch[sorted_edge_min]],
            red_frac_emb[batch[sorted_edge_min]],
        ]
        
        x_edge = th.cat(x_edge, dim=-1)
                
        x_edge = self.dropout(x_edge)
        x_edge = self.edge_in_mlp(x_edge)

        skip_node = [x_all_nodes]
        skip_edge = [x_edge]
        for gin_layer, edge_layer in zip(self.gine_layers, self.edge_layers):
            x_all_nodes = gin_layer(x=x_all_nodes, edge_index=edge_index, edge_attr=x_edge)
            skip_node.append(x_all_nodes)
            x_edge = edge_layer(
                th.cat([x_edge, x_all_nodes[edge_index[0]], x_all_nodes[edge_index[1]]], dim=-1)
            )
            skip_edge.append(x_edge)

        # Skip layer
        x_all_nodes = th.cat(skip_node, dim=-1)
        x_all_nodes = self.dropout(x_all_nodes)
        x_edge = th.cat(skip_edge, dim=-1)
        x_edge = self.dropout(x_edge)

        # Out layers
        out_node = self.node_out_layer(x_all_nodes[node_type == 1])
        out_hyperedge = self.hyperedge_out_layer(x_all_nodes[node_type == 0])
        out_edge = self.edge_out_layer(x_edge)

        # make out_edge symmetric
        out_edge = coalesce(
            th.cat([edge_index, edge_index.flip(0)], dim=-1),
            th.cat([out_edge, out_edge], dim=0),
            reduce="mean",
        )[1]

        return out_node[:, self.node_features_dim:], out_hyperedge[:, self.hyperedge_features_dim:], out_edge, out_node[:, :self.node_features_dim], out_hyperedge[:, :self.hyperedge_features_dim]