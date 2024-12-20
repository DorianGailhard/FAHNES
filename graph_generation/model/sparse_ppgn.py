import torch as th
from torch.nn import Dropout, Linear, Module, ModuleList
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import coalesce
from torch_scatter import scatter

from .mlp import MLP


class SparsePPGN(Module):
    """Our proposed SparsePPGN model.

    Operates on a sparse graph representation.
    """

    def __init__(
        self,
        node_in_features,
        node_out_features,
        node_emb_features,
        hyperedge_in_features,
        hyperedge_out_features,
        hyperedge_emb_features,
        expansion_node_in_features: int,
        expansion_edge_node_in_features: int,
        expansion_edge_in_features: int,
        expansion_node_out_features: int,
        expansion_edge_out_features: int,
        expansion_emb_features: int,
        hidden_features: int,
        ppgn_features: int,
        num_layers: int,
        dropout: float = 0.0,
        self_conditioning: bool = False
    ):
        super().__init__()
        
        # Feature type
        if node_in_features != 0 and isinstance(node_in_features, int):
            self.node_features = 0 # coding for vector feature
            self.node_features_dim = node_in_features
        else:
            self.node_features = -1 # coding for no features
            self.node_features_dim = 0
            node_emb_features = 0

        # Embedding layers
        self.node_emb_layer = Linear(expansion_node_in_features*(1+self_conditioning), expansion_emb_features)
        self.edge_node_emb_layer = Linear(expansion_edge_node_in_features*(1+self_conditioning), expansion_emb_features)
        self.edge_emb_layer = Linear(expansion_edge_in_features*(1+self_conditioning), expansion_emb_features)
        self.noise_cond_emb_layer = Linear(1, expansion_emb_features)
        self.red_frac_emb_layer = Linear(1, expansion_emb_features)
        self.target_size_emb_layer = PositionalEncoding(expansion_emb_features)

        # In layers
        self.node_in_mlp = MLP(5 * expansion_emb_features + node_emb_features, [hidden_features, hidden_features])
        self.edge_node_in_mlp = MLP(5 * expansion_emb_features + node_emb_features, [hidden_features, hidden_features])
        self.edge_in_mlp = MLP(6 * expansion_emb_features + 2*node_emb_features, [hidden_features, hidden_features])

        # GNN layers
        self.sparse_ppgn_layers = ModuleList(
            [SparsePPGNLayer(hidden_features, ppgn_features) for _ in range(num_layers)]
        )
        
        # Out layers
        self.node_out_layer = Linear(
            (num_layers + 1) * hidden_features, expansion_node_out_features + node_out_features + hyperedge_out_features
        )
        self.edge_node_out_layer = Linear(
            (num_layers + 1) * hidden_features, expansion_node_out_features + node_out_features + hyperedge_out_features
        )
        self.edge_out_layer = Linear(
            (num_layers + 1) * hidden_features, expansion_edge_out_features
        )

        # Dropout
        self.dropout = Dropout(dropout)
        
        # Features
        if self.node_features == 0: # Vector features
            # Embedding layers
            self.features_node_emb_layer = Linear(node_in_features*(1+self_conditioning), node_emb_features)
        

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
        node_emb,
        edge_node_emb,
        noise_cond,
        red_frac,
        target_size,
    ):
        # Embedding
        node_attr_emb = self.node_emb_layer(node_attr)#th.cat((node_cluster_size, node_attr), dim=1))
        edge_node_attr_emb = self.edge_node_emb_layer(edge_node_attr)
        edge_attr_emb = self.edge_emb_layer(edge_attr)
        noise_cond_emb = self.noise_cond_emb_layer(noise_cond[..., None])
        red_frac_emb = self.red_frac_emb_layer(red_frac[..., None])
        target_size_emb = self.target_size_emb_layer(target_size[..., None])
        
        if self.node_features == 0: # vector features
            node_features_emb = self.features_node_emb_layer(node_features)
        
        # Input
        # Node features
        if self.node_features == -1: # no features
            x_node = th.cat(
                [
                    node_attr_emb,
                    node_emb,
                    noise_cond_emb[batch[node_type == 1]],
                    red_frac_emb[batch[node_type == 1]],
                    target_size_emb[batch[node_type == 1]],
                ],
                dim=-1,
            )
        elif self.node_features == 0: # vector features
            x_node = th.cat(
                [
                    node_attr_emb,
                    node_emb,
                    noise_cond_emb[batch[node_type == 1]],
                    red_frac_emb[batch[node_type == 1]],
                    target_size_emb[batch[node_type == 1]],
                    node_features_emb[node_type == 1]
                ],
                dim=-1,
            )
            
        x_node = self.dropout(x_node)
        x_node = self.node_in_mlp(x_node)
        
        
        if self.node_features == -1: # no features
            x_edge_node = th.cat(
                [
                    edge_node_attr_emb,
                    edge_node_emb,
                    noise_cond_emb[batch[node_type == 0]],
                    red_frac_emb[batch[node_type == 0]],
                    target_size_emb[batch[node_type == 0]],
                ],
                dim=-1,
            )
        elif self.node_features == 0: # vector features
            x_edge_node = th.cat(
                [
                    edge_node_attr_emb,
                    edge_node_emb,
                    noise_cond_emb[batch[node_type == 0]],
                    red_frac_emb[batch[node_type == 0]],
                    target_size_emb[batch[node_type == 0]],
                    node_features_emb[node_type == 0]
                ],
                dim=-1,
            )
        
        x_edge_node = self.dropout(x_edge_node)
        x_edge_node = self.edge_node_in_mlp(x_edge_node)
        
        
        edge_batch = batch[edge_index[0]]
        
        all_nodes_emb = th.zeros(node_emb.size(0) + edge_node_emb.size(0), node_emb.size(1), device=node_emb.device)
        all_nodes_emb[node_type == 0] = edge_node_emb
        all_nodes_emb[node_type == 1] = node_emb
        
        if self.node_features == -1: # no features
            x_edge = th.cat(
                [
                    edge_attr_emb,
                    all_nodes_emb[th.minimum(edge_index[0], edge_index[1])], # edge index is sorted so left side nodes always come first
                    all_nodes_emb[th.maximum(edge_index[0], edge_index[1])],
                    noise_cond_emb[edge_batch],
                    red_frac_emb[edge_batch],
                    target_size_emb[edge_batch],
                ],
                dim=-1,
            )
        else:
            x_edge = th.cat(
                [
                    edge_attr_emb,
                    all_nodes_emb[th.minimum(edge_index[0], edge_index[1])],
                    all_nodes_emb[th.maximum(edge_index[0], edge_index[1])],
                    noise_cond_emb[edge_batch],
                    red_frac_emb[edge_batch],
                    target_size_emb[edge_batch],
                    node_features_emb[edge_index[0]],
                    node_features_emb[edge_index[1]],
                ],
                dim=-1,
            )
        
        x_edge = self.dropout(x_edge)
        x_edge = self.edge_in_mlp(x_edge)            

        # construct triangle_index
        # the indexed elements are the edges of the triangles (including self-loops)
        # for each triangle (a, b, c) the message x[a] * x[b] is sent to x[c]
        
        # Add self-loops
        self_loop_index = th.arange(node_attr.size(0) + edge_node_attr.size(0), device=node_attr.device)[None, :].expand(2, -1)
        edge_index_ext = th.cat([self_loop_index, edge_index], dim=1)
        
        x = th.zeros(x_node.size(0) + x_edge_node.size(0), x_node.size(1), device=x_node.device)
        x[node_type == 0] = x_edge_node
        x[node_type == 1] = x_node
        x = th.cat([x, x_edge], dim=0)
        
        # Total number of nodes + edge-nodes
        n = node_attr.size(0) + edge_node_attr.size(0)
        
        # Map edges to unique edge IDs
        edge_id = edge_index_ext[0] * n + edge_index_ext[1]
        edge_id_to_edge_num = th.full((n * n,), -1, dtype=th.long, device=x.device)
        edge_id_to_edge_num[edge_id] = th.arange(edge_id.size(0), device=x.device)
        
        # For a bipartite graph, each triangle is of the form :
        # (a, b) -> (b, b) -> (b, a)
        # (a, b) -> (b, a) -> (a, a)
        # (a, a) -> (a, b) -> (b, a)
        # (a, a) -> (a, a) -> (a, a)
        triangle_index_type_1 = th.stack(
            [
                edge_id_to_edge_num[edge_index[0] * n + edge_index[1]],  # Edge (a, b)
                edge_id_to_edge_num[edge_index[1] * n + edge_index[1]],  # Edge (b, b)
                edge_id_to_edge_num[edge_index[1] * n + edge_index[0]]   # Edge (b, a)
            ]
        )
        
        triangle_index_type_2 = th.stack(
            [
                edge_id_to_edge_num[edge_index[0] * n + edge_index[1]],  # Edge (a, b)
                edge_id_to_edge_num[edge_index[1] * n + edge_index[0]],  # Edge (b, a)
                edge_id_to_edge_num[edge_index[0] * n + edge_index[0]]   # Edge (a, a)
            ]
        )
        
        triangle_index_type_3 = th.stack(
            [
                edge_id_to_edge_num[self_loop_index[0] * n + self_loop_index[0]],  # Edge (a, a)
                edge_id_to_edge_num[self_loop_index[0] * n + self_loop_index[1]],  # Edge (a, b)
                edge_id_to_edge_num[self_loop_index[1] * n + self_loop_index[0]]   # Edge (b, a)
            ]
        )
        
        triangle_index_type_4 = th.stack(
            [
                edge_id_to_edge_num[self_loop_index[0] * n + self_loop_index[0]],  # Edge (a, a)
                edge_id_to_edge_num[self_loop_index[0] * n + self_loop_index[0]],  # Edge (a, a)
                edge_id_to_edge_num[self_loop_index[0] * n + self_loop_index[0]]   # Edge (a, a)
            ]
        )
        
        triangle_index = th.cat(
            [
                triangle_index_type_1,
                triangle_index_type_2,
                triangle_index_type_3,
                triangle_index_type_4
            ], dim = 1
        )


        # Layers
        num_messages = scatter(
            th.ones(triangle_index.size(1), device=x.device), triangle_index[2], dim=0
        )
        norm_factor = 1.0 / num_messages.sqrt()

        skip = [x]
        for layer in self.sparse_ppgn_layers:
            x = layer(x, triangle_index, norm_factor)
            skip.append(x)

        # Skip layer
        x = th.cat(skip, dim=-1)
        x = self.dropout(x)

        # Out layers
        out_node = self.node_out_layer(x[:n][node_type == 1])
        out_edge_node = self.edge_node_out_layer(x[:n][node_type == 0])
        out_edge = self.edge_out_layer(x[n:])

        # make out_edge symmetric
        out_edge = coalesce(
            th.cat([edge_index, edge_index.flip(0)], dim=-1),
            th.cat([out_edge, out_edge], dim=0),
            reduce="mean",
        )[1]

        return out_node[:, self.node_features_dim:], out_edge_node[:, self.node_features_dim:], out_edge, out_node[:, :self.node_features_dim], out_edge_node[:, :self.node_features_dim]


class SparsePPGNLayer(Module):
    def __init__(self, hidden_features, ppgn_features):
        super().__init__()

        self.mlp1 = MLP(
            in_features=hidden_features,
            hidden_features=[hidden_features, ppgn_features],
        )
        self.mlp2 = MLP(
            in_features=hidden_features,
            hidden_features=[hidden_features, ppgn_features],
        )
        self.mlp3 = MLP(
            in_features=hidden_features + ppgn_features,
            hidden_features=[hidden_features, hidden_features],
        )

    def forward(self, x, triangle_index, norm_factor):
        m1 = self.mlp1(x)
        m2 = self.mlp2(x)

        m = scatter(
            m1[triangle_index[0]] * m2[triangle_index[1]], triangle_index[2], dim=0
        )
        m = m * norm_factor[:, None]

        x = self.mlp3(th.cat([x, m], dim=-1))
        return x