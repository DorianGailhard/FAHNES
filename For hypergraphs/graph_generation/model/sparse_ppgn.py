import torch as th
from torch.nn import Dropout, Linear, Module, ModuleList, LayerNorm
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.utils import coalesce
from torch_scatter import scatter

from .mlp import MLP


class SparsePPGN(Module):
    """Our proposed SparsePPGN model.

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
        expansion_incidence_in_dim: int,
        expansion_node_out_dim: int,
        expansion_hyperedge_out_dim: int,
        expansion_incidence_out_dim: int,
        expansion_emb_dim: int,
        hidden_dim: int,
        ppgn_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        self_conditioning: bool = False
    ):
        super().__init__()
        
        # Feature type
        if node_features_in_dim != 0:
            self.node_dim = 0 # coding for vector feature
            self.node_features_dim = node_features_in_dim
        else:
            self.node_dim = -1 # coding for no features
            self.node_features_dim = 0
            node_features_emb_dim = 0

        if hyperedge_features_in_dim != 0:
            self.hyperedge_dim = 0 # coding for vector feature
            self.hyperedge_features_dim = hyperedge_features_in_dim
        else:
            self.hyperedge_dim = -1 # coding for no features
            self.hyperedge_features_dim = 0
            hyperedge_features_emb_dim = 0

        # Embedding layers
        self.node_emb_layer = Linear(expansion_node_in_dim*(1+self_conditioning), expansion_emb_dim)
        self.hyperedge_emb_layer = Linear(expansion_hyperedge_in_dim*(1+self_conditioning), expansion_emb_dim)
        self.incidence_emb_layer = Linear(expansion_incidence_in_dim*(1+self_conditioning), expansion_emb_dim)
        self.noise_cond_emb_layer = Linear(1, expansion_emb_dim)
        self.red_frac_emb_layer = Linear(1, expansion_emb_dim)
        self.node_budget_emb_layer = PositionalEncoding(expansion_emb_dim, base_freq=1e-4)
        
        if self.node_features_dim > 0:
            self.node_features_emb_layer = Linear(self.node_features_dim*(1+self_conditioning), node_features_emb_dim)
            self.node_features_film_proj = Linear(self.node_features_dim, 2 * node_features_emb_dim)
            self.node_features_norm = LayerNorm(node_features_emb_dim)
        
        if self.hyperedge_features_dim > 0:
            self.hyperedge_features_emb_layer = Linear(self.hyperedge_features_dim*(1+self_conditioning), hyperedge_features_emb_dim)
            self.hyperedge_features_film_proj = Linear(self.hyperedge_features_dim, 2 * hyperedge_features_emb_dim)
            self.hyperedge_features_norm = LayerNorm(hyperedge_features_emb_dim)

        # In layers
        self.node_in_mlp = MLP(5 * expansion_emb_dim + node_features_emb_dim, [hidden_dim, hidden_dim])
        self.hyperedge_in_mlp = MLP(4 * expansion_emb_dim + hyperedge_features_emb_dim, [hidden_dim, hidden_dim])
        self.incidence_in_mlp = MLP(5 * expansion_emb_dim, [hidden_dim, hidden_dim])

        # GNN layers
        self.sparse_ppgn_layers = ModuleList(
            [SparsePPGNLayer(hidden_dim, ppgn_dim) for _ in range(num_layers)]
        )
        
        # Out layers        
        self.node_out_layer = Linear((num_layers + 1) * hidden_dim, expansion_node_out_dim + node_features_out_dim)
        self.hyperedge_out_layer = Linear((num_layers + 1) * hidden_dim, expansion_hyperedge_out_dim + hyperedge_features_out_dim)
        self.incidence_out_layer = Linear((num_layers + 1) * hidden_dim, expansion_incidence_out_dim)

        # Dropout
        self.dropout = Dropout(dropout)

    def forward(
        self,
        incidence_index,
        batch,
        node_type,
        budget,
        node_attr,
        hyperedge_attr,
        incidence_attr,
        node_features_initial,
        node_features,
        hyperedge_features_initial,
        hyperedge_features,
        node_emb,
        hyperedge_emb,
        noise_cond,
        red_frac,
        target_size,
    ):
        # Embedding
        node_attr_emb = self.node_emb_layer(node_attr)
        hyperedge_attr_emb = self.hyperedge_emb_layer(hyperedge_attr)
        incidence_attr_emb = self.incidence_emb_layer(incidence_attr)
        noise_cond_emb = self.noise_cond_emb_layer(noise_cond[..., None])
        red_frac_emb = self.red_frac_emb_layer(red_frac[..., None])
        node_budget_emb = self.node_budget_emb_layer(budget[..., None])
        
        if self.node_dim == 0: # vector features
            node_features_emb = self.node_features_emb_layer(node_features)
            node_features_emb = self.node_features_norm(node_features_emb)

            # Generate FiLM scale and shift from condition
            gamma_beta = self.node_features_film_proj(node_features_initial)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            
            node_features_emb = gamma * node_features_emb + beta
        
        
        if self.hyperedge_dim == 0: # vector features
            hyperedge_features_emb = self.hyperedge_features_emb_layer(hyperedge_features)
            hyperedge_features_emb = self.hyperedge_features_norm(hyperedge_features_emb)

            # Generate FiLM scale and shift from condition
            gamma_beta = self.hyperedge_features_film_proj(hyperedge_features_initial)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            
            hyperedge_features_emb = gamma * hyperedge_features_emb + beta
        
        
        # Input
        # Nodes
        x_node = [
            node_attr_emb,
            node_emb,
            noise_cond_emb[batch[node_type == 1]],
            red_frac_emb[batch[node_type == 1]],
            node_budget_emb,
        ]
        
        if self.node_dim == 0: # vector features
            x_node.append(node_features_emb)
 
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
        
        if self.hyperedge_dim == 0: # vector features
            x_hyperedge.append(hyperedge_features_emb)
        
        x_hyperedge = th.cat(x_hyperedge, dim=-1)
        
        x_hyperedge = self.dropout(x_hyperedge)
        x_hyperedge = self.hyperedge_in_mlp(x_hyperedge)
        
        # incidences
        sorted_incidence_min = th.minimum(incidence_index[0], incidence_index[1])
        sorted_incidence_max = th.maximum(incidence_index[0], incidence_index[1])
        
        node_indices = th.zeros_like(batch)
        node_indices[node_type == 1] = th.arange(node_attr_emb.size(0), device=node_attr_emb.device)
        node_indices[node_type == 0] = th.arange(hyperedge_attr_emb.size(0), device=hyperedge_attr_emb.device)
        
        x_incidence = [
            incidence_attr_emb,
            node_emb[node_indices[sorted_incidence_min]],
            hyperedge_emb[node_indices[sorted_incidence_max]],
            noise_cond_emb[batch[sorted_incidence_min]],
            red_frac_emb[batch[sorted_incidence_min]]
        ]
        
        x_incidence = th.cat(x_incidence, dim=-1)
                
        x_incidence = self.dropout(x_incidence)
        x_incidence = self.incidence_in_mlp(x_incidence)            

        # construct triangle_index
        # the indexed elements are the edges of the triangles (including self-loops)
        # for each triangle (a, b, c) the message x[a] * x[b] is sent to x[c]
        
        # Add self-loops
        self_loop_index = th.arange(node_attr.size(0) + hyperedge_attr.size(0), device=node_attr.device)[None, :].expand(2, -1)
        edge_index_ext = th.cat([self_loop_index, incidence_index], dim=1)
        
        x = th.zeros(x_node.size(0) + x_hyperedge.size(0), x_node.size(1), device=x_node.device)
        x[node_type == 0] = x_hyperedge
        x[node_type == 1] = x_node
        x = th.cat([x, x_incidence], dim=0)
        
        # Total number of nodes + edge-nodes
        n = node_attr.size(0) + hyperedge_attr.size(0)
        
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
                edge_id_to_edge_num[incidence_index[0] * n + incidence_index[1]],  # Edge (a, b)
                edge_id_to_edge_num[incidence_index[1] * n + incidence_index[1]],  # Edge (b, b)
                edge_id_to_edge_num[incidence_index[1] * n + incidence_index[0]]   # Edge (b, a)
            ]
        )
        
        triangle_index_type_2 = th.stack(
            [
                edge_id_to_edge_num[incidence_index[0] * n + incidence_index[1]],  # Edge (a, b)
                edge_id_to_edge_num[incidence_index[1] * n + incidence_index[0]],  # Edge (b, a)
                edge_id_to_edge_num[incidence_index[0] * n + incidence_index[0]]   # Edge (a, a)
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
        out_hyperedge = self.hyperedge_out_layer(x[:n][node_type == 0])
        out_incidence = self.incidence_out_layer(x[n:])
        
        # make out_incidence symmetric
        out_incidence = coalesce(
            th.cat([incidence_index, incidence_index.flip(0)], dim=-1),
            th.cat([out_incidence, out_incidence], dim=0),
            reduce="mean",
        )[1]

        return out_node[:, self.node_features_dim:], out_hyperedge[:, self.hyperedge_features_dim:], out_incidence, out_node[:, :self.node_features_dim], out_hyperedge[:, :self.hyperedge_features_dim]


class SparsePPGNLayer(Module):
    def __init__(self, hidden_dim, ppgn_dim):
        super().__init__()

        self.mlp1 = MLP(
            in_dim=hidden_dim,
            hidden_dim=[hidden_dim, ppgn_dim],
        )
        self.mlp2 = MLP(
            in_dim=hidden_dim,
            hidden_dim=[hidden_dim, ppgn_dim],
        )
        self.mlp3 = MLP(
            in_dim=hidden_dim + ppgn_dim,
            hidden_dim=[hidden_dim, hidden_dim],
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