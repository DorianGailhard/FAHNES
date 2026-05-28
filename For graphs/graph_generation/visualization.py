import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw

def plot_graph(pred_graphs):
    """Visualize predicted graphs using spring layout."""
    n = int(min(5, np.ceil(np.sqrt(len(pred_graphs)))))
    fig, axs = plt.subplots(n, min(n, np.ceil(len(pred_graphs)/n)), figsize=(50, 50))

    ## Draw each graph
    for i in range(min(n * n, len(pred_graphs))):
        G = pred_graphs[i]
        ax = axs[i // n, i % n]

        pos = nx.spring_layout(G, seed=0)

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=200, node_color='skyblue',
            edgecolors='black', linewidths=1, alpha=1
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=2, edge_color='salmon', alpha=0.7
        )

        ax.title.set_text(f"N = {len(G)}")
        ax.title.set_fontsize(40)
        ax.axis('off')

    fig.tight_layout()
    return fig


def plot_point_cloud(pred_graphs):
    """Visualize predicted graphs as 3D point clouds."""
    n = int(min(5, np.ceil(np.sqrt(len(pred_graphs)))))
    fig = plt.figure(figsize=(50, 50))
    axs = []

    for i in range(min(n, int(np.ceil(len(pred_graphs)/n)))):
        for j in range(n):
            index = i * n + j
            if index >= len(pred_graphs):
                break
            ax = fig.add_subplot(min(n, int(np.ceil(len(pred_graphs)/n))), n, index + 1, projection='3d')
            axs.append(ax)

    for i in range(min(n * n, len(pred_graphs))):
        ax = axs[i]
        G = pred_graphs[i]

        positions = {node: G.nodes[node]["feature"] for node in G.nodes}
        pos_array = np.array(list(positions.values()))

        xs, ys, zs = pos_array[:,0], pos_array[:,1], pos_array[:,2]
        colors = (zs - zs.min()) / (zs.max() - zs.min() + 1e-8)
        ax.scatter(xs, ys, zs, s=20, c=colors, cmap='viridis', alpha=0.8)

        pos_min = np.min(pos_array)
        pos_max = np.max(pos_array)
        ax.set_xlim(pos_min, pos_max)
        ax.set_ylim(pos_min, pos_max)
        ax.set_zlim(pos_min, pos_max)

        ax.title.set_text(f"N = {len(G.nodes)}")
        ax.title.set_fontsize(40)
        ax.axis('off')

    fig.tight_layout()
    return fig


def plot_molecule(pred_graphs):
    """Visualize predicted graphs as molecular structures."""
    n = int(min(5, np.ceil(np.sqrt(len(pred_graphs)))))
    fig, axs = plt.subplots(n, min(n, np.ceil(len(pred_graphs)/n)), figsize=(50, 50))

    allowed_atom_types = [1, 6, 7, 8, 9]  # H, C, N, O, F
    atom_types_decode = {idx: atomic_num for idx, atomic_num in enumerate(allowed_atom_types)}

    for i in range(min(n * n, len(pred_graphs))):
        ax = axs[i // n, i % n]
        G = pred_graphs[i]

        mol = Chem.RWMol()
        node_to_atom = {}

        for node in G.nodes:
            one_hot = G.nodes[node]["feature"]
            atom_type = atom_types_decode[np.argmax(one_hot)]
            if atom_type is not None:
                atom = Chem.Atom(atom_type)
                atom.SetNoImplicit(True)
                atom_idx = mol.AddAtom(atom)
                node_to_atom[node] = atom_idx

        for u, v in G.edges:
            if mol.GetBondBetweenAtoms(node_to_atom[u], node_to_atom[v]) is None:
                mol.AddBond(node_to_atom[u], node_to_atom[v], Chem.BondType.UNSPECIFIED)

        mol = mol.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Sanitization failed: {e}")

        img = Draw.MolToImage(mol, size=(500, 500))
        ax.imshow(img)
        
        ax.title.set_text(f"N = {len(G.nodes)}")
        ax.title.set_fontsize(40)
        ax.axis('off')

    fig.tight_layout()
    return fig
