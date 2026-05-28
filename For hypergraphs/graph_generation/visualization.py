import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from rdkit import Chem
from rdkit.Chem import Draw


def plot_hypergraph(pred_hypergraphs):
    """Visualize predicted hypergraphs as 2D convex hull expansions."""
    n = int(min(5, np.ceil(np.sqrt(len(pred_hypergraphs)))))
    fig, axs = plt.subplots(min(n, int(np.ceil(len(pred_hypergraphs)/n))), n, figsize=(50, 50))

    node_size = 200
    edge_width = 2
    node_color = 'skyblue'
    edge_color = 'salmon'
    alpha = 0.7

    for i in range(min(n * n, len(pred_hypergraphs))):
        ax = axs[i // n, i % n]
        H = pred_hypergraphs[i]

        # Clique expansion since hnx lacks a spring layout
        G = nx.Graph()
        for edge in H.edges:
            nodes = list(H.edges[edge])
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    G.add_edge(nodes[i], nodes[j])

        # Layout
        pos = nx.spring_layout(G, k=1, scale=1, iterations=100)

        # Draw hyperedges using convex hulls
        for i, edge in enumerate(H.edges):
            nodes = list(H.edges[edge])
            if len(nodes) > 2:
                points = np.array([pos[node] for node in nodes])
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]

                hull_color = np.random.random(3) * 0.5 + 0.5
                ax.fill(hull_points[:, 0], hull_points[:, 1], color=hull_color, alpha=0.3)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1],
                            color=edge_color, linewidth=edge_width, alpha=alpha)
            elif len(nodes) == 2:
                node1, node2 = nodes
                ax.plot([pos[node1][0], pos[node2][0]],
                        [pos[node1][1], pos[node2][1]],
                        color=edge_color, linewidth=edge_width, alpha=alpha)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                               node_color=node_color, alpha=1, linewidths=1, edgecolors='black')
        
        ax.title.set_text(f"N = {len(H.nodes)}")
        ax.title.set_fontsize(40)
        ax.axis('off')

    fig.tight_layout()
    return fig


def plot_mesh(pred_hypergraphs):
    """Visualize hypergraph predictions as 3D meshes."""
    n = int(min(5, np.ceil(np.sqrt(len(pred_hypergraphs)))))
    fig = plt.figure(figsize=(50, 50))
    axs = []

    for i in range(min(n, int(np.ceil(len(pred_hypergraphs)/n)))):
        for j in range(n):
            index = i * n + j
            if index >= len(pred_hypergraphs):
                break
            ax = fig.add_subplot(min(n, int(np.ceil(len(pred_hypergraphs)/n))), n, index + 1, projection='3d')
            axs.append(ax)

    for i in range(min(n * n, len(pred_hypergraphs))):
        ax = axs[i]
        H = pred_hypergraphs[i]

        positions = {node: H.nodes[node].feature for node in H.nodes}
        pos_array = np.array(list(positions.values()))

        # Plot nodes
        ax.scatter(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], c='b', s=20)

        face_collection = []
        line_collection = []
        for edge in H.edges:
            edge_nodes = list(H.edges[edge])
            if len(edge_nodes) > 2:
                face = np.array([positions[node] for node in edge_nodes])
                face_collection.append(face)
            elif len(edge_nodes) == 2:
                line = np.array([positions[node] for node in edge_nodes])
                line_collection.append(line)

        if face_collection:
            poly = Poly3DCollection(face_collection, alpha=0.25, facecolor='r', edgecolor='k')
            ax.add_collection3d(poly)

        if line_collection:
            lines = Line3DCollection(line_collection, colors='black', linewidths=2)
            ax.add_collection3d(lines)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        pos_min = np.min(pos_array)
        pos_max = np.max(pos_array)
        ax.set_xlim(pos_min, pos_max)
        ax.set_ylim(pos_min, pos_max)
        ax.set_zlim(pos_min, pos_max)

        ax.title.set_text(f"N = {len(H.nodes)}")
        ax.title.set_fontsize(40)

    fig.tight_layout()
    return fig


def plot_molecule(pred_hypergraphs, explicit_H):
    """Visualize hypergraph predictions as molecular structures."""
    n = int(min(5, np.ceil(np.sqrt(len(pred_hypergraphs)))))
    fig, axs = plt.subplots(min(n, int(np.ceil(len(pred_hypergraphs)/n))), n, figsize=(50, 50))

    if explicit_H:
        allowed_atom_types = [1, 6, 7, 8, 9, 5, 14, 15, 16, 17, 35, 53]
    else:
        allowed_atom_types = [6, 7, 8, 9, 5, 14, 15, 16, 17, 35, 53]

    atom_types_decode = {idx: atomic_num for idx, atomic_num in enumerate(allowed_atom_types)}
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    for i in range(min(n * n, len(pred_hypergraphs))):
        ax = axs[i // n, i % n]
        H = pred_hypergraphs[i]
        mol = Chem.RWMol()
        node_to_atom = {}

        for node in H.nodes:
            one_hot = H.nodes[node].feature
            atom_type = atom_types_decode[np.argmax(one_hot)]
            if atom_type is not None:
                atom = Chem.Atom(atom_type)
                if explicit_H:
                    atom.SetNoImplicit(True)
                atom_idx = mol.AddAtom(atom)
                node_to_atom[node] = atom_idx

        for edge in H.edges:
            nodes = list(H.edges[edge])
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    u, v = nodes[i], nodes[j]
                    bond_type = np.argmax(H.edges[edge].feature)
                    if mol.GetBondBetweenAtoms(node_to_atom[u], node_to_atom[v]) is None and bond_type <= 3:
                        mol.AddBond(node_to_atom[u], node_to_atom[v], bond_types[bond_type])

        img = Draw.MolToImage(mol, size=(500, 500))
        ax.imshow(img)
        
        ax.title.set_text(f"N = {len(H.nodes)}")
        ax.title.set_fontsize(40)
        ax.axis('off')

    fig.tight_layout()
    return fig