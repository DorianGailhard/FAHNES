import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx
import numpy as np

from scipy.spatial import cKDTree

# suppress networkx future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


from util.eval_helper import (
    clustering_stats,
    compute_list_eigh,
    degree_stats,
    eval_acc_planar_graph,
    eval_acc_sbm_graph,
    eval_acc_tree_graph,
    eval_fraction_isomorphic,
    eval_fraction_unique,
    eval_fraction_unique_non_isomorphic_valid,
    is_planar_graph,
    is_sbm_graph,
    orbit_stats_all,
    spectral_filter_stats,
    spectral_stats,
)


@dataclass
class Metric(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __call__(
        self,
        reference_graphs: list[nx.Graph],
        predicted_graphs: list[nx.Graph],
        train_graphs: list[nx.Graph],
    ) -> float:
        pass


class NodeNumDiff(Metric):
    def __str__(self):
        return "NodeNumDiff"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        ref_node_num = np.array([G.number_of_nodes() for G in reference_graphs])
        pred_node_num = np.array([G.number_of_nodes() for G in predicted_graphs])
        return np.mean(np.abs(ref_node_num - pred_node_num))


class NodeDegree(Metric):
    def __str__(self):
        return "Deg."

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return degree_stats(reference_graphs, predicted_graphs)


class ClusteringCoefficient(Metric):
    def __str__(self):
        return "Clus."

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return clustering_stats(reference_graphs, predicted_graphs)


class OrbitCount(Metric):
    def __str__(self):
        return "Orbit"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return orbit_stats_all(reference_graphs, predicted_graphs)


class Spectral(Metric):
    # Eigenvalues of normalized Laplacian
    def __str__(self):
        return "Spectral"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return spectral_stats(reference_graphs, predicted_graphs)


class Wavelet(Metric):
    def __str__(self):
        return "Wavelet"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        reference_eigvals, reference_eigvecs = compute_list_eigh(reference_graphs)
        predicted_eigvals, predicted_eigvecs = compute_list_eigh(predicted_graphs)
        return spectral_filter_stats(
            reference_eigvecs, reference_eigvals, predicted_eigvecs, predicted_eigvals
        )


class Ratio(Metric):
    def __init__(self):
        self.cache = {}

    def __str__(self):
        return "Ratio"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        metrics = [
            NodeDegree(),
            ClusteringCoefficient(),
            OrbitCount(),
            Spectral(),
            Wavelet(),
        ]

        # Use cached train metrics if available otherwise compute them
        reference_graphs_key = hash(tuple(reference_graphs))
        if reference_graphs_key in self.cache:
            train_metrics = self.cache[reference_graphs_key]
        else:
            train_metrics = np.array(
                [m(reference_graphs, train_graphs, train_graphs) for m in metrics]
            )
            self.cache[reference_graphs_key] = train_metrics

        predicted_metrics = np.array(
            [m(reference_graphs, predicted_graphs, train_graphs) for m in metrics]
        )

        return np.mean(predicted_metrics / (train_metrics + 1e-8))


class Uniqueness(Metric):
    def __str__(self):
        return "Uniqueness"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_fraction_unique(predicted_graphs, precise=True)


class Novelty(Metric):
    def __str__(self):
        return "Novelty"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return 1 - eval_fraction_isomorphic(
            fake_graphs=predicted_graphs, train_graphs=train_graphs
        )


class ValidPlanar(Metric):
    def __str__(self):
        return "ValidPlanar"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_acc_planar_graph(predicted_graphs)


class ValidTree(Metric):
    def __str__(self):
        return "ValidTree"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_acc_tree_graph(predicted_graphs)


class ValidSBM(Metric):
    def __str__(self):
        return "ValidSBM"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_acc_sbm_graph(predicted_graphs)


class UniqueNovelValidPlanar(Metric):
    def __str__(self):
        return "UniqueNovelValidPlanar"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_fraction_unique_non_isomorphic_valid(
            predicted_graphs, train_graphs, is_planar_graph
        )[2]


class UniqueNovelValidTree(Metric):
    def __str__(self):
        return "UniqueNovelValidTree"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_fraction_unique_non_isomorphic_valid(
            predicted_graphs, train_graphs, nx.is_tree
        )[2]


class UniqueNovelValidSBM(Metric):
    def __str__(self):
        return "UniqueNovelValidSBM"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_fraction_unique_non_isomorphic_valid(
            predicted_graphs, train_graphs, is_sbm_graph
        )[2]
    
    

# Point clouds
def _graphs_to_pointclouds(graphs):
    """
    Convert networkx graphs into numpy arrays of node features.
    Assumes node features are stored as graph.nodes[n]['feature'].
    """
    pointclouds = []
    for g in graphs:
        # Collect node features
        features = []
        for n in g.nodes:
            feat = g.nodes[n].get("feature", None)
            if feat is not None:
                features.append(feat)
        if features:
            pointclouds.append(np.array(features, dtype=np.float32))
    return pointclouds

def _chamfer_distance(pred_pc, ref_pc):
    """
    Compute Chamfer Distance between two point clouds.
    """
    ref_tree = cKDTree(ref_pc)
    pred_tree = cKDTree(pred_pc)

    dist_pred_to_ref, _ = ref_tree.query(pred_pc)
    dist_ref_to_pred, _ = pred_tree.query(ref_pc)

    return np.mean(dist_pred_to_ref) + np.mean(dist_ref_to_pred)


class ChamferNearestNeighborDistance(Metric):
    def __str__(self):
        return "ChamferNearestNeighborDistance"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        # Convert graphs into point clouds (just node features)
        ref_pcs = _graphs_to_pointclouds(reference_graphs)
        pred_pcs = _graphs_to_pointclouds(predicted_graphs)

        chamfer_dists = []
        for pred_pc in pred_pcs:
            min_chamfer_distance = float("inf")
            for ref_pc in ref_pcs:
                chamfer_dist = _chamfer_distance(pred_pc, ref_pc)
                if chamfer_dist < min_chamfer_distance:
                    min_chamfer_distance = chamfer_dist
            chamfer_dists.append(min_chamfer_distance)

        return np.mean(chamfer_dists)


class ChamferCoverage(Metric):
    def __str__(self):
        return "ChamferCoverage"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        reference_pcs = _graphs_to_pointclouds(reference_graphs)
        predicted_pcs = _graphs_to_pointclouds(predicted_graphs)

        matched_ref_indices = set()
        for pred_pc in predicted_pcs:
            min_dist = float('inf')
            min_idx = -1
            for i, ref_pc in enumerate(reference_pcs):
                dist = _chamfer_distance(pred_pc, ref_pc)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            if min_idx != -1:
                matched_ref_indices.add(min_idx)
        return len(matched_ref_indices) / len(reference_pcs) if reference_pcs else 0.0


class ChamferDiversity(Metric):
    def __str__(self):
        return "ChamferDiversity"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        predicted_pcs = _graphs_to_pointclouds(predicted_graphs)

        if len(predicted_pcs) < 2:
            return 0.0

        total_dist = 0.0
        count = 0
        for i in range(len(predicted_pcs)):
            for j in range(i + 1, len(predicted_pcs)):
                total_dist += _chamfer_distance(predicted_pcs[i], predicted_pcs[j])
                count += 1
        return total_dist / count if count > 0 else 0.0


class ChamferMMD(Metric):
    def __str__(self):
        return "ChamferMMD"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        reference_pcs = _graphs_to_pointclouds(reference_graphs)
        predicted_pcs = _graphs_to_pointclouds(predicted_graphs)

        chamfer_dists = []
        for ref_pc in reference_pcs:
            min_dist = float('inf')
            for pred_pc in predicted_pcs:
                dist = _chamfer_distance(ref_pc, pred_pc)
                if dist < min_dist:
                    min_dist = dist
            chamfer_dists.append(min_dist)
        return np.mean(chamfer_dists)