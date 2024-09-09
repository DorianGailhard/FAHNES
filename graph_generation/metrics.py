from abc import ABC, abstractmethod
from dataclasses import dataclass

import hypernetx as hnx

import numpy as np

from collections import Counter
from scipy.stats import wasserstein_distance

from util.eval_helper import (
    spectral_stats,
    eval_fraction_unique,
    eval_fraction_isomorphic,
    eval_acc_tree_hypergraph,
    eval_acc_sbm_hypergraph,
)

@dataclass
class Metric(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __call__(
        self,
        reference_hypergraphs: list[hnx.Hypergraph],
        predicted_hypergraphs: list[hnx.Hypergraph],
        train_hypergraphs: list[hnx.Hypergraph],
    ) -> float:
        pass


class NodeNumDiff(Metric):
    def __str__(self):
        return "NodeNumDiff"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        ref_node_num = np.array([len(H.nodes)
                                for H in reference_hypergraphs])
        pred_node_num = np.array([len(H.nodes)
                                 for H in predicted_hypergraphs])
        return np.mean(np.abs(ref_node_num - pred_node_num))


class NodeDegreeDistrWasserstein(Metric):
    def __str__(self):
        return "NodeDegreeDistrWasserstein"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        reference_dist = []
        for H in reference_hypergraphs:
            reference_dist += hnx.reports.descriptive_stats.degree_dist(H)
    
        pred_dist = []
        for H in predicted_hypergraphs:
            pred_dist += hnx.reports.descriptive_stats.degree_dist(H)

        # Convert to counters
        degree_dist1 = Counter(reference_dist)
        degree_dist2 = Counter(pred_dist)
        
        # Extract keys (degree values) and values (frequencies) from the counters
        degree_dist1_keys = list(degree_dist1.keys())
        degree_dist1_values = list(degree_dist1.values())
        degree_dist2_keys = list(degree_dist2.keys())
        degree_dist2_values = list(degree_dist2.values())
        
        # Compute the Wasserstein distance
        return wasserstein_distance(
            np.array(degree_dist1_keys),
            np.array(degree_dist2_keys),
            np.array(degree_dist1_values),
            np.array(degree_dist2_values)
        )


class EdgeSizeDistrWasserstein(Metric):
    def __str__(self):
        return "EdgeSizeDistrWasserstein"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        reference_dist = []
        for H in reference_hypergraphs:
            reference_dist += hnx.reports.descriptive_stats.edge_size_dist(H)
    
        pred_dist = []
        for H in predicted_hypergraphs:
            pred_dist += hnx.reports.descriptive_stats.edge_size_dist(H)

        # Convert to counters
        degree_dist1 = Counter(reference_dist)
        degree_dist2 = Counter(pred_dist)
        
        # Extract keys (degree values) and values (frequencies) from the counters
        degree_dist1_keys = list(degree_dist1.keys())
        degree_dist1_values = list(degree_dist1.values())
        degree_dist2_keys = list(degree_dist2.keys())
        degree_dist2_values = list(degree_dist2.values())
        
        # Compute the Wasserstein distance
        return wasserstein_distance(
            np.array(degree_dist1_keys),
            np.array(degree_dist2_keys),
            np.array(degree_dist1_values),
            np.array(degree_dist2_values)
        )
    

class Spectral(Metric):
    # Eigenvalues of normalized Laplacian
    def __str__(self):
        return "Spectral"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        return spectral_stats(reference_hypergraphs, predicted_hypergraphs)
    

class Uniqueness(Metric):
    def __str__(self):
        return "Uniqueness"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        return eval_fraction_unique(predicted_hypergraphs, precise=True)


class Novelty(Metric):
    def __str__(self):
        return "Novelty"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        return 1 - eval_fraction_isomorphic(
            fake_hypergraphs=predicted_hypergraphs, train_hypergraphs=train_hypergraphs
        )
    
    
class CentralityCloseness(Metric):
    def __str__(self):
        return "CentralityCloseness"

    def compute_centrality_distribution(self, hypergraphs):
        all_centralities = []
        for H in hypergraphs:
            centralities = hnx.algorithms.s_centrality_measures.s_closeness_centrality(H)
            all_centralities.extend(centralities.values())
        return np.array(all_centralities)

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        centrality_pred = self.compute_centrality_distribution(predicted_hypergraphs)
        centrality_reference = self.compute_centrality_distribution(reference_hypergraphs)
        
        return wasserstein_distance(centrality_pred, centrality_reference)


class CentralityBetweenness(Metric):
    def __str__(self):
        return "CentralityBetweenness"

    def compute_centrality_distribution(self, hypergraphs):
        all_centralities = []
        for H in hypergraphs:
            centralities = hnx.algorithms.s_centrality_measures.s_betweenness_centrality(H)
            all_centralities.extend(centralities.values())
        return np.array(all_centralities)

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        centrality_pred = self.compute_centrality_distribution(predicted_hypergraphs)
        centrality_reference = self.compute_centrality_distribution(reference_hypergraphs)
        
        return wasserstein_distance(centrality_pred, centrality_reference)


class CentralityHarmonic(Metric):
    def __str__(self):
        return "CentralityHarmonic"

    def compute_centrality_distribution(self, hypergraphs):
        all_centralities = []
        for H in hypergraphs:
            centralities = hnx.algorithms.s_centrality_measures.s_harmonic_centrality(H)
            all_centralities.extend(centralities.values())
        return np.array(all_centralities)

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        centrality_pred = self.compute_centrality_distribution(predicted_hypergraphs)
        centrality_reference = self.compute_centrality_distribution(reference_hypergraphs)
        
        return wasserstein_distance(centrality_pred, centrality_reference)


class NodeFeaturesWasserstein(Metric):
    def __str__(self):
        return "NodeFeaturesWasserstein"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        predicted_features = [H.nodes[n].feature for H in predicted_hypergraphs for n in H.nodes]
        reference_features = [H.nodes[n].feature for H in reference_hypergraphs for n in H.nodes]
        
        return wasserstein_distance(predicted_features, reference_features)
    
    
class HyperedgeFeaturesWasserstein(Metric):
    def __str__(self):
        return "HyperedgeFeaturesWasserstein"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        predicted_features = [H.edges[e].feature for H in predicted_hypergraphs for e in H.edges]
        reference_features = [H.edges[e].feature for H in reference_hypergraphs for e in H.edges]
        
        return wasserstein_distance(predicted_features, reference_features)