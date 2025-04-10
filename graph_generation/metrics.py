from abc import ABC, abstractmethod
from dataclasses import dataclass

import hypernetx as hnx

import numpy as np

from collections import Counter
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

from rdkit import Chem

from functools import cache

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

class ValidEgo(Metric):
    def is_ego_hypergraph(self, H):
        if len(H.nodes) == 0 or len(H.edges) == 0:
            return False
    
        # Find the node that appears in the most hyperedges
        node_frequencies = {node: 0 for node in H.nodes}
        for edge in H.edges:
            for node in H.edges[edge]:
                node_frequencies[node] += 1
        
        potential_ego = max(node_frequencies, key=node_frequencies.get)
        
        # Check if the potential ego is in all or most hyperedges
        edges_with_ego = sum(1 for edge in H.edges if potential_ego in H.edges[edge])
        if edges_with_ego < len(H.edges):
            return False
    
        # Check if all other nodes are directly connected to the ego
        nodes_connected_to_ego = set()
        for edge in H.edges:
            if potential_ego in H.edges[edge]:
                nodes_connected_to_ego.update(H.edges[edge])
        
        if nodes_connected_to_ego != set(H.nodes):
            return False
    
        # Check if there are any hyperedges not including the ego or its direct connections
        for edge in H.edges:
            if potential_ego not in H.edges[edge] and not any(node in nodes_connected_to_ego for node in H.edges[edge]):
                return False
    
        return True
    
    def __str__(self):
        return "ValidEgo"
    
    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        valid = 0
        
        for H in predicted_hypergraphs:
            if self.is_ego_hypergraph(H):
                valid += 1
        
        return valid/len(predicted_hypergraphs)
    

class ValidHypertree(Metric):
    def __str__(self):
        return "ValidHypertree"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        return eval_acc_tree_hypergraph(predicted_hypergraphs)
    

class ValidSBM(Metric):
    def __str__(self):
        return "ValidSBM"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        return eval_acc_sbm_hypergraph(predicted_hypergraphs)
    

class ChamferDistance:
    def __str__(self):
        return "ChamferDistance"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs=None):
        predicted_features = np.array([H.nodes[n].feature for H in predicted_hypergraphs for n in H.nodes])
        reference_features = np.array([H.nodes[n].feature for H in reference_hypergraphs for n in H.nodes])

        return self.chamfer_distance(predicted_features, reference_features)

    def chamfer_distance(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X.shape[0] == 0 or Y.shape[0] == 0:
            return np.inf

        # Compute pairwise distances
        dists = cdist(X, Y)

        # Compute Chamfer distance
        min_dist_x = np.min(dists, axis=1)
        min_dist_y = np.min(dists, axis=0)
        chamfer_dist = np.mean(min_dist_x) + np.mean(min_dist_y)

        return chamfer_dist
    

def convert_to_mol(H):
    """Convert a hypergraph to an rdkit Mol"""
    # Convert node features to a hashable tuple
    node_features = tuple(
        (node, tuple(H.nodes[node].feature.tolist())) 
        for node in H.nodes
    )
    
    # Convert edge features to a hashable tuple
    edge_features = tuple(
        (tuple(list(H.edges[edge])), tuple(H.edges[edge].feature.tolist())) 
        for edge in H.edges
    )
    
    return convert_cached_hypergraph_to_mol(node_features, edge_features)


atom_types = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # H, C, N, O, F
atom_types_decode = {v: k for k, v in atom_types.items()}

bond_types = [Chem.rdchem.BondType.SINGLE,  
              Chem.rdchem.BondType.DOUBLE,  
              Chem.rdchem.BondType.TRIPLE,  
              Chem.rdchem.BondType.AROMATIC]

@cache
def convert_cached_hypergraph_to_mol(node_features, edge_features):
    """Convert cached hypergraph representation to RDKit molecule"""    
    mol = Chem.RWMol()
    node_to_atom = {}
    
    # Add atoms
    for node, feature in node_features:
        atom_type = atom_types_decode[np.argmax(feature)]
        atom = Chem.Atom(atom_type)
        atom.SetNoImplicit(True)
        atom_idx = mol.AddAtom(atom)
        node_to_atom[node] = atom_idx
    
    # Add bonds
    for (nodes, feature) in edge_features:
        if len(nodes) == 2:
            if mol.GetBondBetweenAtoms(node_to_atom[nodes[0]], node_to_atom[nodes[1]]) is None:
                mol.AddBond(
                    node_to_atom[nodes[0]], 
                    node_to_atom[nodes[1]], 
                    bond_types[np.argmax(feature)]
                )
    
    return mol


def check_valency(mol):
    """Check if all atoms have the correct valency, assuming explicit hydrogens."""
    for atom in mol.GetAtoms():
        valence = atom.GetExplicitValence()
        atomic_num = atom.GetAtomicNum()
        expected_valence = Chem.GetPeriodicTable().GetDefaultValence(atomic_num)
        
        if valence != expected_valence:
            return False  # Incorrect valency
    
    return True

class ValidMolecule(Metric):
    def __str__(self):
        return "ValidMolecule"
        
    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        valid_count = 0
        total = len(predicted_hypergraphs)
        
        for H in predicted_hypergraphs:
            if H.is_connected():
                mol = convert_to_mol(H)
                if not mol:
                    continue

                try:
                    Chem.SanitizeMol(mol)
                except:
                    continue

                if check_valency(mol):
                    valid_count += 1
                
        return valid_count / total if total > 0 else 0

class UniqueMolecule(Metric):
    def __str__(self):
        return "UniqueMolecule"
        
    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        smiles_set = set()
        total = len(predicted_hypergraphs)
        
        for H in predicted_hypergraphs:
            mol = convert_to_mol(H)
            smiles = Chem.MolToSmiles(mol)
            smiles_set.add(smiles)
        
        unique_count = len(smiles_set)
        return unique_count / total if total > 0 else 0

class NovelMolecule(Metric):
    def __str__(self):
        return "NovelMolecule"
        
    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        smiles_set = set()
        train_smiles_set = set(Chem.MolToSmiles(convert_to_mol(H)) for H in train_hypergraphs if convert_to_mol(H))
        total = len(predicted_hypergraphs)
        
        for H in predicted_hypergraphs:
            mol = convert_to_mol(H)
            smiles = Chem.MolToSmiles(mol)
            if smiles not in train_smiles_set and smiles is not None:
                smiles_set.add(smiles)
        
        novel_count = len(smiles_set)
        return novel_count / total if total > 0 else 0
