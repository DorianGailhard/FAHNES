from rdkit import Chem
from fcd import get_fcd

from .metrics_hypergraph import Metric

import numpy as np


allowed_atom_types = [1, 6, 7, 8, 9, 5, 14, 15, 16, 17, 35, 53]  # H, C, N, O, F, B, Si, P, S, Cl, Br, I
atom_types_decode = {idx: atomic_num for idx, atomic_num in enumerate(allowed_atom_types)}

allowed_atom_types_implicit_H = [6, 7, 8, 9, 5, 14, 15, 16, 17, 35, 53]  # C, N, O, F, B, Si, P, S, Cl, Br, I
atom_types_decode_implicit_H = {idx: atomic_num for idx, atomic_num in enumerate(allowed_atom_types_implicit_H)}

bond_types = [Chem.rdchem.BondType.SINGLE,  
              Chem.rdchem.BondType.DOUBLE,  
              Chem.rdchem.BondType.TRIPLE,  
              Chem.rdchem.BondType.AROMATIC]


def convert_to_mol(H, implicit_H):
    """Convert a hypergraph to an rdkit Mol"""    
    mol = Chem.RWMol()
    node_to_atom = {}
    
    # Add atoms
    for node in H.nodes:        
        if implicit_H == False:
            atom_type = atom_types_decode[np.argmax(H.nodes[node].feature)]
        else:
            atom_type = atom_types_decode_implicit_H[np.argmax(H.nodes[node].feature)]
            
        atom = Chem.Atom(atom_type)
        
        if implicit_H == False:
            atom.SetNoImplicit(True)
            
        atom_idx = mol.AddAtom(atom)
        node_to_atom[node] = atom_idx
    
    # Add bonds
    for edge in H.edges:
        nodes = list(H.edges[edge])
        # Connect every pair of nodes (clique)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                bond_type = np.argmax(H.edges[edge].feature)
                if mol.GetBondBetweenAtoms(node_to_atom[u], node_to_atom[v]) is None and bond_type <= 3: # Ignore functional group hyperedges
                    mol.AddBond(
                        node_to_atom[u], node_to_atom[v],
                        bond_types[bond_type]
                    )
    
    return mol


class ValidMolecule(Metric):
    def __init__(self, implicit_H=False):
        self.implicit_H = implicit_H
        
    def __str__(self):
        return "ValidMolecule"
        
    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        valid_count = 0
        total = len(predicted_hypergraphs)
        
        for H in predicted_hypergraphs:
            if H.is_connected():
                mol = convert_to_mol(H, self.implicit_H)
                
                try:
                    Chem.SanitizeMol(mol)
                    valid_count += 1
                except:
                    continue
                
        return valid_count / total if total > 0 else 0

class UniqueMolecule(Metric):
    def __init__(self, implicit_H=False):
        self.implicit_H = implicit_H
        
    def __str__(self):
        return "UniqueMolecule"
        
    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        smiles_set = set()
        total = len(predicted_hypergraphs)
        
        for H in predicted_hypergraphs:
            mol = convert_to_mol(H, self.implicit_H)
            smiles = Chem.MolToSmiles(mol)
            smiles_set.add(smiles)
        
        unique_count = len(smiles_set)
        return unique_count / total if total > 0 else 0

class NovelMolecule(Metric):
    def __init__(self, implicit_H=False):
        self.implicit_H = implicit_H
        
    def __str__(self):
        return "NovelMolecule"
        
    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        train_smiles_set = set(Chem.MolToSmiles(convert_to_mol(H, self.implicit_H)) for H in train_hypergraphs)
        total = len(predicted_hypergraphs)
        novel_count = 0
        
        for H in predicted_hypergraphs:
            mol = convert_to_mol(H, self.implicit_H)
            smiles = Chem.MolToSmiles(mol)
            if smiles not in train_smiles_set and smiles is not None:
                novel_count += 1
                
        return novel_count / total if total > 0 else 0

class FCD(Metric):
    def __init__(self, implicit_H=False, device='cpu'):
        self.device = device
        self.implicit_H = implicit_H
        
    def __str__(self):
        return "FCD"
        
    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        smiles_list_reference = []
        for H in reference_hypergraphs:
            mol = convert_to_mol(H, self.implicit_H)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
                smi = Chem.MolToSmiles(mol, canonical=True)
                smiles_list_reference.append(smi)
            except:
                continue
    
        smiles_list_predicted = []
        for H in predicted_hypergraphs:
            mol = convert_to_mol(H, self.implicit_H)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
                smi = Chem.MolToSmiles(mol, canonical=True)
                smiles_list_predicted.append(smi)
            except:
                continue
            
        return get_fcd(smiles_list_reference, smiles_list_predicted, device=self.device)