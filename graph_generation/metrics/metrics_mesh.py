from .metrics_hypergraph import Metric

from util.point_cloud_helper import PointNet

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import sqrtm

import torch
from torch_geometric.transforms import SamplePoints
from torch_geometric.data import Data


class ChamferNearestNeighborDistance(Metric):
    def __str__(self):
        return "ChamferNearestNeighborDistance"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs):
        ref_pcs = self._hypergraphs_to_pointclouds(reference_hypergraphs)
        pred_pcs = self._hypergraphs_to_pointclouds(predicted_hypergraphs)

        chamfer_dists = []
        for pred_pc in pred_pcs:
            min_chamfer_distance = float('inf')
            for ref_pc in ref_pcs:
                chamfer_dist = self._chamfer_distance(pred_pc, ref_pc)
                if chamfer_dist < min_chamfer_distance:
                    min_chamfer_distance = chamfer_dist
            chamfer_dists.append(min_chamfer_distance)

        return np.mean(chamfer_dists)

    def _hypergraphs_to_pointclouds(self, hypergraphs, n_samples=1024):
        pointclouds = []
        sampler = SamplePoints(num=n_samples)

        for h in hypergraphs:
            node_id_map = {node_id: idx for idx, node_id in enumerate(h.nodes)}
            vertices = torch.tensor([h.nodes[node_id].feature for node_id in h.nodes], dtype=torch.float)

            faces = []
            for edge in h.edges:
                face = list(map(node_id_map.get, h.edges[edge]))

                if len(face) < 3:
                    continue
                elif len(face) == 3:
                    faces.append(face)
                else:
                    v0 = face[0]
                    for i in range(1, len(face) - 1):
                        faces.append([v0, face[i], face[i + 1]])

            if not faces:
                continue

            faces = torch.tensor(faces, dtype=torch.long).T  # Shape: [3, num_faces]
            data = Data(pos=vertices, face=faces)
            sampled = sampler(data)
            pointclouds.append(sampled.pos.numpy())

        return pointclouds

    def _chamfer_distance(self, pred_pc, ref_pc):
        ref_tree = cKDTree(ref_pc)
        pred_tree = cKDTree(pred_pc)

        dist_pred_to_ref, _ = ref_tree.query(pred_pc)
        dist_ref_to_pred, _ = pred_tree.query(ref_pc)

        return np.mean(dist_pred_to_ref) + np.mean(dist_ref_to_pred)



# Doesn't seem to work
class FrechetMeshPointCloudDistance():
    def __init__(self, device='cpu'):
        self.device = device
        
        # Initialize PointNet
        self.model = PointNet()
        
        # Load pretrained weights if needed
        checkpoint_path = 'model.t7'
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        from collections import OrderedDict
        new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()  # set to evaluation mode

    def __str__(self):
        return "FrechetMeshPointCloudDistance"

    def __call__(self, reference_hypergraphs, predicted_hypergraphs, train_hypergraphs=None):
        # Step 1: Convert hypergraphs to point clouds (uniformly sample 1024 points per mesh)
        ref_pointclouds = self._hypergraphs_to_pointclouds(reference_hypergraphs)
        pred_pointclouds = self._hypergraphs_to_pointclouds(predicted_hypergraphs)

        # Step 2: Extract intermediate features
        ref_feats = self._extract_layerwise_features(ref_pointclouds)
        pred_feats = self._extract_layerwise_features(pred_pointclouds)
        
        # Step 3: Compute layer-wise feature distance
        return self._layerwise_distance(ref_feats, pred_feats)

    def _hypergraphs_to_pointclouds(self, hypergraphs, n_samples=1024):
        pointclouds = []
        sampler = SamplePoints(num=n_samples)

        for h in hypergraphs:
            node_id_map = {node_id: idx for idx, node_id in enumerate(h.nodes)}
            vertices = torch.tensor(np.array([h.nodes[node_id].feature for node_id in h.nodes]), dtype=torch.float)

            faces = []
            for edge in h.edges:
                face = list(map(node_id_map.get, h.edges[edge]))

                if len(face) < 3:
                    continue
                elif len(face) == 3:
                    faces.append(face)
                else:
                    v0 = face[0]
                    for i in range(1, len(face) - 1):
                        faces.append([v0, face[i], face[i + 1]])

            if not faces:
                continue

            faces = torch.tensor(faces, dtype=torch.long).T  # Shape: [3, num_faces]
            data = Data(pos=vertices, face=faces)
            sampled = sampler(data)
            pointclouds.append(sampled.pos.numpy())

        return pointclouds

    def _extract_layerwise_features(self, pointclouds):
        """
        Extract intermediate layer features from PointNet.
        Returns a dict of feature maps per layer.
        """
        pointclouds = torch.tensor(pointclouds, dtype=torch.float32).to(self.device)
        pointclouds = pointclouds.transpose(1, 2)  # [B, 3, N]
    
        with torch.no_grad():
            output = self.model(pointclouds)
        return output  # namedtuple of {layer_name: tensor of shape [B, C, N]}

    def _layerwise_distance(self, feats0, feats1):
        """
        Compute spatial average of squared differences per layer.
    
        Args:
            feats0, feats1: namedtuple of feature maps per layer, each [B, C, N]
    
        Returns:
            Scalar distance (float)
        """
        val = 0.0
    
        for key in feats0._fields:  # namedtuple field names: conv1, conv2, ...
            f0 = getattr(feats0, key)
            f1 = getattr(feats1, key)
    
            diffs = (f0 - f1) ** 2          # [B, C, N]
            diffs = diffs.sum(dim=1)       # Sum over channels → [B, N]
            diffs = diffs.mean(dim=1)      # Mean over points → [B]
            val += diffs.mean().item()     # Mean over batch
        return val
            
    def _compute_stats(self, features):
        """Compute mean and covariance of features."""
        mu = np.mean(features, axis=0)  # Mean of features across the batch
        sigma = np.cov(features, rowvar=False)  # Covariance of features
        return mu, sigma

    def _frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Fréchet Distance between two Gaussians."""
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)