import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch as th
import torch.multiprocessing as mp
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import networkx as nx

import graph_generation as gg


def get_expansion_items(cfg: DictConfig, train_graphs):
    # Spectral Features
    spectrum_extractor = (
        gg.spectral.SpectrumExtractor(
            num_features=cfg.spectral.num_features,
            normalized=cfg.spectral.normalized_laplacian,
        )
        if cfg.spectral.num_features > 0
        else None
    )

    # Train Dataset
    red_factory = gg.reduction.ReductionFactory(
        contraction_family=cfg.reduction.contraction_family,
        cost_type=cfg.reduction.cost_type,
        topological_cost_importance=cfg.reduction.get('topological_cost_importance', 1),
        preserved_eig_size=cfg.reduction.preserved_eig_size,
        sqrt_partition_size=cfg.reduction.sqrt_partition_size,
        weighted_reduction=cfg.reduction.weighted_reduction,
        min_red_frac=cfg.reduction.min_red_frac,
        max_red_frac=cfg.reduction.max_red_frac,
        red_threshold=cfg.reduction.red_threshold,
        rand_lambda=cfg.reduction.rand_lambda,
        extract_node_features=(cfg.dataset.node_features_dim>0),
    )

    if cfg.reduction.num_red_seqs > 0:
        train_dataset = gg.data.FiniteRandRedDataset(
            graphs=train_graphs,
            red_factory=red_factory,
            spectrum_extractor=spectrum_extractor,
            num_red_seqs=cfg.reduction.num_red_seqs,
        )
    else:
        train_dataset = gg.data.InfiniteRandRedDataset(
            graphs=train_graphs,
            red_factory=red_factory,
            spectrum_extractor=spectrum_extractor,
        )

    # Dataloader
    is_mp = cfg.reduction.num_red_seqs < 0 and cfg.training.max_num_workers > 0 # if infinite dataset
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=Batch.from_data_list,
        num_workers=min(mp.cpu_count(), cfg.training.max_num_workers) * is_mp,
        multiprocessing_context="spawn" if is_mp else None,
    )

    # Model
    if cfg.spectral.num_features > 0:
        sign_net = gg.model.SignNet(
            num_eigenvectors=cfg.spectral.num_features,
            hidden_features=cfg.sign_net.hidden_features,
            out_features=cfg.model.expansion_emb_features,
            num_layers=cfg.sign_net.num_layers,
        )
    else:
        sign_net = None

    if cfg.model.name == "ppgn":
        model = gg.model.SparsePPGN(
            node_in_features=cfg.dataset.node_features_dim,
            node_out_features=cfg.dataset.node_features_dim,
            node_emb_features=cfg.model.node_emb_features,
            expansion_node_in_features=2,
            expansion_edge_in_features=1,
            expansion_node_out_features=2,
            expansion_edge_out_features=1,
            expansion_emb_features=cfg.model.expansion_emb_features,
            hidden_features=cfg.model.hidden_features,
            ppgn_features=cfg.model.ppgn_features,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            self_conditioning=cfg.diffusion.self_conditioning
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    # Diffusion
    if cfg.diffusion.name == "fm":
        diffusion = gg.diffusion.FM(
            use_lognorm_t=cfg.diffusion.use_lognorm_t,
            self_conditioning=cfg.diffusion.self_conditioning,
            num_steps=cfg.diffusion.num_steps,
            node_features_on_simplex = cfg.dataset.get('node_features_on_simplex', False)
        )
    else:
        raise ValueError(f"Unknown diffusion name: {cfg.diffusion.name}")

    # Method
    method = gg.method.Expansion(
        diffusion=diffusion,
        spectrum_extractor=spectrum_extractor,
        emb_features=cfg.model.expansion_emb_features,
        augmented_radius=cfg.method.augmented_radius,
        augmented_dropout=cfg.method.augmented_dropout,
        deterministic_expansion=cfg.method.deterministic_expansion,
        min_red_frac=cfg.reduction.min_red_frac,
        max_red_frac=cfg.reduction.max_red_frac,
        red_threshold=cfg.reduction.red_threshold,
        node_features_noise_strength = cfg.dataset.get('node_features_noise_strength', 0),
        node_features_on_simplex = cfg.dataset.get('node_features_on_simplex', False)
    )

    return {
        "train_dataloader": train_dataloader,
        "method": method,
        "model": model,
        "sign_net": sign_net
    }

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.debugging:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Fix random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
       
    # graphs
    if cfg.dataset.load:
        dataset_dir = Path("./data")
        dataset_name = cfg.dataset.name
        
        with open(dataset_dir / f"{dataset_name}.pkl", "rb") as f:
            dataset = pickle.load(f)
        
        train_graphs = dataset["train"]
        validation_graphs = dataset["val"]
        test_graphs = dataset["test"]
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")
    
    # keep only largest connected component for train graphs
    train_graphs = [
        G.subgraph(max(nx.connected_components(G), key=len)) for G in train_graphs
    ]

    # Metrics
    validation_metrics = [
        gg.metrics.NodeNumDiff(),
        gg.metrics.NodeDegree(),
        gg.metrics.ClusteringCoefficient(),
        gg.metrics.OrbitCount(),
        gg.metrics.Spectral(),
        gg.metrics.Wavelet(),
        gg.metrics.Ratio(),
        gg.metrics.Uniqueness(),
        gg.metrics.Novelty(),
    ]

    if "planar" in cfg.dataset.name:
        validation_metrics += [
            gg.metrics.ValidPlanar(),
            gg.metrics.UniqueNovelValidPlanar(),
        ]
    elif "tree" in cfg.dataset.name:
        validation_metrics += [
            gg.metrics.ValidTree(),
            gg.metrics.UniqueNovelValidTree(),
        ]
    elif "sbm" in cfg.dataset.name:
        validation_metrics += [
            gg.metrics.ValidSBM(),
            gg.metrics.UniqueNovelValidSBM(),
        ]
    elif "pointCloud" in cfg.dataset.name:
        validation_metrics += [
            gg.metrics.ChamferNearestNeighborDistance(),
            gg.metrics.ChamferCoverage(),
            gg.metrics.ChamferDiversity(),
            gg.metrics.ChamferMMD()
        ]

    # Method
    if cfg.method.name == "expansion":
        method_items = get_expansion_items(cfg, train_graphs)
    else:
        raise ValueError(f"Unknown method name: {cfg.method.name}")
        
    method_items = defaultdict(lambda: None, method_items)

    # Trainer
    th.set_float32_matmul_precision("high")
    trainer = gg.training.Trainer(
        sign_net=method_items["sign_net"],
        model=method_items["model"],
        method=method_items["method"],
        train_dataloader=method_items["train_dataloader"],
        train_graphs=train_graphs,
        validation_graphs=validation_graphs,
        test_graphs=test_graphs,
        metrics=validation_metrics,
        cfg=cfg,
    )
    if cfg.testing:
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
