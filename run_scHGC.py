#!/usr/bin/env python
"""
scHGC Command Line Interface
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import scanpy as sc
import torch
from typing import Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scHGC.scHGC import scHGC
from scHGC.data_preprocess import run_preprocess
from scHGC.graph import run_graph_construction_optimized
from scHGC.utils import setup_logger, ensure_dir

logger = setup_logger('scHGC.CLI')


def load_config(config_path: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Override with command line args
    if getattr(args, 'epochs', None):
        config.setdefault('training', {})['epochs'] = args.epochs
    if getattr(args, 'learning_rate', None):
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    if getattr(args, 'n_hvg', None):
        config.setdefault('preprocessing', {}).setdefault('normalization', {})['n_top_genes'] = args.n_hvg
    if getattr(args, 'latent_dim', None):
        config.setdefault('model', {})['latent_dim'] = args.latent_dim

    return config


def train_mode(args):
    config = load_config(args.config, args)
    config['output_dir'] = args.output or f"results_train_{np.random.randint(10000)}"

    model = scHGC(config=config, device=args.device, random_seed=args.seed)
    model.fit(args.data, val_data=args.val_data, _use_precomputed=True)

    embedding, denoised = model.transform(return_denoised=True)
    model.save()

    logger.info(f"Training completed. Output: {config['output_dir']}")
    logger.info(f"Embedding shape: {embedding.shape}")


def predict_mode(args):
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = scHGC(device=args.device)
    model.load(args.model)

    output_dir = args.output or f"results_predict_{np.random.randint(10000)}"
    ensure_dir(output_dir)

    if args.return_denoised:
        embedding, denoised = model.transform(args.data, return_denoised=True)
        np.save(os.path.join(output_dir, 'denoised.npy'), denoised)
    else:
        embedding = model.transform(args.data)

    np.save(os.path.join(output_dir, 'embedding.npy'), embedding)
    logger.info(f"Prediction saved to {output_dir}")


def evaluate_mode(args):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.cluster import KMeans

    model = scHGC(device=args.device)
    model.load(args.model)

    adata = sc.read(args.data)
    embedding = model.transform(adata)

    metrics = {}
    if args.labels_key and args.labels_key in adata.obs:
        true_labels = adata.obs[args.labels_key].values
        n_clusters = len(np.unique(true_labels))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        pred_labels = kmeans.fit_predict(embedding)

        metrics['ARI'] = adjusted_rand_score(true_labels, pred_labels)
        metrics['NMI'] = normalized_mutual_info_score(true_labels, pred_labels)
        metrics['Silhouette'] = silhouette_score(embedding, true_labels)

        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")

    if args.output:
        ensure_dir(args.output)
        with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        np.save(os.path.join(args.output, 'embedding.npy'), embedding)


def preprocess_mode(args):
    config = load_config(args.config, args)
    output_dir = args.output or 'data/processed'
    ensure_dir(output_dir)

    X = run_preprocess(
        config=config.get('preprocessing', {}),
        data_path=args.data,
        output_dir=output_dir
    )
    logger.info(f"Preprocessing completed. Shape: {X.shape}")


def build_graphs_mode(args):
    config = load_config(args.config, args)
    output_dir = args.output or 'data/graphs'
    ensure_dir(output_dir)

    adata = sc.read(args.data)
    graphs, paths = run_graph_construction_optimized(
        config=config.get('graph_construction', {}),
        adata=adata,
        output_dir=output_dir,
        use_parallel=not args.no_parallel,
        use_cache=not args.no_cache
    )

    for name, graph in graphs.items():
        logger.info(f"{name}: {graph.shape[0]} nodes, {graph.nnz} edges")


def main():
    parser = argparse.ArgumentParser(description='scHGC: Single-cell Heterogeneous Graph Contrastive Learning')
    subparsers = parser.add_subparsers(dest='mode', help='Running mode')

    # Train mode
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('data', type=str, help='Training data path')
    train_parser.add_argument('--val-data', type=str, help='Validation data path')
    train_parser.add_argument('--config', type=str, help='Config file path')
    train_parser.add_argument('-o', '--output', type=str, help='Output directory')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--n-hvg', type=int, help='Number of HVGs')
    train_parser.add_argument('--latent-dim', type=int, help='Latent dimension')
    train_parser.add_argument('--device', type=str, default='auto', help='Device')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Predict mode
    predict_parser = subparsers.add_parser('predict', help='Predict using trained model')
    predict_parser.add_argument('model', type=str, help='Model path')
    predict_parser.add_argument('data', type=str, help='Input data path')
    predict_parser.add_argument('-o', '--output', type=str, help='Output directory')
    predict_parser.add_argument('--return-denoised', action='store_true', help='Return denoised expression')
    predict_parser.add_argument('--device', type=str, default='auto', help='Device')

    # Evaluate mode
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('model', type=str, help='Model path')
    eval_parser.add_argument('data', type=str, help='Test data path')
    eval_parser.add_argument('--labels-key', type=str, help='Labels key in adata.obs')
    eval_parser.add_argument('-o', '--output', type=str, help='Output directory')
    eval_parser.add_argument('--device', type=str, default='auto', help='Device')

    # Preprocess mode
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.add_argument('data', type=str, help='Raw data path')
    preprocess_parser.add_argument('--config', type=str, help='Config file path')
    preprocess_parser.add_argument('-o', '--output', type=str, help='Output directory')
    preprocess_parser.add_argument('--n-hvg', type=int, help='Number of HVGs')

    # Build graphs mode
    graph_parser = subparsers.add_parser('build-graphs', help='Build graphs')
    graph_parser.add_argument('data', type=str, help='Preprocessed data path')
    graph_parser.add_argument('--config', type=str, help='Config file path')
    graph_parser.add_argument('-o', '--output', type=str, help='Output directory')
    graph_parser.add_argument('--no-parallel', action='store_true', help='Disable parallel')
    graph_parser.add_argument('--no-cache', action='store_true', help='Disable cache')

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(0)

    logger.info(f"scHGC v1.0.0 - PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")

    try:
        if args.mode == 'train':
            train_mode(args)
        elif args.mode == 'predict':
            predict_mode(args)
        elif args.mode == 'evaluate':
            evaluate_mode(args)
        elif args.mode == 'preprocess':
            preprocess_mode(args)
        elif args.mode == 'build-graphs':
            build_graphs_mode(args)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()