"""
scHGC: Single-cell Heterogeneous Graph Contrastive Learning
"""

import os
import torch
import numpy as np
import scanpy as sc
from scipy import sparse
from typing import Optional, Dict, Any, Tuple, Union, List
import yaml

from .data_preprocess import run_preprocess, load_data
from .graph import run_graph_construction_optimized
from .model import create_scHGC_model
from .train import Trainer
from .utils import (
    setup_logger,
    set_random_seed,
    ensure_dir,
    sparse_to_torch,
    save_config,
    get_device
)

logger = setup_logger('scHGC', level='INFO')


class scHGC:
    """Single-cell Heterogeneous Graph Contrastive Learning model."""

    def __init__(self, config: Optional[Union[Dict, str]] = None,
                 device: Optional[str] = None, random_seed: int = 42):
        self.random_seed = random_seed
        self.device = get_device(device)
        self.config = self._load_config(config)
        self.is_trained = False

        set_random_seed(random_seed)

        self.model = None
        self.trainer = None
        self.graphs = None
        self.adata_processed = None
        self.embedding_ = None
        self.denoised_expression_ = None
        self.history_ = None
        self.X_train = None

        self.output_dir = self.config.get('output_dir', 'results')
        self._setup_directories()

    def _load_config(self, config: Optional[Union[Dict, str]]) -> Dict:
        # 递归合并字典的辅助函数
        def merge_dicts(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    d1[k] = merge_dicts(d1[k], v)
                else:
                    d1[k] = v
            return d1
        default_cfg = self._default_config()
        if config is None:
            return default_cfg
        user_cfg = {}
        if isinstance(config, str):
            with open(config, 'r') as f:
                user_cfg = yaml.safe_load(f)
        elif isinstance(config, dict):
            user_cfg = config
        return merge_dicts(default_cfg, user_cfg)

    def _default_config(self) -> Dict:
        return {
            'preprocessing': {
                'quality_control': {'min_genes': 200, 'min_cells': 3, 'max_genes': 2500, 'max_pct_mito': 0.05},
                'normalization': {'target_sum': 1e4, 'n_top_genes': 2048, 'log_transform': True, 'scale': True}
            },
            'graph_construction': {
                'knn': {'k': 15, 'n_pca': 100, 'weight_method': 'gaussian'},
                'mnn': {'k': 20, 'n_pca': 50, 'weight_method': 'inverse'},
                'cluster': {'resolution': 1.0, 'algorithm': 'louvain'},
                'normalize': True, 'normalize_method': 'symmetric'
            },
            'model': {
                'n_genes': 2048, 'hidden_dim': 512, 'latent_dim': 128,
                'decoder_hidden_dim': 512, 'fusion_type': 'concatenate',
                'conv_type': 'GCN', 'dropout': 0.1
            },
            'training': {
                'epochs': 200, 'learning_rate': 0.001, 'weight_decay': 0.0001,
                'gamma': 0.5, 'early_stop_patience': 20, 'optimizer': 'AdamW'
            }
        }

    def _setup_directories(self):
        dirs = ['checkpoints', 'embeddings', 'logs']
        for d in dirs:
            ensure_dir(os.path.join(self.output_dir, d))

    def preprocess(self, data: Union[str, sc.AnnData, np.ndarray]) -> Tuple[np.ndarray, sc.AnnData]:
        if isinstance(data, str):
            adata = load_data(data)
        elif isinstance(data, sc.AnnData):
            adata = data.copy()
        elif isinstance(data, np.ndarray):
            adata = sc.AnnData(data)
        else:
            raise ValueError("Invalid data type")

        X, adata_processed = run_preprocess(
            self.config['preprocessing'], adata=adata,
            output_dir=None, return_adata=True
        )

        self.config['model']['n_genes'] = X.shape[1]
        self.adata_processed = adata_processed
        return X, adata_processed

    def build_graphs(self, adata: Optional[sc.AnnData] = None) -> Dict[str, sparse.csr_matrix]:
        if adata is None:
            if self.adata_processed is None:
                raise ValueError("No processed data available")
            adata = self.adata_processed
        graph_output_dir = os.path.join(self.output_dir, 'graphs')

        graphs, _ = run_graph_construction_optimized(
            config=self.config['graph_construction'],
            adata=adata,
            output_dir=graph_output_dir,
            use_parallel=True,
            use_cache=True
        )
        self.graphs = graphs
        return graphs

    def _prepare_data(self, X: np.ndarray, graphs: Dict[str, sparse.csr_matrix]) -> Dict:
        adj_list = []
        for name in ['knn', 'mnn', 'cluster_mnn']:
            A = graphs[name]
            A_torch = sparse_to_torch(A) if sparse.issparse(A) else torch.FloatTensor(A)
            adj_list.append(A_torch)
        return {'X': X, 'adj_list': adj_list}

    def fit(self, data: Union[str, sc.AnnData, np.ndarray],
            val_data: Optional[Union[sc.AnnData, Tuple]] = None,
            _use_precomputed: bool = False) -> 'scHGC':

        if _use_precomputed:
            logger.info("Using precomputed data and graphs. Skipping preprocessing.")
            # 1. Load the AnnData object path
            adata_path = data
            adata = sc.read(adata_path)
            self.adata_processed = adata

            # 2. CRITICAL FIX: Load the HVG-filtered feature matrix from its .npy file
            processed_dir = os.path.dirname(adata_path)
            feature_path = os.path.join(processed_dir, 'processed_features.npy')
            if not os.path.exists(feature_path):
                # Fallback for PyTorch tensor file
                feature_path_pt = os.path.join(processed_dir, 'processed_features.pt')
                if os.path.exists(feature_path_pt):
                    X = torch.load(feature_path_pt).numpy()
                else:
                    raise FileNotFoundError(
                        f"Feature file not found at {feature_path} or {feature_path_pt}. Make sure it was generated by the 'preprocess' step.")
            else:
                X = np.load(feature_path)
            logger.info(f"Loaded feature matrix of shape {X.shape} from {feature_path}")

            # 3. Set the correct number of genes for the model
            self.config['model']['n_genes'] = X.shape[1]

            # 4. Load pre-built graphs
            graph_dir = os.path.join(os.path.dirname(os.path.dirname(data)), 'graphs')
            if not os.path.exists(graph_dir):
                graph_dir = os.path.join(self.output_dir, 'graphs')  # Fallback path

            logger.info(f"Loading graphs from {graph_dir}...")
            self.graphs = {}
            for name in ['knn', 'mnn', 'cluster_mnn']:
                graph_path = os.path.join(graph_dir, f'graph_{name}.npz')
                if os.path.exists(graph_path):
                    self.graphs[name] = sparse.load_npz(graph_path)
                else:
                    raise FileNotFoundError(f"Graph file not found: {graph_path}. Please run 'build-graphs' first.")
            logger.info(f"Loaded {len(self.graphs)} graphs.")

        else:
            # Original workflow: process from scratch
            X, adata = self.preprocess(data)
            if self.graphs is None:
                self.build_graphs(adata)

        # Create model if it doesn't exist
        if self.model is None:
            self.model = create_scHGC_model(self.config)
        self.X_train = X

        # Prepare data and train
        train_data = self._prepare_data(X, self.graphs)
        val_data_dict = None
        if val_data is not None:
            if isinstance(val_data, tuple):
                X_val, graphs_val = val_data
            else:
                X_val, adata_val = self.preprocess(val_data)
                graphs_val = self.build_graphs(adata_val)
            val_data_dict = self._prepare_data(X_val, graphs_val)

        self.trainer = Trainer(self.model, self.config, train_data, self.device)
        self.history_ = self.trainer.train(val_data_dict)
        self.is_trained = True

        logger.info("Training completed")
        return self

    def transform(self, data: Optional[Union[sc.AnnData, np.ndarray]] = None,
                  return_denoised: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if not self.is_trained:
            raise ValueError("Model not trained")

        if data is None:
            if self.X_train is None:
                raise ValueError("Training data not found. Please run fit() first.")
            X = self.X_train
            graphs = self.graphs
        else:
            X, adata = self.preprocess(data)
            graphs = self.build_graphs(adata)

        data_dict = self._prepare_data(X, graphs)
        X_tensor = torch.FloatTensor(data_dict['X']).to(self.device)
        adj_list = [adj.to(self.device) for adj in data_dict['adj_list']]

        self.model.eval()
        with torch.no_grad():
            Z_final = self.model.get_latent_representation(X_tensor, adj_list)
            embedding = Z_final.cpu().numpy()

            if return_denoised:
                mu = self.model.get_denoised_expression(X_tensor, adj_list)
                denoised = mu.cpu().numpy()
                return embedding, denoised

        return embedding

    def fit_transform(self, data: Union[str, sc.AnnData, np.ndarray],
                      return_denoised: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        self.fit(data)
        return self.transform(return_denoised=return_denoised)

    def save(self, path: Optional[str] = None):
        if not self.is_trained:
            raise ValueError("Model not trained")

        if path is None:
            path = os.path.join(self.output_dir, 'model.pth')

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history_
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']

        if self.model is None:
            self.model = create_scHGC_model(self.config)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.is_trained = True
        self.history_ = checkpoint.get('history')
        logger.info("Model loaded")


def run_scHGC(data_path: str, config_path: Optional[str] = None,
              output_dir: str = 'results') -> Tuple[np.ndarray, np.ndarray]:
    """Quick run function for scHGC analysis."""
    config = yaml.safe_load(open(config_path, 'r')) if config_path else None

    model = scHGC(config=config)
    model.output_dir = output_dir

    embedding, denoised = model.fit_transform(data_path, return_denoised=True)
    model.save()

    np.save(os.path.join(output_dir, 'embedding.npy'), embedding)
    np.save(os.path.join(output_dir, 'denoised.npy'), denoised)

    return embedding, denoised