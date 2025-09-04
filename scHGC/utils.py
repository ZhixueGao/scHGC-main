"""
scHGC Utility Functions Module
"""

import os
import torch
import torch.nn as nn
import numpy as np
import random
import yaml
import json
import hashlib
import logging
from scipy import sparse
from typing import Optional, Dict, Any, Union, Tuple, List
from datetime import datetime
import time


# ========== Data Conversion Tools ==========

def sparse_to_torch(adj: sparse.csr_matrix) -> torch.sparse.FloatTensor:
    adj_coo = adj.tocoo()
    indices = torch.LongTensor(np.vstack([adj_coo.row, adj_coo.col]))
    values = torch.FloatTensor(adj_coo.data)
    return torch.sparse.FloatTensor(indices, values, adj_coo.shape)


def torch_to_sparse(adj_torch: torch.sparse.FloatTensor) -> sparse.csr_matrix:
    indices = adj_torch._indices().cpu().numpy()
    values = adj_torch._values().cpu().numpy()
    shape = adj_torch.shape
    return sparse.csr_matrix((values, (indices[0], indices[1])), shape=shape)


def adj_to_edge_index(adj: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    if adj.is_sparse:
        indices = adj._indices()
    else:
        indices = adj.nonzero().t()
    return indices


def edge_index_to_adj(edge_index: torch.Tensor, num_nodes: int,
                      values: Optional[torch.Tensor] = None) -> torch.sparse.FloatTensor:
    if values is None:
        values = torch.ones(edge_index.size(1), device=edge_index.device)
    return torch.sparse.FloatTensor(edge_index, values, (num_nodes, num_nodes))


def ensure_tensor(data: Union[np.ndarray, torch.Tensor, sparse.csr_matrix],
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if sparse.issparse(data):
        data = data.toarray()
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data.to(dtype)


# ========== Numerical Stability Tools ==========

def nan2zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def nan2inf(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def nelem(x: torch.Tensor) -> torch.Tensor:
    n = torch.sum(~torch.isnan(x).float())
    return torch.where(n == 0., torch.ones_like(n), n)


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    n = nelem(x)
    x = nan2zero(x)
    return torch.sum(x) / n


def clip_values(x: torch.Tensor, min_val: float = -10, max_val: float = 10) -> torch.Tensor:
    return torch.clamp(x, min=min_val, max=max_val)


# ========== Cache Management ==========

class CacheManager:
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = {}

    def _get_hash(self, data: Any, params: dict) -> str:
        if isinstance(data, (np.ndarray, torch.Tensor)):
            data_bytes = data.cpu().numpy().tobytes() if isinstance(data, torch.Tensor) else data.tobytes()
        else:
            data_bytes = str(data).encode()
        data_hash = hashlib.md5(data_bytes).hexdigest()[:8]
        param_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()[:8]
        return f"{data_hash}_{param_hash}"

    def get(self, key: str, data: Any = None, params: Dict = None) -> Optional[Any]:
        if data is not None and params is not None:
            key = self._get_hash(data, params)

        if key in self.memory_cache:
            return self.memory_cache[key]

        cache_file = os.path.join(self.cache_dir, f'{key}.npz')
        if os.path.exists(cache_file):
            cached_data = np.load(cache_file, allow_pickle=True)
            if 'data' in cached_data:
                result = cached_data['data']
            else:
                result = cached_data
            self.memory_cache[key] = result
            return result
        return None

    def save(self, key: str, value: Any, data: Any = None, params: Dict = None):
        if data is not None and params is not None:
            key = self._get_hash(data, params)

        self.memory_cache[key] = value
        cache_file = os.path.join(self.cache_dir, f'{key}.npz')

        if isinstance(value, (np.ndarray, torch.Tensor)):
            value_np = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            np.savez_compressed(cache_file, data=value_np)
        else:
            np.savez_compressed(cache_file, data=np.array(value, dtype=object))

    def clear(self):
        self.memory_cache.clear()
        for file in os.listdir(self.cache_dir):
            if file.endswith('.npz'):
                os.remove(os.path.join(self.cache_dir, file))


# ========== Device Management ==========

def get_device(device: Optional[str] = None) -> torch.device:
    if device == 'auto' or device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def get_gpu_info() -> Dict[str, Any]:
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': []
    }

    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
        for i in range(info['device_count']):
            info['devices'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i)
            })
    return info


def move_to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    return data


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ========== Configuration Management ==========

def load_config(config_path: str) -> Dict[str, Any]:
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")


def save_config(config: Dict[str, Any], path: str):
    if path.endswith('.yaml') or path.endswith('.yml'):
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {path}")


def merge_configs(default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    result = default.copy()
    for key, value in custom.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def get_default_config() -> Dict[str, Any]:
    return {
        'preprocessing': {
            'quality_control': {
                'min_genes': 200,
                'min_cells': 3,
                'max_genes': 2500,
                'max_pct_mito': 0.05
            },
            'normalization': {
                'target_sum': 1e4,
                'n_top_genes': 2048,
                'log_transform': True,
                'scale': True
            }
        },
        'graph_construction': {
            'knn': {'k': 15, 'n_pca': 100, 'weight_method': 'gaussian'},
            'mnn': {'k': 20, 'n_pca': 50, 'weight_method': 'inverse'},
            'cluster': {'resolution': 1.0, 'algorithm': 'louvain'},
            'normalize': True,
            'normalize_method': 'symmetric'
        },
        'model': {
            'n_genes': 2048,
            'hidden_dim': 512,
            'latent_dim': 128,
            'decoder_hidden_dim': 512,
            'fusion_type': 'concatenate',
            'conv_type': 'GCN',
            'dropout': 0.1
        },
        'training': {
            'epochs': 200,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'gamma': 0.5,
            'early_stop_patience': 20,
            'optimizer': 'AdamW',
            'scheduler': 'ReduceLROnPlateau'
        }
    }


# ========== Logging and Progress ==========

def setup_logger(name: str = 'scHGC', level: str = 'INFO',
                 log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def start(self):
        self.start_time = time.time()

    def stop(self) -> float:
        if self.start_time is None:
            return self.elapsed
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ========== File IO Tools ==========

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_numpy(data: Union[np.ndarray, torch.Tensor], path: str):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    np.save(path, data)


def load_numpy(path: str, to_torch: bool = False) -> Union[np.ndarray, torch.Tensor]:
    data = np.load(path)
    if to_torch:
        return torch.from_numpy(data)
    return data


def save_sparse(matrix: sparse.csr_matrix, path: str):
    sparse.save_npz(path, matrix)


def load_sparse(path: str, to_torch: bool = False) -> Union[sparse.csr_matrix, torch.sparse.FloatTensor]:
    matrix = sparse.load_npz(path)
    if to_torch:
        return sparse_to_torch(matrix)
    return matrix


def save_checkpoint(state: Dict[str, Any], path: str, is_best: bool = False):
    torch.save(state, path)
    if is_best:
        best_path = path.replace('.pth', '_best.pth')
        torch.save(state, best_path)


def load_checkpoint(path: str, device: Optional[str] = None) -> Dict[str, Any]:
    device = get_device(device)
    return torch.load(path, map_location=device)


def get_file_size(path: str) -> str:
    size = os.path.getsize(path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


# ========== Random Seed Management ==========

def set_random_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RandomState:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.np_state = None
        self.random_state = None
        self.torch_state = None
        self.cuda_state = None

    def __enter__(self):
        self.np_state = np.random.get_state()
        self.random_state = random.getstate()
        self.torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_state = torch.cuda.get_rng_state_all()
        set_random_seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.np_state)
        random.setstate(self.random_state)
        torch.set_rng_state(self.torch_state)
        if self.cuda_state is not None:
            torch.cuda.set_rng_state_all(self.cuda_state)


# ========== Network Building Tools ==========

def get_activation_layer(activation: str) -> nn.Module:
    activations = {
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'leaky_relu': nn.LeakyReLU(0.2),
        'prelu': nn.PReLU(),
        'gelu': nn.GELU(),
        'selu': nn.SELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'softplus': nn.Softplus(),
        'swish': nn.SiLU()
    }
    if activation.lower() not in activations:
        raise ValueError(f"Unknown activation: {activation}")
    return activations[activation.lower()]


def create_mlp(input_dim: int, hidden_dims: List[int], output_dim: int,
               activation: str = 'relu', dropout: float = 0.0,
               use_batch_norm: bool = False, use_layer_norm: bool = False,
               final_activation: Optional[str] = None) -> nn.Sequential:
    layers = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(get_activation_layer(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))
    if final_activation:
        layers.append(get_activation_layer(final_activation))

    return nn.Sequential(*layers)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = True