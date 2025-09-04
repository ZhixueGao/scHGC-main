"""
scHGC Data Preprocessing Module
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from typing import Optional, Union, Tuple, Dict, Any
import warnings
import logging
from .utils import setup_logger, ensure_dir, save_numpy

logger = setup_logger('scHGC.preprocess', level='INFO')


def load_data(data_path: str, first_column_names: bool = True,
              delimiter: str = ',', transpose: bool = False) -> sc.AnnData:
    logger.info(f"Loading data from {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    file_ext = os.path.splitext(data_path)[1].lower()

    if file_ext == '.h5ad':
        adata = sc.read_h5ad(data_path)
    elif file_ext == '.h5':
        adata = sc.read_10x_h5(data_path)
    elif file_ext in ['.csv', '.txt', '.tsv']:
        if file_ext == '.csv':
            delimiter = ','
        elif file_ext == '.tsv':
            delimiter = '\t'

        data = pd.read_csv(data_path, delimiter=delimiter,
                          index_col=0 if first_column_names else None)
        if transpose:
            data = data.T

        adata = sc.AnnData(data)
        adata.obs_names = data.index.astype(str)
        adata.var_names = data.columns.astype(str)
    elif file_ext == '.mtx':
        adata = sc.read_mtx(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    adata.var_names_make_unique()
    adata.raw = adata.copy()

    logger.info(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")
    return adata


def quality_control(adata: sc.AnnData, min_genes: int = 200, min_cells: int = 3,
                    max_genes: int = 2500, max_pct_mito: float = 0.05,
                    remove_mito: bool = True) -> sc.AnnData:
    logger.info("Starting quality control")
    initial_shape = adata.shape

    mito_genes = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1 if sparse.issparse(adata.X) else (adata.X > 0).sum(axis=1)
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1 if sparse.issparse(adata.X) else adata.X.sum(axis=1)
    adata.obs['pct_mito'] = adata[:, mito_genes].X.sum(axis=1).A1 / adata.obs['n_counts'] if mito_genes.sum() > 0 else 0

    logger.info(f"Filtering cells: min_genes={min_genes}, max_genes={max_genes}, max_pct_mito={max_pct_mito}")
    cell_filter = ((adata.obs['n_genes'] >= min_genes) &
                   (adata.obs['n_genes'] <= max_genes) &
                   (adata.obs['pct_mito'] <= max_pct_mito))
    adata = adata[cell_filter, :]

    logger.info(f"Filtering genes: min_cells={min_cells}")
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if remove_mito and mito_genes.sum() > 0:
        adata = adata[:, ~mito_genes]
        logger.info(f"Removed {mito_genes.sum()} mitochondrial genes")

    logger.info(f"Quality control: {initial_shape} -> {adata.shape}")
    return adata


def normalize_and_select_hvg(adata: sc.AnnData, target_sum: float = 1e4,
                             n_top_genes: int = 2048, flavor: str = 'seurat',
                             min_mean: float = 0.0125, max_mean: float = 3,
                             min_disp: float = 0.5, log_transform: bool = True,
                             scale: bool = True, max_value: Optional[float] = 10) -> Tuple[np.ndarray, sc.AnnData]:
    logger.info("Starting normalization and HVG selection")

    logger.info(f"Normalizing to target sum: {target_sum}")
    sc.pp.normalize_total(adata, target_sum=target_sum)

    if log_transform:
        logger.info("Applying log1p transformation")
        sc.pp.log1p(adata)

    logger.info(f"Selecting {n_top_genes} highly variable genes using {flavor} method")
    if flavor == 'seurat_v3':
        if 'counts' in adata.layers:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,
                                       flavor=flavor, layer='counts')
        else:
            warnings.warn("Seurat v3 requires raw counts, falling back to seurat method")
            flavor = 'seurat'

    if flavor in ['seurat', 'cell_ranger']:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor,
                                   min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)

    adata_full = adata.copy()
    adata = adata[:, adata.var['highly_variable']]
    logger.info(f"Selected {adata.n_vars} highly variable genes")

    if scale:
        logger.info("Scaling data to zero mean and unit variance")
        sc.pp.scale(adata, max_value=max_value)

    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    X = X.astype(np.float32)

    adata_full.uns['hvg_indices'] = np.where(adata_full.var['highly_variable'])[0]
    adata_full.uns['hvg_names'] = adata.var_names.tolist()

    return X, adata_full


def compute_size_factors(adata: sc.AnnData, method: str = 'deseq') -> np.ndarray:
    if method == 'median':
        counts = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        size_factors = np.median(counts / np.median(counts, axis=0), axis=1)
    elif method == 'deseq':
        counts = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        geo_means = np.exp(np.log(counts + 1).mean(axis=0))
        size_factors = np.median(counts / geo_means, axis=1)
    else:
        size_factors = adata.X.sum(axis=1).A1 if sparse.issparse(adata.X) else adata.X.sum(axis=1)

    size_factors = np.maximum(size_factors, 1e-6)
    return size_factors


def save_processed_data(X: np.ndarray, adata: sc.AnnData,
                       output_dir: str = 'data/processed',
                       prefix: str = 'processed') -> Dict[str, str]:
    ensure_dir(output_dir)
    paths = {}

    tensor_path = os.path.join(output_dir, f'{prefix}_features.pt')
    torch.save(torch.FloatTensor(X), tensor_path)
    paths['features'] = tensor_path
    logger.info(f"Saved feature matrix to {tensor_path}")

    numpy_path = os.path.join(output_dir, f'{prefix}_features.npy')
    save_numpy(X, numpy_path)
    paths['features_numpy'] = numpy_path

    adata_path = os.path.join(output_dir, f'{prefix}_adata.h5ad')
    adata.write_h5ad(adata_path)
    paths['adata'] = adata_path
    logger.info(f"Saved AnnData object to {adata_path}")

    metadata = {
        'n_cells': adata.n_obs,
        'n_genes': X.shape[1],
        'n_total_genes': adata.n_vars,
        'hvg_indices': adata.uns.get('hvg_indices', []).tolist(),
        'hvg_names': adata.uns.get('hvg_names', []),
        'cell_names': adata.obs_names.tolist(),
    }

    metadata_path = os.path.join(output_dir, f'{prefix}_metadata.npz')
    np.savez(metadata_path, **metadata)
    paths['metadata'] = metadata_path
    logger.info(f"Saved metadata to {metadata_path}")

    return paths


def run_preprocess(config: Dict[str, Any], data_path: Optional[str] = None,
                  adata: Optional[sc.AnnData] = None,
                  output_dir: Optional[str] = 'data/processed',
                  return_adata: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, sc.AnnData]]:
    logger.info("Starting scHGC data preprocessing pipeline")

    if adata is None:
        if data_path is None:
            raise ValueError("Either data_path or adata must be provided")
        adata = load_data(data_path)

    qc_params = config.get('quality_control', {})
    adata = quality_control(
        adata,
        min_genes=qc_params.get('min_genes', 200),
        min_cells=qc_params.get('min_cells', 3),
        max_genes=qc_params.get('max_genes', 2500),
        max_pct_mito=qc_params.get('max_pct_mito', 0.05),
        remove_mito=qc_params.get('remove_mito', True)
    )

    norm_params = config.get('normalization', {})
    X, adata = normalize_and_select_hvg(
        adata,
        target_sum=norm_params.get('target_sum', 1e4),
        n_top_genes=norm_params.get('n_top_genes', 2048),
        flavor=norm_params.get('flavor', 'seurat'),
        min_mean=norm_params.get('min_mean', 0.0125),
        max_mean=norm_params.get('max_mean', 3),
        min_disp=norm_params.get('min_disp', 0.5),
        log_transform=norm_params.get('log_transform', True),
        scale=norm_params.get('scale', True),
        max_value=norm_params.get('max_value', 10)
    )

    if output_dir:
        save_processed_data(X, adata, output_dir)

    logger.info(f"Preprocessing completed! Final data shape: {X.shape}")

    if return_adata:
        return X, adata
    return X


def quick_preprocess(data_path: str, n_top_genes: int = 2048,
                    min_cells: int = 3, min_genes: int = 200) -> np.ndarray:
    config = {
        'quality_control': {
            'min_genes': min_genes,
            'min_cells': min_cells,
        },
        'normalization': {
            'n_top_genes': n_top_genes,
            'log_transform': True,
            'scale': True
        }
    }
    return run_preprocess(config, data_path=data_path, output_dir=None)