"""
scHGC Graph Construction Module
"""

import os
import numpy as np
import scanpy as sc
import torch
import hashlib
import logging
from scipy import sparse
from typing import Optional, Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from .utils import (
    CacheManager,
    ensure_dir,
    save_sparse,
    setup_logger
)

logger = setup_logger('scHGC.graph', level='INFO')

def detect_available_backends() -> List[str]:
    backends = []
    try:
        import faiss
        backends.append('faiss_cpu')
        if faiss.get_num_gpus() > 0:
            backends.append('faiss_gpu')
    except ImportError:
        pass
    try:
        import cuml
        backends.append('rapids')
    except ImportError:
        pass
    try:
        import pynndescent
        backends.append('pynndescent')
    except ImportError:
        pass
    backends.append('sklearn')
    return backends


def compute_pca_with_cache(adata: sc.AnnData, n_comps: int = 50, cache_manager: Optional[CacheManager] = None) -> np.ndarray:
    cache_key = f"pca_{n_comps}"
    if cache_manager:
        cached_pca = cache_manager.get(cache_key)
        if cached_pca is not None:
            adata.obsm['X_pca'] = cached_pca
            return cached_pca

    logger.info(f"Computing PCA with {n_comps} components")
    if sparse.issparse(adata.X):
        from sklearn.decomposition import TruncatedSVD
        pca = TruncatedSVD(n_components=n_comps, random_state=42)
        X_pca = pca.fit_transform(adata.X)
        adata.obsm['X_pca'] = X_pca
    else:
        sc.pp.pca(adata, n_comps=n_comps)
        X_pca = adata.obsm['X_pca']

    if cache_manager:
        cache_manager.save(cache_key, X_pca)
    return X_pca


def _knn_faiss(embedding: np.ndarray, k: int, metric: str = 'euclidean', use_gpu: bool = False,
               approximate: bool = False, include_self: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import faiss
    except ImportError:
        return _knn_sklearn(embedding, k, metric, include_self)

    n_samples, dim = embedding.shape
    k_search = k if include_self else k + 1
    embedding = embedding.astype(np.float32)

    if approximate and n_samples > 50000:
        nlist = int(np.sqrt(n_samples))
        quantizer = faiss.IndexFlatL2(dim) if metric == 'euclidean' else faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(embedding)
        logger.info(f"Using approximate FAISS IVF index with {nlist} clusters")
    else:
        if metric == 'euclidean':
            index = faiss.IndexFlatL2(dim)
        elif metric == 'cosine':
            faiss.normalize_L2(embedding)
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        logger.info("Using FAISS GPU acceleration")

    index.add(embedding)
    distances, indices = index.search(embedding, k_search)

    if not include_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    if metric == 'euclidean':
        distances = np.sqrt(distances)

    return distances, indices


def _knn_rapids(embedding: np.ndarray, k: int, metric: str = 'euclidean',
                include_self: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from cuml.neighbors import NearestNeighbors
        logger.info("Using RAPIDS cuML for GPU-accelerated KNN")
    except ImportError:
        return _knn_sklearn(embedding, k, metric, include_self)

    k_search = k if include_self else k + 1
    embedding = embedding.astype(np.float32)

    nbrs = NearestNeighbors(n_neighbors=k_search, metric=metric, algorithm='brute')
    nbrs.fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    if hasattr(distances, 'get'):
        distances = distances.get()
        indices = indices.get()

    if not include_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    return distances, indices


def _knn_pynndescent(embedding: np.ndarray, k: int, metric: str = 'euclidean',
                     include_self: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import pynndescent
        logger.info("Using PyNNDescent for approximate KNN")
    except ImportError:
        return _knn_sklearn(embedding, k, metric, include_self)

    k_search = k if include_self else k + 1
    index = pynndescent.NNDescent(embedding, n_neighbors=k_search, metric=metric, n_jobs=-1, verbose=False)
    indices, distances = index.query(embedding, k=k_search)

    if not include_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    return distances, indices


def _knn_sklearn(embedding: np.ndarray, k: int, metric: str = 'euclidean',
                 include_self: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    k_search = k if include_self else k + 1
    nbrs = NearestNeighbors(n_neighbors=k_search, metric=metric, algorithm='auto', n_jobs=-1)
    nbrs.fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    if not include_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    return distances, indices


def build_knn_from_embedding(embedding: np.ndarray, k: int = 15, metric: str = 'euclidean',
                             backend: str = 'auto', include_self: bool = False,
                             approximate: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = embedding.shape[0]

    if backend == 'auto':
        available = detect_available_backends()
        priority = ['faiss_gpu', 'rapids', 'faiss_cpu', 'pynndescent', 'sklearn']
        for b in priority:
            if b in available:
                backend = b
                break
        logger.info(f"Auto-selected backend: {backend}")

    if backend.startswith('faiss'):
        return _knn_faiss(embedding, k, metric, use_gpu=(backend == 'faiss_gpu'),
                         approximate=approximate, include_self=include_self)
    elif backend == 'rapids':
        return _knn_rapids(embedding, k, metric, include_self)
    elif backend == 'pynndescent':
        return _knn_pynndescent(embedding, k, metric, include_self)
    else:
        return _knn_sklearn(embedding, k, metric, include_self)


def calculate_weights(distances: np.ndarray, method: str = 'gaussian', **kwargs) -> np.ndarray:
    if method == 'uniform':
        return np.ones_like(distances)
    elif method == 'inverse':
        return 1.0 / (1.0 + distances)
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', distances.mean())
        return np.exp(-(distances ** 2) / (2 * sigma ** 2))
    elif method == 'adaptive_gaussian':
        if len(distances.shape) == 2:
            sigma = distances[:, -1:] + 1e-8
            return np.exp(-(distances ** 2) / (2 * sigma ** 2))
        else:
            sigma = distances.mean()
            return np.exp(-(distances ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown weight method: {method}")


def batch_correction(adata: sc.AnnData, batch_key: str, method: str = 'harmony',
                     n_pca: int = 50, cache_manager: Optional[CacheManager] = None,
                     **kwargs) -> Tuple[np.ndarray, str]:
    if 'X_pca' not in adata.obsm:
        X_pca = compute_pca_with_cache(adata, n_pca, cache_manager)
    else:
        X_pca = adata.obsm['X_pca']

    if not batch_key or batch_key not in adata.obs.columns:
        return X_pca, 'X_pca'

    n_batches = adata.obs[batch_key].nunique()
    if n_batches < 2:
        return X_pca, 'X_pca'

    logger.info(f"Found {n_batches} batches, applying {method} correction")

    try:
        if method == 'harmony':
            import scanpy.external as sce
            sce.pp.harmony_integrate(adata, batch_key, adjusted_basis='X_pca')
            return adata.obsm['X_pca_harmony'], 'X_pca_harmony'
        elif method == 'mnn':
            from scanpy.external import pp
            pp.mnn_correct(adata, batch_key=batch_key, n_neighbors=kwargs.get('n_neighbors', 20))
            return adata.obsm['X_mnn'], 'X_mnn'
        elif method == 'combat':
            sc.pp.combat(adata, key=batch_key)
            X_pca = compute_pca_with_cache(adata, n_pca, cache_manager, use_cache=False)
            return X_pca, 'X_pca_combat'
        elif method == 'bbknn':
            import bbknn
            bbknn.bbknn(adata, batch_key=batch_key, n_pcs=n_pca)
            return X_pca, 'X_pca_bbknn'
        elif method == 'scanorama':
            import scanpy.external as sce
            sce.pp.scanorama_integrate(adata, batch_key)
            return adata.obsm['X_scanorama'], 'X_scanorama'
    except Exception as e:
        logger.warning(f"{method} correction failed: {e}, using simple centering")

    X_pca_corrected = X_pca.copy()
    for batch in adata.obs[batch_key].unique():
        mask = adata.obs[batch_key] == batch
        X_pca_corrected[mask] -= X_pca_corrected[mask].mean(axis=0)
        X_pca_corrected[mask] /= (X_pca_corrected[mask].std(axis=0) + 1e-8)
    return X_pca_corrected, 'X_pca_centered'


def build_knn_graph(adata: sc.AnnData, k: int = 15, n_pca: int = 100, metric: str = 'euclidean',
                    backend: str = 'auto', weight_method: str = 'gaussian', symmetrize: bool = True,
                    cache_manager: Optional[CacheManager] = None, **kwargs) -> sparse.csr_matrix:
    n_cells = adata.n_obs
    use_rep = kwargs.get('use_rep')

    if use_rep and use_rep in adata.obsm:
        X = adata.obsm[use_rep]
    else:
        if n_pca > 0:
            X = compute_pca_with_cache(adata, n_pca, cache_manager)
        else:
            X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X

    distances, indices = build_knn_from_embedding(X, k, metric, backend, approximate=kwargs.get('approximate', False))
    weights = calculate_weights(distances, weight_method, **kwargs)

    row_idx = np.repeat(np.arange(n_cells), k)
    col_idx = indices.flatten()
    weights_flat = weights.flatten()

    A_knn = sparse.csr_matrix((weights_flat, (row_idx, col_idx)), shape=(n_cells, n_cells))

    if symmetrize:
        A_knn = (A_knn + A_knn.T) / 2

    logger.info(f"KNN graph: {n_cells} nodes, {A_knn.nnz} edges")
    return A_knn


def build_mnn_graph(adata: sc.AnnData, batch_key: Optional[str] = None, batch_correction_method: str = 'harmony',
                    k: int = 20, n_pca: int = 50, backend: str = 'auto', weight_method: str = 'inverse',
                    cache_manager: Optional[CacheManager] = None, **kwargs) -> Tuple[sparse.csr_matrix, sc.AnnData]:
    n_cells = adata.n_obs

    if batch_key:
        X_integrated, rep_key = batch_correction(adata, batch_key, batch_correction_method, n_pca, cache_manager, **kwargs)
    else:
        X_integrated = compute_pca_with_cache(adata, n_pca, cache_manager)
        rep_key = 'X_pca'

    distances, indices = build_knn_from_embedding(X_integrated, k, 'euclidean', backend,
                                                  approximate=kwargs.get('approximate', False))
    weights = calculate_weights(distances, weight_method, **kwargs)

    row_idx = np.repeat(np.arange(n_cells), k)
    col_idx = indices.flatten()
    weights_flat = weights.flatten()

    A_mnn = sparse.csr_matrix((weights_flat, (row_idx, col_idx)), shape=(n_cells, n_cells))
    A_mnn = (A_mnn + A_mnn.T) / 2

    adata.obsp['mnn_connectivities'] = A_mnn
    adata.uns['mnn_rep_key'] = rep_key

    logger.info(f"MNN graph: {n_cells} nodes, {A_mnn.nnz} edges")
    return A_mnn, adata


def build_cluster_mnn_graph(adata: sc.AnnData, A_mnn: sparse.csr_matrix, resolution: float = 1.0,
                            algorithm: str = 'louvain', min_cluster_size: int = 10,
                            **kwargs) -> Tuple[sparse.csr_matrix, sc.AnnData]:
    n_cells = adata.n_obs
    logger.info("Computing neighbor graph for clustering")
    use_rep = adata.uns.get('mnn_rep_key', 'X_pca')
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15)
    if 'connectivities' not in adata.obsp:
        adata.obsp['connectivities'] = A_mnn

    if algorithm == 'louvain':
        sc.tl.louvain(adata, resolution=resolution, random_state=42)
        cluster_key = 'louvain'
    else:
        sc.tl.leiden(adata, resolution=resolution, random_state=42)
        cluster_key = 'leiden'

    cluster_labels = adata.obs[cluster_key].values
    cluster_sizes = adata.obs[cluster_key].value_counts()
    valid_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index

    rows, cols = A_mnn.nonzero()
    data = A_mnn.data

    cluster_match = cluster_labels[rows] == cluster_labels[cols]
    valid_mask = np.isin(cluster_labels[rows], valid_clusters)
    mask = cluster_match & valid_mask

    A_cluster_mnn = sparse.csr_matrix((data[mask], (rows[mask], cols[mask])), shape=(n_cells, n_cells))
    A_cluster_mnn = (A_cluster_mnn + A_cluster_mnn.T) / 2

    adata.obsp['cluster_mnn_connectivities'] = A_cluster_mnn

    logger.info(f"Cluster-MNN graph: {n_cells} nodes, {A_cluster_mnn.nnz} edges")
    return A_cluster_mnn, adata


def normalize_adjacency(A: sparse.csr_matrix, method: str = 'symmetric') -> sparse.csr_matrix:
    if method == 'none':
        return A

    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1

    if method == 'symmetric':
        d_sqrt_inv = sparse.diags(1.0 / np.sqrt(degrees))
        return (d_sqrt_inv @ A @ d_sqrt_inv).tocsr()
    elif method == 'random_walk':
        d_inv = sparse.diags(1.0 / degrees)
        return (d_inv @ A).tocsr()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def build_graphs_parallel(adata: sc.AnnData, config: Dict[str, Any],
                         cache_manager: Optional[CacheManager] = None) -> Dict[str, sparse.csr_matrix]:
    knn_config = config.get('knn', {})
    mnn_config = config.get('mnn', {})
    cluster_config = config.get('cluster', {})

    knn_config['cache_manager'] = cache_manager
    mnn_config['cache_manager'] = cache_manager

    n_workers = max(1, mp.cpu_count() - 1)
    with ThreadPoolExecutor(max_workers=min(2, n_workers)) as executor:
        future_knn = executor.submit(build_knn_graph, adata.copy(), **knn_config)
        future_mnn = executor.submit(build_mnn_graph, adata.copy(), **mnn_config)

        A_knn = future_knn.result()
        A_mnn, adata = future_mnn.result()

    A_cluster_mnn, adata = build_cluster_mnn_graph(adata, A_mnn, **cluster_config)

    if config.get('normalize', True):
        normalize_method = config.get('normalize_method', 'symmetric')
        A_knn = normalize_adjacency(A_knn, normalize_method)
        A_mnn = normalize_adjacency(A_mnn, normalize_method)
        A_cluster_mnn = normalize_adjacency(A_cluster_mnn, normalize_method)

    return {'knn': A_knn, 'mnn': A_mnn, 'cluster_mnn': A_cluster_mnn}


def run_graph_construction_optimized(config: Dict[str, Any], adata_path: Optional[str] = None,
                                    adata: Optional[sc.AnnData] = None, output_dir: str = 'data/processed',
                                    use_parallel: bool = True, use_cache: bool = True) -> Tuple[Dict[str, sparse.csr_matrix], Dict[str, str]]:
    if adata is None:
        if adata_path is None:
            raise ValueError("Either adata or adata_path must be provided")
        logger.info(f"Loading data from {adata_path}")
        adata = sc.read_h5ad(adata_path)

    cache_manager = CacheManager(os.path.join(output_dir, 'cache')) if use_cache else None

    if use_parallel:
        graphs = build_graphs_parallel(adata, config, cache_manager)
    else:
        knn_config = config.get('knn', {})
        mnn_config = config.get('mnn', {})
        cluster_config = config.get('cluster', {})

        A_knn = build_knn_graph(adata, cache_manager=cache_manager, **knn_config)
        A_mnn, adata = build_mnn_graph(adata, cache_manager=cache_manager, **mnn_config)
        A_cluster_mnn, adata = build_cluster_mnn_graph(adata, A_mnn, **cluster_config)

        if config.get('normalize', True):
            normalize_method = config.get('normalize_method', 'symmetric')
            A_knn = normalize_adjacency(A_knn, normalize_method)
            A_mnn = normalize_adjacency(A_mnn, normalize_method)
            A_cluster_mnn = normalize_adjacency(A_cluster_mnn, normalize_method)

        graphs = {'knn': A_knn, 'mnn': A_mnn, 'cluster_mnn': A_cluster_mnn}

    paths = {}
    if output_dir:
        ensure_dir(output_dir)
        for name, A in graphs.items():
            path = os.path.join(output_dir, f'graph_{name}.npz')
            save_sparse(A, path)
            paths[name] = path
            logger.info(f"Saved {name} graph to {path}")

        adata_path = os.path.join(output_dir, 'adata_with_graphs.h5ad')
        adata.write_h5ad(adata_path)
        paths['adata'] = adata_path

    return graphs, paths


def run_graph_construction(config: Dict[str, Any], adata_path: Optional[str] = None,
                          adata: Optional[sc.AnnData] = None, output_dir: str = 'data/processed') -> Tuple[Dict[str, sparse.csr_matrix], Dict[str, str]]:
    return run_graph_construction_optimized(config, adata_path, adata, output_dir, use_parallel=True, use_cache=True)