"""
scHGC Core Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import random
from .layers import ZINBOutputLayer, GraphConvLayer, MultiViewFusion, CrossViewPredictor
import logging
from .utils import setup_logger, set_random_seed

logger = setup_logger('scHGC.model', level='INFO')


class GCNEncoder(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 128,
                 dropout: float = 0.1, use_batch_norm: bool = False, conv_type: str = 'GCN'):
        super(GCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.conv1 = GraphConvLayer(
            in_features=input_dim, out_features=hidden_dim,
            conv_type=conv_type, dropout=dropout, activation=None
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.conv2 = GraphConvLayer(
            in_features=hidden_dim, out_features=output_dim,
            conv_type=conv_type, dropout=0.0, activation=None
        )
        logger.info(f"GCNEncoder initialized: {input_dim} -> {hidden_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        x = self.conv1(x, adj)
        x = F.relu(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        output = self.conv2(x, adj)
        return output


class Path1_GeneAE(nn.Module):
    def __init__(self, latent_dim: int = 128, n_genes: int = 2048,
                 decoder_hidden_dim: int = 512, fusion_type: str = 'concatenate',
                 dropout: float = 0.1):
        super(Path1_GeneAE, self).__init__()
        self.latent_dim = latent_dim
        self.n_genes = n_genes

        self.fusion_layer = MultiViewFusion(
            input_dims=[latent_dim, latent_dim, latent_dim],
            output_dim=latent_dim, fusion_type=fusion_type, dropout=dropout
        )

        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(decoder_hidden_dim),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU()
        )

        self.zinb_output_layer = ZINBOutputLayer(
            input_dim=decoder_hidden_dim, n_genes=n_genes,
            dropout=dropout, use_batch_norm=False
        )

        logger.info(f"Path1_GeneAE initialized: fusion={fusion_type}, decoder_hidden={decoder_hidden_dim}")

    def forward(self, z_knn: torch.Tensor, z_mnn: torch.Tensor,
                z_cluster_mnn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = [z_knn, z_mnn, z_cluster_mnn]
        z_fused = self.fusion_layer(views)
        hidden = self.decoder_mlp(z_fused)
        mu, theta, pi = self.zinb_output_layer(hidden)
        return mu, theta, pi

    def get_fused_embedding(self, z_knn: torch.Tensor, z_mnn: torch.Tensor,
                           z_cluster_mnn: torch.Tensor) -> torch.Tensor:
        views = [z_knn, z_mnn, z_cluster_mnn]
        return self.fusion_layer(views)


"""
修复后的Path2_LatentGraphAE类 - 增强数值稳定性
"""


class Path2_LatentGraphAE(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.1, temperature: float = 1.0):
        super(Path2_LatentGraphAE, self).__init__()
        self.latent_dim = latent_dim
        self.temperature = temperature

        self.cross_view_encoder = CrossViewPredictor(
            input_dim=latent_dim * 2, hidden_dim=hidden_dim,
            output_dim=latent_dim, n_layers=n_layers,
            dropout=dropout, use_residual=False
        )

        logger.info(f"Path2_LatentGraphAE initialized: hidden={hidden_dim}, n_layers={n_layers}")

    def forward(self, z_view1: torch.Tensor, z_view2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Z_final = self.cross_view_encoder(z_view1, z_view2)

        Z_final_norm = F.normalize(Z_final, p=2, dim=1)

        inner_product = torch.mm(Z_final_norm, Z_final_norm.t()) / self.temperature

        inner_product = torch.clamp(inner_product, min=-10, max=10)

        A_hat = torch.sigmoid(inner_product)

        A_hat = torch.clamp(A_hat, min=1e-7, max=1 - 1e-7)

        if torch.isnan(A_hat).any() or torch.isinf(A_hat).any():
            logger.warning("NaN or Inf detected in A_hat, replacing with safe values")
            A_hat = torch.nan_to_num(A_hat, nan=0.5, posinf=1 - 1e-7, neginf=1e-7)
            A_hat = torch.clamp(A_hat, min=1e-7, max=1 - 1e-7)

        return A_hat, Z_final

    def get_embedding(self, z_view1: torch.Tensor, z_view2: torch.Tensor) -> torch.Tensor:
        return self.cross_view_encoder(z_view1, z_view2)


class scHGC(nn.Module):
    def __init__(self, n_genes: int = 2048, hidden_dim: int = 512, latent_dim: int = 128,
                 decoder_hidden_dim: int = 512, fusion_type: str = 'concatenate',
                 conv_type: str = 'GCN', dropout: float = 0.1, random_seed: int = 42):
        super(scHGC, self).__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.random_seed = random_seed

        set_random_seed(random_seed)

        self.gcn_knn = GCNEncoder(
            input_dim=n_genes, hidden_dim=hidden_dim,
            output_dim=latent_dim, dropout=dropout, conv_type=conv_type
        )
        self.gcn_mnn = GCNEncoder(
            input_dim=n_genes, hidden_dim=hidden_dim,
            output_dim=latent_dim, dropout=dropout, conv_type=conv_type
        )
        self.gcn_cluster = GCNEncoder(
            input_dim=n_genes, hidden_dim=hidden_dim,
            output_dim=latent_dim, dropout=dropout, conv_type=conv_type
        )

        self.path1_decoder = Path1_GeneAE(
            latent_dim=latent_dim, n_genes=n_genes,
            decoder_hidden_dim=decoder_hidden_dim,
            fusion_type=fusion_type, dropout=dropout
        )

        self.path2_decoder = Path2_LatentGraphAE(
            latent_dim=latent_dim, hidden_dim=latent_dim,
            n_layers=2, dropout=dropout
        )

        logger.info(f"scHGC model initialized: n_genes={n_genes}, hidden={hidden_dim}, "
                   f"latent={latent_dim}, fusion={fusion_type}")

    def forward(self, x: torch.Tensor, adj_list: List[torch.sparse.FloatTensor],
                training: bool = True, cross_view_indices: Optional[Tuple[int, int, int]] = None) -> Dict[str, torch.Tensor]:
        A_knn, A_mnn, A_cluster_mnn = adj_list

        Z_knn = self.gcn_knn(x, A_knn)
        Z_mnn = self.gcn_mnn(x, A_mnn)
        Z_cluster = self.gcn_cluster(x, A_cluster_mnn)

        mu, theta, pi = self.path1_decoder(Z_knn, Z_mnn, Z_cluster)

        embeddings = [Z_knn, Z_mnn, Z_cluster]
        adjacencies = [A_knn, A_mnn, A_cluster_mnn]
        view_names = ['knn', 'mnn', 'cluster']

        if training:
            if cross_view_indices is None:
                indices = [0, 1, 2]
                random.shuffle(indices)
                input1_idx, input2_idx, target_idx = indices
            else:
                input1_idx, input2_idx, target_idx = cross_view_indices
        else:
            input1_idx, input2_idx, target_idx = 0, 1, 2

        Z_input1 = embeddings[input1_idx]
        Z_input2 = embeddings[input2_idx]
        A_target = adjacencies[target_idx]

        A_recon, Z_final = self.path2_decoder(Z_input1, Z_input2)

        outputs = {
            'mu': mu, 'theta': theta, 'pi': pi,
            'A_recon': A_recon, 'A_target': A_target, 'Z_final': Z_final,
            'Z_knn': Z_knn, 'Z_mnn': Z_mnn, 'Z_cluster': Z_cluster,
            'cross_view_info': {
                'input1': view_names[input1_idx],
                'input2': view_names[input2_idx],
                'target': view_names[target_idx],
                'indices': (input1_idx, input2_idx, target_idx)
            }
        }
        return outputs

    def get_latent_representation(self, x: torch.Tensor, adj_list: List[torch.sparse.FloatTensor],
                                 return_all: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, adj_list, training=False)
            if return_all:
                return {
                    'Z_final': outputs['Z_final'],
                    'Z_knn': outputs['Z_knn'],
                    'Z_mnn': outputs['Z_mnn'],
                    'Z_cluster': outputs['Z_cluster']
                }
            else:
                return outputs['Z_final']

    def get_denoised_expression(self, x: torch.Tensor, adj_list: List[torch.sparse.FloatTensor]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, adj_list, training=False)
            return outputs['mu']

    def encode(self, x: torch.Tensor, adj_list: List[torch.sparse.FloatTensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A_knn, A_mnn, A_cluster_mnn = adj_list
        Z_knn = self.gcn_knn(x, A_knn)
        Z_mnn = self.gcn_mnn(x, A_mnn)
        Z_cluster = self.gcn_cluster(x, A_cluster_mnn)
        return Z_knn, Z_mnn, Z_cluster


def create_scHGC_model(config: Dict) -> scHGC:
    model_config = config.get('model', {})
    model = scHGC(
        n_genes=model_config.get('n_genes', 2048),
        hidden_dim=model_config.get('hidden_dim', 512),
        latent_dim=model_config.get('latent_dim', 128),
        decoder_hidden_dim=model_config.get('decoder_hidden_dim', 512),
        fusion_type=model_config.get('fusion_type', 'concatenate'),
        conv_type=model_config.get('conv_type', 'GCN'),
        dropout=model_config.get('dropout', 0.1),
        random_seed=model_config.get('random_seed', 42)
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: Total params={total_params:,}, Trainable params={trainable_params:,}")

    return model