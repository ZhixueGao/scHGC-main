"""
scHGC Custom Neural Network Layers Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional, Tuple, Union
import numpy as np
import math
import logging
from .utils import (
    setup_logger,
    get_activation_layer,
    create_mlp
)
logger = setup_logger('scHGC.layers', level='INFO')

TORCH_GEOMETRIC_AVAILABLE = False
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
    from torch_geometric.utils import add_self_loops, degree
    TORCH_GEOMETRIC_AVAILABLE = True
    logger.info("torch_geometric is available - using optimized GNN layers")
except ImportError:
    logger.info("torch_geometric not found - using custom implementations")

"""
修复的ZINB输出层 - 增强数值稳定性
"""


class ZINBOutputLayer(nn.Module):
    def __init__(self, input_dim: int, n_genes: int, dropout: float = 0.0,
                 use_batch_norm: bool = False, activation_mu: str = 'softplus',
                 activation_theta: str = 'softplus'):
        super(ZINBOutputLayer, self).__init__()
        self.input_dim = input_dim
        self.n_genes = n_genes
        self.dropout = dropout
        self.activation_mu = activation_mu
        self.activation_theta = activation_theta

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.batch_norm = nn.BatchNorm1d(input_dim) if use_batch_norm else None
        self.linear = nn.Linear(input_dim, n_genes * 3)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        with torch.no_grad():
            n = self.n_genes
            self.linear.bias[:n].fill_(0.01)
            self.linear.bias[n:2 * n].fill_(1.0)
            self.linear.bias[2 * n:].fill_(-3.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)

        output = self.linear(x)

        output = torch.clamp(output, min=-10, max=10)

        mu_raw, theta_raw, pi_raw = torch.chunk(output, 3, dim=-1)

        if self.activation_mu == 'exp':
            mu_raw = torch.clamp(mu_raw, max=5)
            mu = torch.exp(mu_raw)
        elif self.activation_mu == 'softplus':
            mu = F.softplus(mu_raw) + 1e-4
            mu = torch.clamp(mu, max=1e4)
        else:
            mu = F.softplus(mu_raw) + 1e-4
            mu = torch.clamp(mu, max=1e4)

        if self.activation_theta == 'exp':
            theta_raw = torch.clamp(theta_raw, min=-5, max=5)
            theta = torch.exp(theta_raw)
        elif self.activation_theta == 'softplus':
            theta = F.softplus(theta_raw) + 1e-4
            theta = torch.clamp(theta, min=1e-4, max=1e4)
        else:
            theta = F.softplus(theta_raw) + 1e-4
            theta = torch.clamp(theta, min=1e-4, max=1e4)

        pi = torch.sigmoid(pi_raw)

        if torch.isnan(mu).any() or torch.isinf(mu).any():
            logger.warning("NaN or Inf in mu, replacing with safe values")
            mu = torch.nan_to_num(mu, nan=1.0, posinf=1e4, neginf=1e-4)

        if torch.isnan(theta).any() or torch.isinf(theta).any():
            logger.warning("NaN or Inf in theta, replacing with safe values")
            theta = torch.nan_to_num(theta, nan=1.0, posinf=1e4, neginf=1e-4)

        if torch.isnan(pi).any() or torch.isinf(pi).any():
            logger.warning("NaN or Inf in pi, replacing with safe values")
            pi = torch.nan_to_num(pi, nan=0.1, posinf=1.0, neginf=0.0)

        return mu, theta, pi

    def get_params(self, x: torch.Tensor) -> dict:
        mu, theta, pi = self.forward(x)
        return {'mu': mu, 'theta': theta, 'pi': pi}


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, conv_type: str = 'GCN',
                 bias: bool = True, dropout: float = 0.0, activation: Optional[str] = None,
                 use_torch_geometric: bool = True, **kwargs):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv_type = conv_type
        self.dropout = dropout
        self.use_torch_geometric = use_torch_geometric and TORCH_GEOMETRIC_AVAILABLE

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = self._get_activation(activation)

        if self.use_torch_geometric:
            self._init_torch_geometric_layer(conv_type, bias, **kwargs)
        else:
            self._init_custom_layer(bias)

    def _init_torch_geometric_layer(self, conv_type: str, bias: bool, **kwargs):
        if conv_type == 'GCN':
            self.conv = GCNConv(
                self.in_features, self.out_features, bias=bias,
                add_self_loops=kwargs.get('add_self_loops', True),
                normalize=kwargs.get('normalize', True)
            )
        elif conv_type == 'GAT':
            self.conv = GATConv(
                self.in_features, self.out_features,
                heads=kwargs.get('heads', 8),
                concat=kwargs.get('concat', False),
                dropout=self.dropout, bias=bias
            )
        elif conv_type == 'SAGE':
            self.conv = SAGEConv(
                self.in_features, self.out_features,
                normalize=kwargs.get('normalize', False), bias=bias
            )
        elif conv_type == 'GIN':
            mlp = nn.Sequential(
                nn.Linear(self.in_features, self.out_features),
                nn.ReLU(),
                nn.Linear(self.out_features, self.out_features)
            )
            self.conv = GINConv(mlp)
        else:
            raise ValueError(f"Unknown conv_type for torch_geometric: {conv_type}")

    def _init_custom_layer(self, bias: bool):
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if not self.use_torch_geometric:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

    def _get_activation(self, activation: Optional[str]):
        if activation is None:
            return None
        elif activation == 'relu':
            return F.relu
        elif activation == 'elu':
            return F.elu
        elif activation == 'leaky_relu':
            return F.leaky_relu
        elif activation == 'gelu':
            return F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor, edge_index_or_adj: Union[torch.Tensor, torch.sparse.FloatTensor]) -> torch.Tensor:
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)

        if self.use_torch_geometric:
            if edge_index_or_adj.dim() == 2 and edge_index_or_adj.shape[0] == 2:
                edge_index = edge_index_or_adj
            else:
                edge_index = self._adj_to_edge_index(edge_index_or_adj)
            output = self.conv(x, edge_index)
        else:
            if edge_index_or_adj.dim() == 2 and edge_index_or_adj.shape[0] == 2:
                adj = self._edge_index_to_adj(edge_index_or_adj, x.size(0))
            else:
                adj = edge_index_or_adj
            output = self._custom_forward(x, adj)

        if self.activation is not None:
            output = self.activation(output)
        return output

    def _custom_forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        if adj.is_sparse:
            output = torch.sparse.mm(adj, support)
        else:
            output = torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

    def _adj_to_edge_index(self, adj: torch.Tensor) -> torch.Tensor:
        if adj.is_sparse:
            indices = adj._indices()
        else:
            indices = adj.nonzero().t()
        return indices

    def _edge_index_to_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.sparse.FloatTensor:
        values = torch.ones(edge_index.size(1), device=edge_index.device)
        adj = torch.sparse.FloatTensor(edge_index, values, (num_nodes, num_nodes))
        return adj


class MultiViewFusion(nn.Module):
    def __init__(self, input_dims: list, output_dim: int, fusion_type: str = 'concatenate',
                 use_layer_norm: bool = False, dropout: float = 0.0, temperature: float = 1.0):
        super(MultiViewFusion, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.n_views = len(input_dims)
        self.temperature = temperature

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else None

        if fusion_type == 'concatenate':
            total_dim = sum(input_dims)
            self.fusion_layer = nn.Linear(total_dim, output_dim)
        elif fusion_type == 'attention':
            self.attention_weights = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, dim // 4),
                    nn.ReLU(),
                    nn.Linear(dim // 4, 1)
                ) for dim in input_dims
            ])
            self.projections = nn.ModuleList([
                nn.Linear(dim, output_dim) for dim in input_dims
            ])
        elif fusion_type in ['mean', 'max']:
            self.projections = nn.ModuleList([
                nn.Linear(dim, output_dim) for dim in input_dims
            ])
        elif fusion_type == 'gated':
            self.projections = nn.ModuleList([
                nn.Linear(dim, output_dim) for dim in input_dims
            ])
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.Sigmoid()
                ) for dim in input_dims
            ])
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, views: list) -> torch.Tensor:
        assert len(views) == self.n_views, f"Expected {self.n_views} views, got {len(views)}"

        if self.fusion_type == 'concatenate':
            concatenated = torch.cat(views, dim=-1)
            if self.dropout:
                concatenated = self.dropout(concatenated)
            fused = self.fusion_layer(concatenated)

        elif self.fusion_type == 'attention':
            attention_scores = []
            for i, view in enumerate(views):
                score = self.attention_weights[i](view)
                attention_scores.append(score)

            attention_scores = torch.cat(attention_scores, dim=-1)
            attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)

            projected_views = []
            for i, view in enumerate(views):
                projected = self.projections[i](view)
                weighted = projected * attention_weights[:, i:i + 1]
                projected_views.append(weighted)
            fused = sum(projected_views)

        elif self.fusion_type == 'mean':
            projected_views = [self.projections[i](view) for i, view in enumerate(views)]
            fused = torch.stack(projected_views, dim=0).mean(dim=0)

        elif self.fusion_type == 'max':
            projected_views = [self.projections[i](view) for i, view in enumerate(views)]
            fused = torch.stack(projected_views, dim=0).max(dim=0)[0]

        elif self.fusion_type == 'gated':
            gated_views = []
            for i, view in enumerate(views):
                projected = self.projections[i](view)
                gate = self.gates[i](view)
                gated = projected * gate
                gated_views.append(gated)
            fused = sum(gated_views)

        if self.layer_norm is not None:
            fused = self.layer_norm(fused)
        return fused


class CrossViewPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2, dropout: float = 0.1, use_residual: bool = False,
                 activation: str = 'relu'):
        super(CrossViewPredictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual

        layers = []
        current_dim = input_dim

        for i in range(n_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(get_activation_layer(activation))
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        if use_residual and input_dim == output_dim:
            self.residual = True
        else:
            self.residual = False

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z_concat = torch.cat([z1, z2], dim=-1)
        z_pred = self.mlp(z_concat)
        if self.residual:
            z_pred = z_pred + z_concat
        return z_pred


