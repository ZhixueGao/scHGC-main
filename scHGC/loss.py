"""
scHGC Loss Functions Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging
from .utils import (
    setup_logger,
    nan2zero,
    nan2inf,
    nelem,
    reduce_mean
)

logger = setup_logger('scHGC.loss', level='INFO')

"""
修复的ZINB损失函数 - 增强数值稳定性
"""


class ZINBLoss(nn.Module):
    def __init__(self, ridge_lambda: float = 0.0, scale_factor: float = 1.0, eps: float = 1e-10):
        super(ZINBLoss, self).__init__()
        self.ridge_lambda = ridge_lambda
        self.scale_factor = scale_factor
        self.eps = eps

    def forward(self, x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor,
                pi: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        eps = self.eps
        scale_factor = self.scale_factor

        x = x.float()
        x = torch.clamp(x, min=0)

        mu = mu.float() * scale_factor
        mu = torch.clamp(mu, min=eps, max=1e6)

        theta = theta.float()
        theta = torch.clamp(theta, min=eps, max=1e6)

        pi = pi.float()
        pi = torch.clamp(pi, min=eps, max=1 - eps)

        log_theta = torch.log(theta + eps)
        log_mu = torch.log(mu + eps)
        log_mu_theta = torch.log(mu + theta + eps)

        lgamma_x_theta = torch.lgamma(torch.clamp(x + theta, min=1e-8, max=1e8))
        lgamma_x_1 = torch.lgamma(torch.clamp(x + 1, min=1, max=1e8))
        lgamma_theta = torch.lgamma(torch.clamp(theta, min=1e-8, max=1e8))

        nb_term1 = lgamma_x_theta - lgamma_x_1 - lgamma_theta
        nb_term2 = x * (log_mu - log_mu_theta)
        nb_term3 = theta * (log_theta - log_mu_theta)

        nb_term1 = torch.clamp(nb_term1, min=-1e10, max=1e10)
        nb_term2 = torch.clamp(nb_term2, min=-1e10, max=1e10)
        nb_term3 = torch.clamp(nb_term3, min=-1e10, max=1e10)

        nb_nll = -(nb_term1 + nb_term2 + nb_term3)
        nb_nll = torch.clamp(nb_nll, min=-1e10, max=1e10)

        log_1_minus_pi = torch.log(1 - pi + eps)
        nb_case = nb_nll - log_1_minus_pi

        theta_div = theta / (theta + mu + eps)
        theta_div = torch.clamp(theta_div, min=eps, max=1 - eps)

        zero_nb = torch.pow(theta_div, theta)
        zero_nb = torch.clamp(zero_nb, min=eps, max=1 - eps)

        log_pi = torch.log(pi + eps)
        log_1_minus_pi_zero_nb = torch.log((1 - pi) * zero_nb + eps)

        max_val = torch.maximum(log_pi, log_1_minus_pi_zero_nb)
        zero_case = -(max_val + torch.log(
            torch.exp(log_pi - max_val) + torch.exp(log_1_minus_pi_zero_nb - max_val) + eps
        ))

        result = torch.where(x < 1e-8, zero_case, nb_case)

        if self.ridge_lambda > 0:
            ridge_penalty = self.ridge_lambda * torch.square(pi)
            result = result + ridge_penalty

        result = torch.clamp(result, min=-1e10, max=1e10)

        if torch.isnan(result).any() or torch.isinf(result).any():
            logger.warning(f"NaN or Inf in ZINB loss, statistics:")
            logger.warning(f"  mu: min={mu.min():.4f}, max={mu.max():.4f}, mean={mu.mean():.4f}")
            logger.warning(f"  theta: min={theta.min():.4f}, max={theta.max():.4f}, mean={theta.mean():.4f}")
            logger.warning(f"  pi: min={pi.min():.4f}, max={pi.max():.4f}, mean={pi.mean():.4f}")

            result = torch.nan_to_num(result, nan=10.0, posinf=10.0, neginf=-10.0)

        if mask is not None:
            result = result * mask
            nelem = torch.sum(mask) + eps
            loss = torch.sum(result) / nelem
        else:
            loss = torch.mean(result)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Final loss is NaN or Inf, returning default value")
            loss = torch.tensor(10.0, device=loss.device, requires_grad=True)

        return loss


class NBLoss(nn.Module):
    def __init__(self, scale_factor: float = 1.0, eps: float = 1e-10):
        super(NBLoss, self).__init__()
        self.scale_factor = scale_factor
        self.eps = eps

    def forward(self, x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        eps = self.eps
        scale_factor = self.scale_factor

        x = x.float()
        mu = mu.float() * scale_factor
        theta = theta.float()

        theta = torch.clamp(theta, min=eps, max=1e6)

        term1 = torch.lgamma(x + theta) - torch.lgamma(x + 1) - torch.lgamma(theta)
        term2 = x * (torch.log(mu + eps) - torch.log(mu + theta + eps))
        term3 = theta * (torch.log(theta + eps) - torch.log(mu + theta + eps))

        nb_nll = -(term1 + term2 + term3)

        if mask is not None:
            nb_nll = nb_nll * mask
            nelem = torch.sum(mask)
            loss = torch.sum(nb_nll) / (nelem + eps)
        else:
            loss = torch.mean(nb_nll)

        loss = nan2inf(loss)
        return loss



class GraphReconstructionLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, neg_sample_ratio: float = 1.0):
        super(GraphReconstructionLoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_sample_ratio = neg_sample_ratio

    def forward(self, A_recon: torch.Tensor, A_target: torch.Tensor,
                use_sampling: bool = False) -> torch.Tensor:
        if A_target.is_sparse:
            A_target = A_target.to_dense()

        A_recon = torch.clamp(A_recon, min=1e-7, max=1 - 1e-7)

        if torch.isnan(A_recon).any() or torch.isinf(A_recon).any():
            logger.warning("NaN or Inf detected in A_recon, replacing with safe values")
            A_recon = torch.nan_to_num(A_recon, nan=0.5, posinf=1 - 1e-7, neginf=1e-7)
            A_recon = torch.clamp(A_recon, min=1e-7, max=1 - 1e-7)

        if use_sampling and A_target.size(0) > 1000:
            loss = self._sampled_bce_loss(A_recon, A_target)
        else:
            loss = self._full_bce_loss(A_recon, A_target)

        return loss

    def _full_bce_loss(self, A_recon: torch.Tensor, A_target: torch.Tensor) -> torch.Tensor:
        A_recon = torch.clamp(A_recon, min=1e-7, max=1 - 1e-7)

        if self.pos_weight != 1.0:
            weight = torch.ones_like(A_target)
            weight[A_target > 0.5] = self.pos_weight

            loss = -weight * (A_target * torch.log(A_recon + 1e-7) +
                              (1 - A_target) * torch.log(1 - A_recon + 1e-7))
            loss = torch.mean(loss)
        else:
            loss = F.binary_cross_entropy(A_recon, A_target, reduction='mean')

        return loss

    def _sampled_bce_loss(self, A_recon: torch.Tensor, A_target: torch.Tensor) -> torch.Tensor:
        pos_indices = (A_target > 0.5).nonzero(as_tuple=True)
        n_pos = pos_indices[0].size(0)

        if n_pos == 0:
            return torch.tensor(0.01, device=A_recon.device)

        n_neg = int(n_pos * self.neg_sample_ratio)
        neg_indices = self._sample_negative_edges(A_target, n_neg)

        pos_pred = A_recon[pos_indices]
        neg_pred = A_recon[neg_indices]

        pos_label = torch.ones_like(pos_pred)
        neg_label = torch.zeros_like(neg_pred)

        all_pred = torch.cat([pos_pred, neg_pred])
        all_label = torch.cat([pos_label, neg_label])

        all_pred = torch.clamp(all_pred, min=1e-7, max=1 - 1e-7)

        loss = F.binary_cross_entropy(all_pred, all_label, reduction='mean')
        return loss

    def _sample_negative_edges(self, A_target: torch.Tensor, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        neg_mask = (A_target < 0.5)
        neg_indices = neg_mask.nonzero(as_tuple=True)

        n_neg_total = neg_indices[0].size(0)
        if n_neg_total == 0:
            n_nodes = A_target.size(0)
            idx1 = torch.randint(0, n_nodes, (min(n_samples, n_nodes),), device=A_target.device)
            idx2 = torch.randint(0, n_nodes, (min(n_samples, n_nodes),), device=A_target.device)
            return (idx1, idx2)

        if n_samples < n_neg_total:
            perm = torch.randperm(n_neg_total, device=A_target.device)[:n_samples]
            sampled_indices = (neg_indices[0][perm], neg_indices[1][perm])
        else:
            sampled_indices = neg_indices

        return sampled_indices


class scHGCLoss(nn.Module):
    def __init__(self, gamma: float = 0.5, zinb_ridge: float = 0.0, graph_pos_weight: float = 1.0):
        super(scHGCLoss, self).__init__()
        self.gamma = gamma
        self.zinb_loss = ZINBLoss(ridge_lambda=zinb_ridge)
        self.graph_loss = GraphReconstructionLoss(pos_weight=graph_pos_weight)

        logger.info(f"scHGCLoss initialized: gamma={gamma}, zinb_ridge={zinb_ridge}, "
                   f"graph_pos_weight={graph_pos_weight}")

    def forward(self, x: torch.Tensor, outputs: dict, return_components: bool = False):
        loss_zinb = self.zinb_loss(x, outputs['mu'], outputs['theta'], outputs['pi'])
        loss_graph = self.graph_loss(outputs['A_recon'], outputs['A_target'])
        loss_total = loss_zinb + self.gamma * loss_graph

        if return_components:
            return {
                'loss_total': loss_total,
                'loss_zinb': loss_zinb,
                'loss_graph': loss_graph
            }
        else:
            return loss_total


def zinb_loss(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor,
              pi: torch.Tensor, ridge_lambda: float = 0.0) -> torch.Tensor:
    loss_fn = ZINBLoss(ridge_lambda=ridge_lambda)
    return loss_fn(x, mu, theta, pi)


def graph_reconstruction_loss(A_recon: torch.Tensor, A_target: torch.Tensor,
                             pos_weight: float = 1.0) -> torch.Tensor:
    loss_fn = GraphReconstructionLoss(pos_weight=pos_weight)
    return loss_fn(A_recon, A_target)


def compute_total_loss(x: torch.Tensor, outputs: dict, gamma: float = 0.5,
                      zinb_ridge: float = 0.0, graph_pos_weight: float = 1.0) -> dict:
    loss_fn = scHGCLoss(gamma=gamma, zinb_ridge=zinb_ridge, graph_pos_weight=graph_pos_weight)
    return loss_fn(x, outputs, return_components=True)