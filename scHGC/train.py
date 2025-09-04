"""
scHGC Model Training and Evaluation Module
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from typing import Dict, Optional, Tuple, List, Any
import logging
import json
from scipy import sparse
from .loss import scHGCLoss
from .model import scHGC
from .utils import (
    setup_logger,
    sparse_to_torch,
    ensure_dir,
    save_checkpoint,
    save_config
)

logger = setup_logger('scHGC.train', level='INFO')


class scHGCDataset(Dataset):
    def __init__(self, X: np.ndarray, adj_list: List[sparse.csr_matrix], transform: Optional[callable] = None):
        self.X = X
        self.adj_list = adj_list
        self.transform = transform
        self.n_cells = X.shape[0]

        if isinstance(X, np.ndarray):
            self.X = torch.FloatTensor(X)

        self.adj_tensors = []
        for adj in adj_list:
            if sparse.issparse(adj):
                self.adj_tensors.append(sparse_to_torch(adj))
            elif isinstance(adj, np.ndarray):
                adj = torch.FloatTensor(adj)
            self.adj_tensors.append(adj)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.X, self.adj_tensors


class Trainer:
    def __init__(self, model: scHGC, config: Dict[str, Any], data: Dict[str, Any], device: Optional[str] = None):
        self.model = model
        self.config = config
        self.data = data

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        self.train_params = config.get('training', {})
        self.epochs = self.train_params.get('epochs', 200)
        self.batch_size = self.train_params.get('batch_size', None)
        self.learning_rate = self.train_params.get('learning_rate', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 0.0001)
        self.gamma = self.train_params.get('gamma', 0.5)
        self.gradient_clip = self.train_params.get('gradient_clip', 5.0)
        self.early_stop_patience = self.train_params.get('early_stop_patience', 20)

        self.optimizer = self._init_optimizer()
        self.loss_fn = scHGCLoss(
            gamma=self.gamma,
            zinb_ridge=self.train_params.get('zinb_ridge', 0.0),
            graph_pos_weight=self.train_params.get('graph_pos_weight', 1.0)
        )
        self.scheduler = self._init_scheduler()

        self.history = {
            'train_loss': [], 'train_loss_zinb': [], 'train_loss_graph': [],
            'val_loss': [], 'val_loss_zinb': [], 'val_loss_graph': [],
            'learning_rate': []
        }

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

        self.output_dir = config.get('output_dir', 'results')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.log_dir)

        self.start_time = time.time()

    def _init_optimizer(self) -> torch.optim.Optimizer:
        optimizer_type = self.train_params.get('optimizer', 'AdamW')
        if optimizer_type == 'AdamW':
            optimizer = AdamW(self.model.parameters(), lr=self.learning_rate,
                            weight_decay=self.weight_decay, betas=(0.9, 0.999))
        elif optimizer_type == 'Adam':
            optimizer = Adam(self.model.parameters(), lr=self.learning_rate,
                           weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        logger.info(f"Optimizer: {optimizer_type}, LR: {self.learning_rate}")
        return optimizer

    def _init_scheduler(self) -> Optional[object]:
        scheduler_type = self.train_params.get('scheduler', 'ReduceLROnPlateau')
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                         patience=10, min_lr=1e-6)
        elif scheduler_type == 'StepLR':
            scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        elif scheduler_type == 'None':
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        if scheduler:
            logger.info(f"LR Scheduler: {scheduler_type}")
        return scheduler

    def train_epoch(self, batch_x: torch.Tensor, batch_adj_list: List[torch.Tensor]) -> Dict[str, float]:
        self.model.train()

        batch_x = batch_x.to(self.device)
        batch_adj_list = [adj.to(self.device) for adj in batch_adj_list]

        self.optimizer.zero_grad()
        outputs = self.model(batch_x, batch_adj_list, training=True)
        losses = self.loss_fn(batch_x, outputs, return_components=True)
        loss = losses['loss_total']

        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'loss_zinb': losses['loss_zinb'].item(),
            'loss_graph': losses['loss_graph'].item()
        }

    def evaluate(self, batch_x: torch.Tensor, batch_adj_list: List[torch.Tensor]) -> Dict[str, float]:
        self.model.eval()

        with torch.no_grad():
            batch_x = batch_x.to(self.device)
            batch_adj_list = [adj.to(self.device) for adj in batch_adj_list]

            outputs = self.model(batch_x, batch_adj_list, training=False)
            losses = self.loss_fn(batch_x, outputs, return_components=True)

            return {
                'loss': losses['loss_total'].item(),
                'loss_zinb': losses['loss_zinb'].item(),
                'loss_graph': losses['loss_graph'].item()
            }

    def train(self, val_data: Optional[Dict[str, Any]] = None) -> Dict[str, List]:
        logger.info(f"Starting scHGC training for {self.epochs} epochs")

        train_dataset = scHGCDataset(self.data['X'], self.data['adj_list'])
        train_x, train_adj_list = train_dataset[0]

        val_x, val_adj_list = None, None
        if val_data is not None:
            val_dataset = scHGCDataset(val_data['X'], val_data['adj_list'])
            val_x, val_adj_list = val_dataset[0]

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            train_metrics = self.train_epoch(train_x, train_adj_list)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_loss_zinb'].append(train_metrics['loss_zinb'])
            self.history['train_loss_graph'].append(train_metrics['loss_graph'])

            val_metrics = None
            if val_x is not None:
                val_metrics = self.evaluate(val_x, val_adj_list)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_loss_zinb'].append(val_metrics['loss_zinb'])
                self.history['val_loss_graph'].append(val_metrics['loss_graph'])

                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
            else:
                current_train_loss = train_metrics['loss']
                if not self.history['train_loss'] or current_train_loss < min(
                        self.history['train_loss'][:-1] or [float('inf')]):
                    self.save_checkpoint(epoch, is_best=True)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric = val_metrics['loss'] if val_metrics else train_metrics['loss']
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            self._print_progress(epoch, train_metrics, val_metrics, current_lr, epoch_time)

            if self.early_stop_patience > 0 and self.patience_counter >= self.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(epoch, is_best=False)

        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time / 60:.2f} minutes")
        if val_x is not None:
            logger.info(f"Best epoch: {self.best_epoch}, Best val loss: {self.best_val_loss:.4f}")

        self.save_history()
        return self.history

    def _print_progress(self, epoch: int, train_metrics: Dict[str, float],
                       val_metrics: Optional[Dict[str, float]], lr: float, epoch_time: float):
        msg = f"Epoch {epoch + 1}/{self.epochs} ({epoch_time:.1f}s) - "
        msg += f"Train Loss: {train_metrics['loss']:.4f} "
        msg += f"(ZINB: {train_metrics['loss_zinb']:.4f}, Graph: {train_metrics['loss_graph']:.4f})"
        if val_metrics:
            msg += f" - Val Loss: {val_metrics['loss']:.4f} "
            msg += f"(ZINB: {val_metrics['loss_zinb']:.4f}, Graph: {val_metrics['loss_graph']:.4f})"
        msg += f" - LR: {lr:.6f}"
        logger.info(msg)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }

        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            logger.info(f"Saving best model to {path}")
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')

        torch.save(checkpoint, path)

    def save_history(self):
        history_path = os.path.join(self.log_dir, 'training_history.json')
        save_config(self.history, history_path)  # 使用utils的save_config
        logger.info(f"Training history saved to {history_path}")

    def load_checkpoint(self, checkpoint_path: str):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)

        logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")


def train_model(model: scHGC, config: Dict[str, Any], data: Dict[str, Any],
                val_data: Optional[Dict[str, Any]] = None,
                device: Optional[str] = None) -> Tuple[scHGC, Dict[str, List]]:
    trainer = Trainer(model, config, data, device)
    history = trainer.train(val_data)

    best_model_path = os.path.join(trainer.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=trainer.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded best model weights")

    return model, history


def load_trained_model(model_path: str, config: Dict[str, Any],
                      device: Optional[str] = None) -> scHGC:
    from .model import create_scHGC_model

    model = create_scHGC_model(config)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded from {model_path}")
    return model