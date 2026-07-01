"""Shared helpers for training-coupled baseline methods."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from common.model import MLP
from common.seed import set_seed
from common.training import ArrayDataset, TrainingConfig
from methods.base import MethodContext


def training_config_from_context(ctx: MethodContext) -> TrainingConfig:
    names = {f.name for f in fields(TrainingConfig)}
    data = {k: v for k, v in ctx.training_config.items() if k in names}
    cfg = TrainingConfig(**data)
    cfg.save_best_model = bool(ctx.training_config.get("save_best_model", False))
    cfg.save_last_model = bool(ctx.training_config.get("save_last_model", True))
    return cfg


def build_model(ctx: MethodContext, seed_offset: int) -> MLP:
    set_seed(ctx.seed + int(seed_offset))
    return MLP(
        input_dim=ctx.input_dim,
        num_classes=ctx.num_classes,
        hidden_dims=tuple(ctx.model_config.get("hidden_dims", (256, 128))),
        dropout=float(ctx.model_config.get("dropout", 0.2)),
        batch_norm=bool(ctx.model_config.get("batch_norm", True)),
    )


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> optim.Optimizer:
    if cfg.optimizer.lower() != "adamw":
        raise ValueError("Only AdamW is currently supported.")
    return optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: Optional[np.ndarray],
    soft_targets: Optional[np.ndarray],
    cfg: TrainingConfig,
    seed: int,
) -> DataLoader:
    dataset = ArrayDataset(
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        sample_weights=None if sample_weights is None else np.asarray(sample_weights, dtype=np.float32),
        soft_targets=None if soft_targets is None else np.asarray(soft_targets, dtype=np.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        generator=generator,
    )


def train_weighted_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, int]:
    model.train()
    ce = nn.CrossEntropyLoss(reduction="none")
    dataset = loader.dataset
    sample_weights = dataset.sample_weights.to(device) if dataset.sample_weights is not None else None
    soft_targets = dataset.soft_targets.to(device) if dataset.soft_targets is not None else None

    total_loss = 0.0
    total_weight = 0.0
    total_samples = 0
    for x, y, local_idx in loader:
        x = x.to(device)
        y = y.to(device)
        local_idx = local_idx.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if soft_targets is None:
            loss_vec = ce(logits, y)
        else:
            q = soft_targets[local_idx]
            loss_vec = -(q * torch.log_softmax(logits, dim=1)).sum(dim=1)

        if sample_weights is None:
            loss = loss_vec.mean()
            total_loss += float(loss_vec.detach().sum().cpu())
            total_weight += float(len(loss_vec))
        else:
            w = sample_weights[local_idx]
            loss = (loss_vec * w).sum() / w.sum().clamp_min(1e-12)
            total_loss += float((loss_vec.detach() * w.detach()).sum().cpu())
            total_weight += float(w.detach().sum().cpu())
        loss.backward()
        optimizer.step()
        total_samples += int(len(x))

    return total_loss / max(total_weight, 1e-12), total_samples


def model_output_paths(ctx: MethodContext, cfg: TrainingConfig) -> tuple[Optional[str], Optional[str]]:
    output_dir = Path(ctx.output_dir) if ctx.output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    last_model_path = str(output_dir / "last_model.pt") if output_dir is not None and cfg.save_last_model else None
    best_model_path = str(output_dir / "best_model.pt") if output_dir is not None and cfg.save_best_model else None
    return last_model_path, best_model_path
