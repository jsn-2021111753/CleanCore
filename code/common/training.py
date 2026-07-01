"""Shared PyTorch training loop for downstream MLP classifiers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class ArrayDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        soft_targets: Optional[np.ndarray] = None,
    ) -> None:
        self.X = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y, dtype=np.int64), dtype=torch.long)
        self.indices = torch.arange(len(self.y), dtype=torch.long)
        self.sample_weights = None if sample_weights is None else torch.tensor(sample_weights, dtype=torch.float32)
        self.soft_targets = None if soft_targets is None else torch.tensor(soft_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index], self.indices[index]


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 300
    early_stop: bool = False
    early_stop_patience: int = 50
    early_stop_min_delta: float = 1e-4
    device: str = "cpu"
    num_workers: int = 0
    save_last_model: bool = True
    save_best_model: bool = False


@dataclass
class TrainingResult:
    history: List[Dict[str, float]]
    epochs_ran: int
    best_epoch: Optional[int]
    best_train_loss: Optional[float]
    train_time_sec: float
    last_model_path: Optional[str]
    best_model_path: Optional[str]


def _build_optimizer(model: nn.Module, cfg: TrainingConfig) -> optim.Optimizer:
    if cfg.optimizer.lower() != "adamw":
        raise ValueError("Only AdamW is currently supported.")
    return optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    sample_weights: Optional[torch.Tensor],
    soft_targets: Optional[torch.Tensor],
) -> tuple[float, int]:
    model.train()
    ce = nn.CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_weight = 0.0
    total_samples = 0

    for x, y, local_idx in loader:
        x = x.to(device)
        y = y.to(device)
        local_idx = local_idx.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if soft_targets is not None:
            q = soft_targets[local_idx].to(device)
            log_probs = torch.log_softmax(logits, dim=1)
            loss_vec = -(q * log_probs).sum(dim=1)
        else:
            loss_vec = ce(logits, y)

        if sample_weights is not None:
            weights = sample_weights[local_idx].to(device)
            loss = (loss_vec * weights).sum() / weights.sum().clamp_min(1e-12)
            total_loss += float((loss_vec.detach() * weights.detach()).sum().cpu())
            total_weight += float(weights.detach().sum().cpu())
        else:
            loss = loss_vec.mean()
            total_loss += float(loss_vec.detach().sum().cpu())
            total_weight += float(len(loss_vec))

        loss.backward()
        optimizer.step()
        total_samples += int(len(y))

    return total_loss / max(total_weight, 1e-12), total_samples


def train_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: TrainingConfig,
    output_dir: Optional[Path] = None,
    sample_weights: Optional[np.ndarray] = None,
    soft_targets: Optional[np.ndarray] = None,
    seed: int = 42,
) -> TrainingResult:
    device = torch.device(cfg.device)
    model.to(device)

    dataset = ArrayDataset(X_train, y_train, sample_weights=sample_weights, soft_targets=soft_targets)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        generator=generator,
    )
    weights_tensor = dataset.sample_weights.to(device) if dataset.sample_weights is not None else None
    soft_targets_tensor = dataset.soft_targets.to(device) if dataset.soft_targets is not None else None
    optimizer = _build_optimizer(model, cfg)

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    last_model_path = str(output_path / "last_model.pt") if output_path is not None and cfg.save_last_model else None
    best_model_path = str(output_path / "best_model.pt") if output_path is not None and cfg.save_best_model else None

    history: List[Dict[str, float]] = []
    best_train_loss = float("inf")
    best_epoch: Optional[int] = None
    bad_epochs = 0
    train_time_sec = 0.0

    for epoch in range(1, int(cfg.max_epochs) + 1):
        epoch_start = time.perf_counter()
        train_loss, samples_seen = _train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            sample_weights=weights_tensor,
            soft_targets=soft_targets_tensor,
        )
        epoch_time = time.perf_counter() - epoch_start
        train_time_sec += epoch_time

        improved = train_loss < (best_train_loss - float(cfg.early_stop_min_delta))
        if improved:
            best_train_loss = float(train_loss)
            best_epoch = int(epoch)
            bad_epochs = 0
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_time_sec": float(epoch_time),
                "samples_seen": float(samples_seen),
                "best_train_loss_so_far": float(best_train_loss),
            }
        )

        if cfg.early_stop and bad_epochs >= int(cfg.early_stop_patience):
            break

    if last_model_path is not None:
        torch.save(model.state_dict(), last_model_path)

    return TrainingResult(
        history=history,
        epochs_ran=len(history),
        best_epoch=best_epoch,
        best_train_loss=None if best_epoch is None else best_train_loss,
        train_time_sec=float(train_time_sec),
        last_model_path=last_model_path,
        best_model_path=best_model_path,
    )


@torch.no_grad()
def predict_classifier(model: nn.Module, X: np.ndarray, batch_size: int = 4096, device: str = "cpu") -> np.ndarray:
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)
    preds: list[np.ndarray] = []
    for start in range(0, len(X), int(batch_size)):
        xb = torch.tensor(X[start : start + int(batch_size)], dtype=torch.float32, device=torch_device)
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64))
    return np.concatenate(preds, axis=0)

