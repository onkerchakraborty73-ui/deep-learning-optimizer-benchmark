
#!/usr/bin/env python3
"""
Rigorous optimizer benchmark:
    Architectures: ResNet-18, ViT-Tiny
    Datasets:      MNIST, CIFAR-10
    Optimizers:    AdamW, Lion, Shampoo, Muon
    Evaluation:    multi-seed, convergence, runtime, memory, update norms,
                   confidence intervals, bootstrap CI, plots, ablations.

Examples
--------
Quick smoke test:
    python optimizer_benchmark.py --datasets mnist --models resnet --optimizers adamw lion --seeds 0 --epochs 2

Recommended benchmark:
    python optimizer_benchmark.py \
        --datasets mnist cifar10 \
        --models resnet vit \
        --optimizers adamw lion shampoo muon \
        --seeds 0 1 2 3 4 \
        --epochs 50

Full 10-seed study:
    python optimizer_benchmark.py \
        --datasets mnist cifar10 \
        --models resnet vit \
        --optimizers adamw lion shampoo muon \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --epochs 100

LR ablation:
    python optimizer_benchmark.py --mode lr_ablation --dataset cifar10 --model resnet \
        --optimizer adamw --seeds 0 1 2 --epochs 50

Requirements:
    pip install torch torchvision pandas numpy matplotlib
Optional:
    CUDA-enabled PyTorch is strongly recommended for the full benchmark.

Notes
-----
1. Muon is available as torch.optim.Muon in sufficiently recent PyTorch releases.
   It is used for 2-D weight matrices and AdamW is used for remaining parameters,
   following the current PyTorch API guidance.
2. Lion is implemented directly from the published/reference PyTorch algorithm.
3. Shampoo below is a practical matrix-Shampoo implementation. It keeps left/right
   second-moment statistics for 2-D parameters, updates inverse-root preconditioners
   periodically, and falls back to AdamW-style updates for non-2-D parameters.
4. For a publication-quality study, do not tune on the test set. This script uses
   train/validation/test separation and reports the test set only after training.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic mode improves reproducibility but may reduce speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18CIFAR(nn.Module):
    """CIFAR/MNIST-friendly ResNet-18: 3x3 stem, no initial max-pool."""

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        image_size=32,
        patch_size=4,
        dim=192,
        depth=6,
        heads=3,
        mlp_dim=768,
        dropout=0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        n_patches = (image_size // patch_size) ** 2

        self.patch = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)                         # B,D,H',W'
        x = x.flatten(2).transpose(1, 2)         # B,N,D
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        return self.head(x)


def build_model(name: str, in_channels: int) -> nn.Module:
    if name == "resnet":
        return ResNet18CIFAR(in_channels=in_channels)
    if name == "vit":
        return ViTTiny(in_channels=in_channels)
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Lion optimizer
# ---------------------------------------------------------------------------

class Lion(Optimizer):
    """Reference-style PyTorch Lion optimizer."""

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    ):
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if wd != 0:
                    p.mul_(1 - lr * wd)

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


# ---------------------------------------------------------------------------
# Practical matrix Shampoo
# ---------------------------------------------------------------------------

def matrix_inverse_root(mat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric inverse square root using eigendecomposition.
    Intended for small/medium preconditioner matrices.
    """
    # Numerical symmetrization
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    vals, vecs = torch.linalg.eigh(mat)
    vals = vals.clamp_min(eps)
    inv_sqrt = vals.rsqrt()
    return (vecs * inv_sqrt.unsqueeze(0)) @ vecs.transpose(-1, -2)


class Shampoo(Optimizer):
    """
    Practical matrix Shampoo for 2-D tensors.

    For a matrix gradient G (m x n):
        L <- beta L + (1-beta) G G^T
        R <- beta R + (1-beta) G^T G
        G_pre <- L^{-1/4} G R^{-1/4}

    The inverse roots are refreshed every `precondition_frequency` steps.
    Non-2D tensors use an AdamW-style fallback state.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0.01,
        precondition_frequency=10,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            freq = group["precondition_frequency"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if wd != 0:
                    p.mul_(1 - lr * wd)

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0

                    if p.ndim == 2:
                        m, n = p.shape
                        state["left"] = torch.eye(
                            m, device=p.device, dtype=p.dtype
                        )
                        state["right"] = torch.eye(
                            n, device=p.device, dtype=p.dtype
                        )
                        state["left_root"] = torch.eye(
                            m, device=p.device, dtype=p.dtype
                        )
                        state["right_root"] = torch.eye(
                            n, device=p.device, dtype=p.dtype
                        )
                        state["momentum"] = torch.zeros_like(p)
                    else:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                if p.ndim == 2:
                    left = state["left"]
                    right = state["right"]

                    left.mul_(beta2).addmm_(g, g.t(), beta=1 - beta2)
                    right.mul_(beta2).addmm_(g.t(), g, beta=1 - beta2)

                    if state["step"] == 1 or state["step"] % freq == 0:
                        # Shampoo requires fourth-root preconditioners.
                        # We obtain them as inverse sqrt of the sqrt matrix:
                        # A^(-1/4) = (A^(1/2))^(-1/2).
                        l_sqrt = matrix_inverse_root(left, eps).inverse()
                        r_sqrt = matrix_inverse_root(right, eps).inverse()

                        state["left_root"] = matrix_inverse_root(
                            l_sqrt, eps
                        )
                        state["right_root"] = matrix_inverse_root(
                            r_sqrt, eps
                        )

                    pre_g = (
                        state["left_root"] @ g @ state["right_root"]
                    )

                    momentum = state["momentum"]
                    momentum.mul_(beta1).add_(pre_g, alpha=1 - beta1)
                    p.add_(momentum, alpha=-lr)

                else:
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        g, g, value=1 - beta2
                    )

                    denom = exp_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-lr)

        return loss


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def split_muon_params(model):
    """
    Muon for 2-D weights; AdamW for biases, norms, embeddings, conv kernels,
    and other non-2-D parameters.

    This follows the current torch.optim.Muon guidance that Muon is intended
    for 2-D hidden-layer weights.
    """
    muon_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Avoid treating biases / normalization parameters as Muon parameters.
        if p.ndim == 2 and not name.endswith(".bias"):
            muon_params.append(p)
        else:
            other_params.append(p)

    return muon_params, other_params


def make_optimizer(
    name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
    device: torch.device,
):
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

    if name == "lion":
        return Lion(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            weight_decay=weight_decay,
        )

    if name == "shampoo":
        return Shampoo(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=weight_decay,
            precondition_frequency=10,
        )

    if name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "Your PyTorch version does not expose torch.optim.Muon. "
                "Upgrade PyTorch to a version that includes it."
            )

        muon_params, other_params = split_muon_params(model)

        # PyTorch's native Muon API is designed for 2-D weights; use AdamW
        # for remaining parameters.
        muon = torch.optim.Muon(
            muon_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.95,
        )
        adam = torch.optim.AdamW(
            other_params,
            lr=lr * 0.3,
            weight_decay=weight_decay,
        )
        return [muon, adam]

    raise ValueError(f"Unknown optimizer: {name}")


def optimizer_zero_grad(optimizer):
    if isinstance(optimizer, list):
        for opt in optimizer:
            opt.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad(set_to_none=True)


def optimizer_step(optimizer):
    if isinstance(optimizer, list):
        for opt in optimizer:
            opt.step()
    else:
        optimizer.step()


def optimizer_state_dict(optimizer):
    if isinstance(optimizer, list):
        return [opt.state_dict() for opt in optimizer]
    return optimizer.state_dict()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_datasets(dataset_name: str, data_dir: str, seed: int):
    if dataset_name == "mnist":
        mean, std = (0.1307,), (0.3081,)
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        full = datasets.MNIST(
            data_dir, train=True, download=True, transform=train_transform
        )
        test = datasets.MNIST(
            data_dir, train=False, download=True, transform=eval_transform
        )
        in_channels = 1

    elif dataset_name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        full = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=train_transform
        )
        test = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=eval_transform
        )
        in_channels = 3

    else:
        raise ValueError(dataset_name)

    # Fixed validation split per seed. Test set remains untouched.
    generator = torch.Generator().manual_seed(seed)
    n_val = int(0.1 * len(full))
    n_train = len(full) - n_val
    train, val = random_split(
        full, [n_train, n_val], generator=generator
    )

    # Validation must use eval transforms, not training augmentation.
    val_base = copy.deepcopy(full)
    val_base.transform = eval_transform
    val = torch.utils.data.Subset(val_base, val.indices)

    return train, val, test, in_channels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy_from_logits(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            bs = y.size(0)
            loss_sum += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()
            total += bs

    return loss_sum / total, correct / total


def parameter_norm(model):
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        total += p.detach().float().pow(2).sum()
    return total.sqrt().item()


def gradient_norm(model):
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum()
    return total.sqrt().item()


def update_norm(model, before):
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        old = before[id(p)]
        total += (p.detach().float() - old.float()).pow(2).sum()
    return total.sqrt().item()


def snapshot_parameters(model):
    return {
        id(p): p.detach().clone()
        for p in model.parameters()
        if p.requires_grad
    }


def get_peak_memory(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return float("nan")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class Config:
    dataset: str
    model: str
    optimizer: str
    seed: int
    epochs: int = 50
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    num_workers: int = 2
    data_dir: str = "./data"
    output_dir: str = "./results"
    grad_clip: float = 1.0
    warmup_epochs: int = 5


def lr_scale_for_optimizer(optimizer: str, base_lr: float) -> float:
    # Lion commonly uses a smaller LR than AdamW.
    # Shampoo and Muon are intentionally left explicit rather than silently
    # hiding optimizer-specific tuning.
    if optimizer == "lion":
        return base_lr * 0.3
    return base_lr


def train_one(config: Config, device: torch.device):
    seed_everything(config.seed)

    out_root = Path(config.output_dir)
    run_dir = (
        out_root
        / config.dataset
        / config.model
        / config.optimizer
        / f"seed_{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, in_channels = make_datasets(
        config.dataset, config.data_dir, config.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
    )

    model = build_model(config.model, in_channels).to(device)
    criterion = nn.CrossEntropyLoss()

    actual_lr = lr_scale_for_optimizer(config.optimizer, config.lr)

    optimizer = make_optimizer(
        config.optimizer,
        model,
        actual_lr,
        config.weight_decay,
        device,
    )

    # Warmup + cosine schedule.
    if isinstance(optimizer, list):
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, config.epochs - config.warmup_epochs)
            )
            for opt in optimizer
        ]
    else:
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, config.epochs - config.warmup_epochs)
            )
        ]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows = []
    best_val = -float("inf")
    best_epoch = -1

    metadata = {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "actual_lr": actual_lr,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()

        # Warmup implemented by scaling optimizer learning rates.
        if config.warmup_epochs > 0 and epoch <= config.warmup_epochs:
            warmup_factor = epoch / config.warmup_epochs
            target_lrs = []
            if isinstance(optimizer, list):
                for opt in optimizer:
                    for group in opt.param_groups:
                        target_lrs.append(group["lr"])
                # Store original LR once.
                if epoch == 1:
                    for opt in optimizer:
                        for group in opt.param_groups:
                            group.setdefault("_target_lr", group["lr"])
                for opt in optimizer:
                    for group in opt.param_groups:
                        group["lr"] = group["_target_lr"] * warmup_factor
            else:
                for group in optimizer.param_groups:
                    if epoch == 1:
                        group.setdefault("_target_lr", group["lr"])
                    group["lr"] = group["_target_lr"] * warmup_factor

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        last_grad_norm = float("nan")
        last_update_norm = float("nan")

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer_zero_grad(optimizer)

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            last_grad_norm = gradient_norm(model)

            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )

            before = snapshot_parameters(model)
            optimizer_step(optimizer)
            last_update_norm = update_norm(model, before)

            bs = y.size(0)
            train_loss_sum += loss.item() * bs
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += bs

        if epoch > config.warmup_epochs:
            if isinstance(optimizer, list):
                for sch in schedulers:
                    sch.step()
            else:
                schedulers[0].step()

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_time = time.perf_counter() - epoch_start
        pnorm = parameter_norm(model)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "config": asdict(config),
                },
                run_dir / "best.pt",
            )

        current_lrs = []
        if isinstance(optimizer, list):
            for opt in optimizer:
                current_lrs.extend(
                    [group["lr"] for group in opt.param_groups]
                )
        else:
            current_lrs = [group["lr"] for group in optimizer.param_groups]

        rows.append({
            "dataset": config.dataset,
            "model": config.model,
            "optimizer": config.optimizer,
            "seed": config.seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "epoch_time_sec": epoch_time,
            "cumulative_time_sec": sum(r["epoch_time_sec"] for r in rows) + epoch_time,
            "learning_rate": float(np.mean(current_lrs)),
            "gradient_norm": last_grad_norm,
            "update_norm": last_update_norm,
            "parameter_norm": pnorm,
            "update_parameter_ratio": (
                last_update_norm / max(pnorm, 1e-12)
            ),
            "peak_memory_mb": get_peak_memory(device),
        })

        print(
            f"[{config.dataset:7s} | {config.model:7s} | "
            f"{config.optimizer:7s} | seed={config.seed:2d}] "
            f"epoch {epoch:3d}/{config.epochs} "
            f"train={train_acc*100:6.2f}% "
            f"val={val_acc*100:6.2f}% "
            f"time={epoch_time:6.1f}s"
        )

    # Load best validation checkpoint before final test evaluation.
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    final = {
        "dataset": config.dataset,
        "model": config.model,
        "optimizer": config.optimizer,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "total_training_time_sec": sum(r["epoch_time_sec"] for r in rows),
        "peak_memory_mb": max(
            [r["peak_memory_mb"] for r in rows if not math.isnan(r["peak_memory_mb"])]
            or [float("nan")]
        ),
        "actual_lr": actual_lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
    }

    pd.DataFrame(rows).to_csv(run_dir / "history.csv", index=False)
    (run_dir / "final.json").write_text(json.dumps(final, indent=2))

    return rows, final


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n_boot=5000, seed=1234, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()

    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) +
         (len(b) - 1) * np.var(b, ddof=1))
        / max(len(a) + len(b) - 2, 1)
    )
    return float((np.mean(a) - np.mean(b)) / max(pooled, 1e-12))


def paired_permutation_pvalue(a, b, n_perm=10000, seed=1234):
    """Two-sided paired permutation test."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return float("nan")

    diff = a - b
    observed = abs(diff.mean())

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diff))
        perm = np.mean(diff * signs)
        count += abs(perm) >= observed

    return float((count + 1) / (n_perm + 1))


def aggregate_results(output_dir):
    output_dir = Path(output_dir)
    files = list(output_dir.glob("*/*/*/seed_*/final.json"))

    finals = []
    histories = []

    for f in files:
        finals.append(json.loads(f.read_text()))
        hist = f.parent / "history.csv"
        if hist.exists():
            histories.append(pd.read_csv(hist))

    if not finals:
        print("No results found.")
        return

    final_df = pd.DataFrame(finals)
    final_df.to_csv(output_dir / "all_final_results.csv", index=False)

    grouped_rows = []
    for keys, group in final_df.groupby(
        ["dataset", "model", "optimizer"]
    ):
        vals = group["test_accuracy"].to_numpy()
        lo, hi = bootstrap_ci(vals)

        grouped_rows.append({
            "dataset": keys[0],
            "model": keys[1],
            "optimizer": keys[2],
            "n_seeds": len(vals),
            "mean_test_accuracy": vals.mean(),
            "std_test_accuracy": vals.std(ddof=1) if len(vals) > 1 else 0.0,
            "min_test_accuracy": vals.min(),
            "max_test_accuracy": vals.max(),
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
            "mean_training_time_sec": group["total_training_time_sec"].mean(),
            "std_training_time_sec": group["total_training_time_sec"].std(ddof=1)
                if len(group) > 1 else 0.0,
            "mean_peak_memory_mb": group["peak_memory_mb"].mean(),
        })

    summary = pd.DataFrame(grouped_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)

    # Pairwise optimizer statistics within each dataset/model.
    stats_rows = []
    optimizers = sorted(final_df["optimizer"].unique())

    for (dataset, model), subset in final_df.groupby(["dataset", "model"]):
        pivot = subset.pivot(
            index="seed", columns="optimizer", values="test_accuracy"
        )

        for i, opt_a in enumerate(optimizers):
            for opt_b in optimizers[i + 1:]:
                if opt_a not in pivot.columns or opt_b not in pivot.columns:
                    continue

                paired = pivot[[opt_a, opt_b]].dropna()
                a = paired[opt_a].to_numpy()
                b = paired[opt_b].to_numpy()

                stats_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "optimizer_a": opt_a,
                    "optimizer_b": opt_b,
                    "n_paired_seeds": len(a),
                    "mean_difference_a_minus_b": float(a.mean() - b.mean()),
                    "cohens_d": cohens_d(a, b) if len(a) > 1 else float("nan"),
                    "paired_permutation_p": paired_permutation_pvalue(a, b),
                })

    pd.DataFrame(stats_rows).to_csv(
        output_dir / "pairwise_statistics.csv", index=False
    )

    return final_df, summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(output_dir):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    history_files = list(output_dir.glob("*/*/*/seed_*/history.csv"))
    if not history_files:
        print("No histories available for plots.")
        return

    df = pd.concat([pd.read_csv(f) for f in history_files], ignore_index=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)

    # Mean +/- SD learning curves.
    for metric, ylabel, filename in [
        ("val_accuracy", "Validation accuracy", "validation_accuracy.png"),
        ("train_loss", "Training loss", "training_loss.png"),
    ]:
        for (dataset, model), subset in df.groupby(["dataset", "model"]):
            plt.figure(figsize=(9, 6))

            for optimizer, opt_df in subset.groupby("optimizer"):
                stats = (
                    opt_df.groupby("epoch")[metric]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                x = stats["epoch"].to_numpy()
                y = stats["mean"].to_numpy()
                s = stats["std"].fillna(0).to_numpy()

                plt.plot(x, y, label=optimizer)
                plt.fill_between(x, y - s, y + s, alpha=0.15)

            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"{dataset.upper()} — {model}")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(
                figures / f"{dataset}_{model}_{filename}",
                dpi=200
            )
            plt.close()

    # Accuracy vs cumulative time.
    for (dataset, model), subset in df.groupby(["dataset", "model"]):
        plt.figure(figsize=(9, 6))

        for optimizer, opt_df in subset.groupby("optimizer"):
            stats = (
                opt_df.groupby("epoch")
                .agg(
                    mean_time=("cumulative_time_sec", "mean"),
                    mean_acc=("val_accuracy", "mean"),
                )
                .reset_index()
            )
            plt.plot(
                stats["mean_time"] / 60,
                stats["mean_acc"],
                label=optimizer
            )

        plt.xlabel("Cumulative training time (minutes)")
        plt.ylabel("Mean validation accuracy")
        plt.title(f"Accuracy vs compute time — {dataset.upper()} — {model}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            figures / f"{dataset}_{model}_accuracy_vs_time.png",
            dpi=200
        )
        plt.close()

    # Final test accuracy distribution.
    final_path = output_dir / "all_final_results.csv"
    if final_path.exists():
        final_df = pd.read_csv(final_path)
        for (dataset, model), subset in final_df.groupby(
            ["dataset", "model"]
        ):
            plt.figure(figsize=(9, 6))
            data = [
                subset.loc[
                    subset.optimizer == opt, "test_accuracy"
                ].to_numpy()
                for opt in sorted(subset.optimizer.unique())
            ]
            labels = sorted(subset.optimizer.unique())
            plt.boxplot(data, labels=labels)
            plt.ylabel("Test accuracy")
            plt.title(f"Final test accuracy — {dataset.upper()} — {model}")
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(
                figures / f"{dataset}_{model}_final_accuracy_boxplot.png",
                dpi=200
            )
            plt.close()

    # Update/parameter ratio.
    for (dataset, model), subset in df.groupby(["dataset", "model"]):
        plt.figure(figsize=(9, 6))
        for optimizer, opt_df in subset.groupby("optimizer"):
            stats = (
                opt_df.groupby("epoch")["update_parameter_ratio"]
                .mean()
                .reset_index()
            )
            plt.plot(
                stats["epoch"],
                stats["update_parameter_ratio"],
                label=optimizer
            )
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Update / parameter norm")
        plt.title(f"Update scale — {dataset.upper()} — {model}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            figures / f"{dataset}_{model}_update_ratio.png",
            dpi=200
        )
        plt.close()

    print(f"Plots saved to: {figures}")


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------

def get_ablation_values(kind: str):
    if kind == "lr":
        return [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    if kind == "wd":
        return [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    if kind == "batch":
        return [32, 64, 128, 256, 512]
    raise ValueError(kind)


def run_ablation(args, device):
    if not args.dataset or not args.model or not args.optimizer:
        raise ValueError(
            "Ablation mode requires --dataset, --model and --optimizer."
        )

    values = get_ablation_values(args.ablation)

    for value in values:
        for seed in args.seeds:
            lr = args.lr
            wd = args.weight_decay
            batch = args.batch_size

            if args.ablation == "lr":
                lr = value
            elif args.ablation == "wd":
                wd = value
            elif args.ablation == "batch":
                batch = value

            cfg = Config(
                dataset=args.dataset,
                model=args.model,
                optimizer=args.optimizer,
                seed=seed,
                epochs=args.epochs,
                batch_size=batch,
                lr=lr,
                weight_decay=wd,
                num_workers=args.num_workers,
                data_dir=args.data_dir,
                output_dir=args.output_dir / f"ablation_{args.ablation}_{value}",
                grad_clip=args.grad_clip,
                warmup_epochs=args.warmup_epochs,
            )
            train_one(cfg, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Rigorous ResNet/ViT optimizer benchmark"
    )

    p.add_argument(
        "--mode",
        choices=["benchmark", "lr_ablation", "wd_ablation", "batch_ablation"],
        default="benchmark",
    )

    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["mnist", "cifar10"],
        default=["cifar10"],
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=["resnet", "vit"],
        default=["resnet", "vit"],
    )
    p.add_argument(
        "--optimizers",
        nargs="+",
        choices=["adamw", "lion", "shampoo", "muon"],
        default=["adamw", "lion", "shampoo", "muon"],
    )

    # Singular forms are used by ablation mode.
    p.add_argument("--dataset", choices=["mnist", "cifar10"])
    p.add_argument("--model", choices=["resnet", "vit"])
    p.add_argument("--optimizer", choices=["adamw", "lion", "shampoo", "muon"])

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--data-dir", type=Path, default=Path("./data"))
    p.add_argument("--output-dir", type=Path, default=Path("./results"))
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate plots after training.",
    )

    p.add_argument(
        "--ablation",
        choices=["lr", "wd", "batch"],
        help="Ablation variable when using *_ablation mode.",
    )

    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print("=" * 80)
    print("RIGOROUS OPTIMIZER BENCHMARK")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {device}")
    if device.type == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(device)}")
    print("=" * 80)

    if args.mode == "benchmark":
        all_finals = []

        for dataset in args.datasets:
            for model in args.models:
                for optimizer in args.optimizers:
                    for seed in args.seeds:
                        cfg = Config(
                            dataset=dataset,
                            model=model,
                            optimizer=optimizer,
                            seed=seed,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            num_workers=args.num_workers,
                            data_dir=str(args.data_dir),
                            output_dir=str(args.output_dir),
                            grad_clip=args.grad_clip,
                            warmup_epochs=args.warmup_epochs,
                        )

                        try:
                            _, final = train_one(cfg, device)
                            all_finals.append(final)
                        except Exception as exc:
                            # Do not silently hide failed experimental cells.
                            print(
                                f"\nFAILED: {dataset}/{model}/{optimizer}/seed={seed}"
                            )
                            print(f"Reason: {repr(exc)}")
                            print(
                                "The run is skipped so the rest of the "
                                "matrix can continue.\n"
                            )

        if all_finals:
            pd.DataFrame(all_finals).to_csv(
                args.output_dir / "all_final_results.csv",
                index=False,
            )

        aggregate_results(args.output_dir)

        if args.make_plots:
            make_plots(args.output_dir)

    else:
        ablation = args.mode.replace("_ablation", "")
        args.ablation = args.ablation or ablation
        run_ablation(args, device)
        aggregate_results(args.output_dir)

        if args.make_plots:
            make_plots(args.output_dir)


if __name__ == "__main__":
    main()
