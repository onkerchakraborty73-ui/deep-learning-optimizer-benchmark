
"""
Rigorous CUDA vision optimizer benchmark.

Architectures:
  - ResNet-18
  - ViT-Tiny

Datasets:
  - MNIST
  - CIFAR-10

Optimizers:
  - AdamW (reported as "adam")
  - Lion
  - Shampoo-Lite (diagonal approximation; NOT full matrix Shampoo)
  - Muon (torch.optim.Muon)

Features:
  - CUDA validation
  - FP16 AMP
  - deterministic seeds
  - synchronized GPU timing
  - peak VRAM
  - per-epoch train/validation metrics
  - per-run CSV/JSON/checkpoint
  - aggregate CSV/JSON
  - summary statistics and plots when pandas/matplotlib are installed

IMPORTANT:
This is intended as a robust experimental starting point. The Shampoo implementation
is explicitly a diagonal approximation. For publication-grade Shampoo claims, replace
it with a validated full Shampoo implementation and document its exact variant.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibility takes priority over maximum throughput.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# Models
# ============================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.in_planes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [
            BasicBlock(self.in_planes, planes, stride)
        ]

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

        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)

        return self.fc(x)


class ViTTiny(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        image_size=32,
        patch_size=4,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_dim=384,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        cls = self.cls_token.expand(
            x.shape[0], -1, -1
        )

        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x[:, 0])

        return self.head(x)


# ============================================================
# Optimizers
# ============================================================

class Lion(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.99),
        weight_decay=1e-2,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            ),
        )

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

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                update = beta1 * exp_avg + (1.0 - beta1) * p.grad

                p.mul_(1.0 - lr * wd)
                p.add_(torch.sign(update), alpha=-lr)

                exp_avg.mul_(beta2)
                exp_avg.add_(p.grad, alpha=1.0 - beta2)

        return loss


class ShampooLite(torch.optim.Optimizer):
    """
    Diagonal Shampoo-style approximation.

    This is intentionally called ShampooLite. It is NOT the full
    matrix-preconditioned Shampoo algorithm.
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        beta=0.999,
        eps=1e-8,
        weight_decay=1e-2,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                beta=beta,
                eps=eps,
                weight_decay=weight_decay,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["v"] = torch.zeros_like(p)

                v = state["v"]
                v.mul_(beta)
                v.addcmul_(p.grad, p.grad, value=1.0 - beta)

                p.mul_(1.0 - lr * wd)
                p.addcdiv_(
                    p.grad,
                    v.sqrt().add(eps),
                    value=-lr,
                )

        return loss


class MuonHybrid:
    """
    Official-style hybrid Muon setup:

      * 2-D parameters -> Muon
      * all non-2-D parameters -> AdamW

    PyTorch documents Muon as an optimizer for 2-D hidden-layer weights
    and recommends a standard optimizer such as AdamW for other parameters.

    This wrapper makes BOTH optimizers part of one training optimizer,
    so no trainable parameter is accidentally left out.
    """

    def __init__(
        self,
        model,
        muon_lr=0.02,
        adam_lr=3e-4,
        weight_decay=1e-2,
    ):
        muon_params = []
        other_params = []

        for p in model.parameters():
            if not p.requires_grad:
                continue

            if p.ndim == 2:
                muon_params.append(p)
            else:
                other_params.append(p)

        if not muon_params:
            raise RuntimeError(
                "MuonHybrid found no 2-D parameters for Muon."
            )

        # PyTorch's Muon requires every parameter passed to it to be 2-D.
        self.muon = torch.optim.Muon(
            muon_params,
            lr=muon_lr,
            weight_decay=weight_decay,
            momentum=0.95,
            nesterov=True,
            adjust_lr_fn="original",
        )

        # Conv kernels, normalization parameters, biases, embeddings, etc.
        self.adamw = torch.optim.AdamW(
            other_params,
            lr=adam_lr,
            weight_decay=weight_decay,
        ) if other_params else None

        self.muon_param_count = sum(
            p.numel() for p in muon_params
        )
        self.fallback_param_count = sum(
            p.numel() for p in other_params
        )

    @property
    def param_groups(self):
        groups = list(self.muon.param_groups)
        if self.adamw is not None:
            groups += list(self.adamw.param_groups)
        return groups

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": (
                self.adamw.state_dict()
                if self.adamw is not None
                else None
            ),
        }

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        if self.adamw is not None and state["adamw"] is not None:
            self.adamw.load_state_dict(state["adamw"])


def make_optimizer(
    name,
    model,
    lr,
    weight_decay,
    muon_lr=0.02,
):
    name = name.lower()

    if name == "adam":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if name == "lion":
        return Lion(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if name == "shampoo":
        return ShampooLite(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "This PyTorch build does not expose torch.optim.Muon."
            )

        return MuonHybrid(
            model,
            muon_lr=muon_lr,
            adam_lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unknown optimizer: {name}"
    )


# ============================================================
# Data
# ============================================================

def build_datasets(dataset_name, data_dir, seed):
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        mean = (0.1307,)
        std = (0.3081,)

        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_aug = datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        train_eval = datasets.MNIST(
            data_dir,
            train=True,
            download=False,
            transform=eval_transform,
        )

        test = datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=eval_transform,
        )

        channels = 1
        classes = 10

    elif dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

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

        train_aug = datasets.CIFAR10(
            data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        train_eval = datasets.CIFAR10(
            data_dir,
            train=True,
            download=False,
            transform=eval_transform,
        )

        test = datasets.CIFAR10(
            data_dir,
            train=False,
            download=True,
            transform=eval_transform,
        )

        channels = 3
        classes = 10

    else:
        raise ValueError(
            "dataset must be mnist or cifar10"
        )

    n = len(train_aug)
    val_size = int(0.10 * n)

    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(
        n,
        generator=generator,
    ).tolist()

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(
        train_aug,
        train_indices,
    )

    val_subset = Subset(
        train_eval,
        val_indices,
    )

    return (
        train_subset,
        val_subset,
        test,
        channels,
        classes,
    )


def make_loaders(
    dataset_name,
    data_dir,
    seed,
    batch_size,
):
    (
        train_ds,
        val_ds,
        test_ds,
        channels,
        classes,
    ) = build_datasets(
        dataset_name,
        data_dir,
        seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        channels,
        classes,
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, amp):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp,
        ):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        batch_n = y.size(0)

        total_loss += loss.item() * batch_n
        total_correct += (
            logits.argmax(dim=1) == y
        ).sum().item()

        total_examples += batch_n

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


def model_parameter_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def gradient_norm(model):
    sq_sum = 0.0

    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            sq_sum += torch.sum(g * g).item()

    return math.sqrt(sq_sum)


# ============================================================
# One experiment
# ============================================================

def run_experiment(
    args,
    dataset_name,
    model_name,
    optimizer_name,
    seed,
):
    seed_everything(seed)

    batch_size = (
        args.vit_batch
        if model_name == "vit"
        else args.resnet_batch
    )

    (
        train_loader,
        val_loader,
        test_loader,
        channels,
        classes,
    ) = make_loaders(
        dataset_name,
        args.data,
        seed,
        batch_size,
    )

    if model_name == "resnet":
        model = ResNet18(
            channels,
            classes,
        )
    elif model_name == "vit":
        model = ViTTiny(
            channels,
            classes,
        )
    else:
        raise ValueError(model_name)

    model = model.cuda()

    optimizer = make_optimizer(
        optimizer_name,
        model,
        args.lr,
        args.weight_decay,
        muon_lr=args.muon_lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    amp_enabled = not args.no_amp

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    run_dir = (
        Path(args.output)
        / dataset_name
        / model_name
        / optimizer_name
        / f"seed_{seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_val_acc = -1.0
    best_epoch = 0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_start = time.perf_counter()

        train_loss_sum = 0.0
        train_correct = 0
        train_examples = 0

        grad_norms = []

        for x, y in train_loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(x)
                loss = F.cross_entropy(
                    logits,
                    y,
                )

            if amp_enabled:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                grad_norms.append(
                    gradient_norm(model)
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()

                grad_norms.append(
                    gradient_norm(model)
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

                optimizer.step()

            batch_n = y.size(0)

            train_loss_sum += (
                loss.item() * batch_n
            )

            train_correct += (
                logits.argmax(dim=1) == y
            ).sum().item()

            train_examples += batch_n

        scheduler.step()

        # Critical: synchronize before timing/VRAM measurements.
        torch.cuda.synchronize()

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            amp_enabled,
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        peak_vram = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        train_loss = (
            train_loss_sum
            / train_examples
        )

        train_acc = (
            train_correct
            / train_examples
        )

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
            "mean_grad_norm": (
                float(np.mean(grad_norms))
                if grad_norms
                else 0.0
            ),
            "peak_vram_gb": peak_vram,
        }

        history.append(row)

        print(
            f"[{dataset_name}|{model_name}|{optimizer_name}"
            f"|seed={seed}] "
            f"epoch {epoch}/{args.epochs} "
            f"train={train_acc:.4f} "
            f"val={val_acc:.4f} "
            f"time={epoch_time:.1f}s "
            f"VRAM={peak_vram:.2f}GB"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                run_dir / "best.pt",
            )

    torch.cuda.synchronize()

    total_time = (
        time.perf_counter()
        - total_start
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        amp_enabled,
    )

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    samples_per_second = (
        len(train_loader.dataset)
        * args.epochs
        / total_time
    )

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "optimizer": optimizer_name,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "amp": amp_enabled,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "parameters": model_parameter_count(model),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_time_sec": total_time,
        "samples_per_sec": samples_per_second,
        "peak_vram_gb": peak_vram,
        "status": "OK",
    }

    if isinstance(optimizer, MuonHybrid):
        result["muon_parameters"] = optimizer.muon_param_count
        result["fallback_adamw_parameters"] = optimizer.fallback_param_count
        result["muon_lr"] = args.muon_lr

    with open(
        run_dir / "history.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)

    with open(
        run_dir / "final.json",
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    return result


# ============================================================
# Aggregation
# ============================================================

def save_summary(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "summary.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    if not results:
        return

    fieldnames = sorted(
        {
            key
            for result in results
            for key in result.keys()
        }
    )

    with open(
        output_dir / "summary.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(results)

        ok = df[df["status"] == "OK"].copy()

        if ok.empty:
            return

        stats = (
            ok.groupby(
                ["dataset", "model", "optimizer"]
            )["test_acc"]
            .agg(
                mean="mean",
                std="std",
                median="median",
                min="min",
                max="max",
                n="count",
            )
            .reset_index()
        )

        stats.to_csv(
            output_dir / "group_statistics.csv",
            index=False,
        )

        plots = output_dir / "plots"
        plots.mkdir(
            parents=True,
            exist_ok=True,
        )

        for (dataset, model), group in ok.groupby(
            ["dataset", "model"]
        ):
            means = (
                group.groupby("optimizer")["test_acc"]
                .mean()
                .sort_values(ascending=False)
            )

            ax = means.plot(
                kind="bar",
                title=(
                    f"{dataset.upper()} — "
                    f"{model} mean test accuracy"
                ),
            )

            ax.set_xlabel("Optimizer")
            ax.set_ylabel("Test accuracy")

            fig = ax.get_figure()
            fig.tight_layout()

            fig.savefig(
                plots
                / f"{dataset}_{model}_accuracy.png",
                dpi=180,
            )

            plt.close(fig)

            ax = group.boxplot(
                column="test_acc",
                by="optimizer",
            )

            ax.set_title(
                f"{dataset.upper()} — "
                f"{model} test accuracy across seeds"
            )

            ax.set_xlabel("Optimizer")
            ax.set_ylabel("Test accuracy")

            plt.suptitle("")

            fig = ax.get_figure()
            fig.tight_layout()

            fig.savefig(
                plots
                / f"{dataset}_{model}_seed_distribution.png",
                dpi=180,
            )

            plt.close(fig)

    except Exception as exc:
        print(
            "Optional pandas/matplotlib summary failed:",
            repr(exc),
        )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CUDA vision optimizer benchmark"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["mnist", "cifar10"],
        default=["mnist", "cifar10"],
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet", "vit"],
        default=["resnet", "vit"],
    )

    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=[
            "adam",
            "lion",
            "shampoo",
            "muon",
        ],
        default=[
            "adam",
            "lion",
            "shampoo",
            "muon",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--muon-lr",
        type=float,
        default=2e-2,
        help="Learning rate for the Muon 2-D parameter group.",
    )

    parser.add_argument(
        "--resnet-batch",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--vit-batch",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--data",
        default="./data",
    )

    parser.add_argument(
        "--output",
        default="./results",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run this inside your CUDA PyTorch environment."
        )

    # Smoke test intentionally uses only one short experiment.
    if args.smoke_test:
        args.datasets = ["mnist"]
        args.models = ["resnet"]
        args.optimizers = ["adam"]
        args.seeds = [0]
        args.epochs = 1

    print("=" * 80)
    print("RIGOROUS CUDA OPTIMIZER BENCHMARK")
    print("=" * 80)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2,
        ),
        "GB",
    )
    print(
        "Muon available:",
        hasattr(torch.optim, "Muon"),
    )
    print(
        "AMP:",
        not args.no_amp,
    )
    print(
        "Muon LR:",
        args.muon_lr,
    )
    print("=" * 80)

    results = []

    total_runs = (
        len(args.datasets)
        * len(args.models)
        * len(args.optimizers)
        * len(args.seeds)
    )

    run_number = 0

    for dataset_name in args.datasets:
        for model_name in args.models:
            for optimizer_name in args.optimizers:
                for seed in args.seeds:
                    run_number += 1

                    print(
                        f"\nRUN {run_number}/{total_runs}"
                    )

                    try:
                        result = run_experiment(
                            args,
                            dataset_name,
                            model_name,
                            optimizer_name,
                            seed,
                        )

                    except Exception as exc:
                        print(
                            "FAILED:",
                            dataset_name,
                            model_name,
                            optimizer_name,
                            f"seed={seed}",
                            repr(exc),
                        )

                        result = {
                            "dataset": dataset_name,
                            "model": model_name,
                            "optimizer": optimizer_name,
                            "seed": seed,
                            "status": "FAILED",
                            "error": repr(exc),
                        }

                    results.append(result)

                    save_summary(
                        results,
                        args.output,
                    )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(
        "Results:",
        Path(args.output).resolve(),
    )


if __name__ == "__main__":
    main()
