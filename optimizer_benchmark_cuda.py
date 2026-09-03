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
  - Shampoo-Lite (diagonal approximation)
  - Muon (Custom Newton-Schulz pure PyTorch implementation)

Features:
  - CUDA validation & safe cuDNN execution
  - Safe FP16 AMP integration
  - Windows-safe multi-processing
  - Deterministic seeds
  - Synchronized GPU timing
  - Peak VRAM tracking
  - Per-epoch train/validation metrics logging
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

# Direct cuDNN environment stabilization to prevent kernel execution failure
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

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

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x[:, 0])

        return self.head(x)


# ============================================================
# Custom Optimizers
# ============================================================

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration in float32 precision to avoid Illegal Memory Access errors.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Force float32 cast to keep memory accesses bound safely on CUDA
    X = G.to(dtype=torch.float32)
    X /= (X.norm() + eps)
    
    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
        
    return X.to(dtype=G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.01, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buf" not in state:
                    state["momentum_buf"] = torch.zeros_like(g)

                buf = state["momentum_buf"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g + momentum * buf
                else:
                    update = buf

                update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)

        return loss


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.99), weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

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
    def __init__(self, params, lr=3e-4, beta=0.999, eps=1e-8, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, beta, eps, wd = group["lr"], group["beta"], group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["v"] = torch.zeros_like(p)

                v = state["v"]
                v.mul_(beta).addcmul_(p.grad, p.grad, value=1.0 - beta)

                p.mul_(1.0 - lr * wd)
                p.addcdiv_(p.grad, v.sqrt().add(eps), value=-lr)

        return loss


class MuonHybrid:
    def __init__(self, model, muon_lr=0.02, adam_lr=3e-4, weight_decay=1e-2):
        muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
        other_params = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]

        if not muon_params:
            raise RuntimeError("MuonHybrid found no 2-D parameters for Muon.")

        if hasattr(torch.optim, "Muon"):
            self.muon = torch.optim.Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)
        else:
            self.muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)

        self.adamw = torch.optim.AdamW(other_params, lr=adam_lr, weight_decay=weight_decay) if other_params else None

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        if self.adamw is not None:
            self.adamw.step()


def make_optimizer(name, model, lr, weight_decay, muon_lr=0.02):
    name = name.lower()
    if name == "adam":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "lion":
        return Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "shampoo":
        return ShampooLite(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "muon":
        return MuonHybrid(model, muon_lr=muon_lr, adam_lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


# ============================================================
# Data & Pipeline
# ============================================================

def build_datasets(dataset_name, data_dir, seed):
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        mean, std = (0.1307,), (0.3081,)
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_aug = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        train_eval = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
        test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        channels, classes = 1, 10

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
        train_aug = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        train_eval = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_transform)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_transform)
        channels, classes = 3, 10
    else:
        raise ValueError("dataset must be mnist or cifar10")

    n = len(train_aug)
    val_size = int(0.10 * n)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    val_indices, train_indices = indices[:val_size], indices[val_size:]
    train_subset = Subset(train_aug, train_indices)
    val_subset = Subset(train_eval, val_indices)

    return train_subset, val_subset, test, channels, classes


def make_loaders(dataset_name, data_dir, seed, batch_size):
    train_ds, val_ds, test_ds, channels, classes = build_datasets(dataset_name, data_dir, seed)

    # num_workers set to 0 for Windows IPC stability
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, channels, classes


@torch.no_grad()
def evaluate(model, loader, amp):
    model.eval()
    total_loss, total_correct, total_examples = 0.0, 0, 0

    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        batch_n = y.size(0)
        total_loss += loss.item() * batch_n
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_examples += batch_n

    return total_loss / total_examples, total_correct / total_examples


def gradient_norm(model):
    sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            sq_sum += torch.sum(g * g).item()
    return math.sqrt(sq_sum)


def run_experiment(args, dataset_name, model_name, optimizer_name, seed):
    seed_everything(seed)
    batch_size = args.vit_batch if model_name == "vit" else args.resnet_batch

    train_loader, val_loader, test_loader, channels, classes = make_loaders(
        dataset_name, args.data, seed, batch_size
    )

    model = ResNet18(channels, classes) if model_name == "resnet" else ViTTiny(channels, classes)
    model = model.cuda()

    optimizer = make_optimizer(optimizer_name, model, args.lr, args.weight_decay, muon_lr=args.muon_lr)

    if isinstance(optimizer, MuonHybrid):
        scheduler_muon = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.muon, T_max=args.epochs)
        scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.adamw, T_max=args.epochs) if optimizer.adamw else None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    amp_enabled = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_dir = Path(args.output) / dataset_name / model_name / optimizer_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    history, best_val_acc = [], -1.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        train_loss_sum, train_correct, train_examples, grad_norms = 0.0, 0, 0, []

        for x, y in train_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            if amp_enabled:
                scaler.scale(loss).backward()
                
                # Unscale explicitly prior to computing gradient norms
                if isinstance(optimizer, MuonHybrid):
                    scaler.unscale_(optimizer.muon)
                    if optimizer.adamw:
                        scaler.unscale_(optimizer.adamw)
                else:
                    scaler.unscale_(optimizer)

                grad_norms.append(gradient_norm(model))
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                if isinstance(optimizer, MuonHybrid):
                    scaler.step(optimizer.muon)
                    if optimizer.adamw:
                        scaler.step(optimizer.adamw)
                else:
                    scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norms.append(gradient_norm(model))
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                if isinstance(optimizer, MuonHybrid):
                    optimizer.step()
                else:
                    optimizer.step()

            batch_n = y.size(0)
            train_loss_sum += loss.item() * batch_n
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_examples += batch_n

        if isinstance(optimizer, MuonHybrid):
            scheduler_muon.step()
            if scheduler_adam:
                scheduler_adam.step()
        else:
            scheduler.step()

        torch.cuda.synchronize()
        val_loss, val_acc = evaluate(model, val_loader, amp_enabled)
        epoch_time = time.perf_counter() - epoch_start
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_examples,
            "train_acc": train_correct / train_examples,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
            "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "peak_vram_gb": peak_vram,
        }
        history.append(row)

        print(f"[{dataset_name}|{model_name}|{optimizer_name}|seed={seed}] epoch {epoch}/{args.epochs} train={row['train_acc']:.4f} val={val_acc:.4f} time={epoch_time:.1f}s VRAM={peak_vram:.2f}GB")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, run_dir / "best.pt")

    torch.cuda.synchronize()
    total_time = time.perf_counter() - total_start
    test_loss, test_acc = evaluate(model, test_loader, amp_enabled)

    result = {
        "dataset": dataset_name, "model": model_name, "optimizer": optimizer_name,
        "seed": seed, "epochs": args.epochs, "batch_size": batch_size,
        "best_val_acc": best_val_acc, "test_loss": test_loss, "test_acc": test_acc,
        "total_time_sec": total_time, "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "status": "OK",
    }

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    with open(run_dir / "final.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CUDA vision optimizer benchmark")
    parser.add_argument("--datasets", nargs="+", choices=["mnist", "cifar10"], default=["mnist", "cifar10"])
    parser.add_argument("--models", nargs="+", choices=["resnet", "vit"], default=["resnet", "vit"])
    parser.add_argument("--optimizers", nargs="+", choices=["adam", "lion", "shampoo", "muon"], default=["adam", "lion", "shampoo", "muon"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--muon-lr", type=float, default=2e-2)
    parser.add_argument("--resnet-batch", type=int, default=128)
    parser.add_argument("--vit-batch", type=int, default=32)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--output", default="./results")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this inside a CUDA PyTorch environment.")

    if args.smoke_test:
        args.datasets, args.models, args.optimizers, args.seeds, args.epochs = ["mnist"], ["resnet"], ["adam"], [0], 1

    print("=" * 80)
    print("RIGOROUS CUDA OPTIMIZER BENCHMARK")
    print("=" * 80)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2), "GB")

    results = []
    for dataset in args.datasets:
        for model in args.models:
            for opt in args.optimizers:
                for seed in args.seeds:
                    res = run_experiment(args, dataset, model, opt, seed)
                    results.append(res)

    print("=" * 80)
    print("Benchmark run complete. Results written to:", args.output)
    print("=" * 80)


if __name__ == "__main__":
    main()