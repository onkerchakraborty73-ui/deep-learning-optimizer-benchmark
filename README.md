
# Rigorous Vision Optimizer Benchmark

This project compares:

- ResNet-18
- ViT-Tiny
- MNIST
- CIFAR-10
- AdamW
- Lion
- Shampoo
- Muon

It supports multi-seed experiments, validation-based checkpoint selection,
test evaluation, bootstrap confidence intervals, paired permutation tests,
effect sizes, convergence/runtime analysis, memory tracking, update/parameter
norm tracking, and LR/weight-decay/batch-size ablations.

## Install

```bash
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch installation if you have an NVIDIA GPU.

## Smoke test

```bash
python optimizer_benchmark.py --datasets mnist --models resnet --optimizers adamw --seeds 0 --epochs 2
```

## Recommended study

```bash
python optimizer_benchmark.py --datasets mnist cifar10 --models resnet vit --optimizers adamw lion shampoo muon --seeds 0 1 2 3 4 --epochs 50 --make-plots
```

## Full 10-seed study

```bash
python optimizer_benchmark.py --datasets mnist cifar10 --models resnet vit --optimizers adamw lion shampoo muon --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 100 --make-plots
```

## Ablations

Learning rate:

```bash
python optimizer_benchmark.py --mode lr_ablation --dataset cifar10 --model resnet --optimizer adamw --seeds 0 1 2 --epochs 50 --make-plots
```

Weight decay:

```bash
python optimizer_benchmark.py --mode wd_ablation --dataset cifar10 --model resnet --optimizer adamw --seeds 0 1 2 --epochs 50 --make-plots
```

Batch size:

```bash
python optimizer_benchmark.py --mode batch_ablation --dataset cifar10 --model resnet --optimizer adamw --seeds 0 1 2 --epochs 50 --make-plots
```

## Output

`results/` contains:

- per-run `history.csv`
- per-run `metadata.json`
- per-run `final.json`
- best validation checkpoint `best.pt`
- `all_final_results.csv`
- `summary.csv`
- `pairwise_statistics.csv`
- publication-style figures

## Experimental caution

Do not select hyperparameters from the test set. Use the validation split for
model selection, then evaluate the selected checkpoint once on the test set.

For a paper, record the exact Python/PyTorch/CUDA/GPU versions and ideally
containerize the environment.
