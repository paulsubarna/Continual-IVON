# Supplementary Code: IVON for Continual Learning

This repository contains the training code accompanying the paper.

## Method

We introduce CoVON (Continual Variational Online Newton), a fast and slow VCL optimization algorithm.

## Structure

```
ivon_vcl.py          # IVON optimizer with continual learning prior (shared)
mnist/
    train_mnist.py   # Permuted MNIST (10 tasks)
    data.py          # Data pipeline (downloads and caches automatically)
language/
    train_lang.py    # Continual language pretraining: EN -> DE -> FR
    model.py         # GPT model definition
```

## Requirements

```
torch
torchvision      # mnist only
numpy
tiktoken         # language only (for tokenization)
```

Optional: `wandb` for logging (pass `--wandb` flag).

## Usage

### Permuted MNIST (10 tasks)

```bash
cd mnist
python train_mnist.py
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--num_tasks` | 10 | Number of tasks |
| `--epochs` | 30 | Epochs per task |
| `--lr_task1` | 5.311e-3 | Learning rate for task 1 |
| `--lr` | 0.021601 | Learning rate for tasks 2+ |
| `--ess` | 3.98e7 | Effective sample size |
| `--gamma_m` | 0.911 | Prior mean interpolation factor |
| `--gamma_s` | 0.798 | Prior precision accumulation factor |

### Continual Language Pretraining (EN -> DE -> FR)

Data preparation: place tokenized `.bin` files in `language/bins/L_40M/` as `train_en.bin`, `val_en.bin`, `train_de.bin`, etc.

```bash
# Single GPU
cd language
python train_lang.py --wandb_run_name my_run

# Multi-GPU (4 GPUs)
torchrun --standalone --nproc_per_node=4 train_lang.py --wandb_run_name my_run

# Resume from DE
python train_lang.py --start_lang de --wandb_run_name my_run
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--iters_per_lang` | 5000 | Optimizer steps per language |
| `--ess` | 1e10 | ESS for EN |
| `--de_ess` | 1.44e13 | ESS for DE |
| `--fr_ess` | 1e15 | ESS for FR |
| `--gamma_s` / `--de_gamma_s` / `--fr_gamma_s` | 0.0124 | Prior precision factor per language |
| `--gamma_m` / `--de_gamma_m` / `--fr_gamma_m` | 0.00106 | Prior mean factor per language |
| `--hess2` | 0.0192 | Hessian reset value at task boundaries |
