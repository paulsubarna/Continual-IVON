# Supplementary Code: CoVON: Fast and Slow Variational Continual Learning

This repository contains the training code accompanying the paper.

## Method

We introduce CoVON (Continual Variational Online Newton), a fast and slow VCL optimization algorithm.

## Structure

```
covon.py          # CoVON optimizer for fast and slow VCL training(shared)
mnist/
    train_mnist.py   # Permuted MNIST (10 tasks)
    data.py          # Data pipeline (downloads and caches automatically)
vision/
    train_core50.py       # CORe50 continual learning (8 sessions, ViT-B/16)
    train_domainnet.py    # DomainNet continual learning (6 domains, ViT-B/16)
language/
    train_lang.py    # Continual language pretraining: EN -> DE -> FR
    finetune_reasoning.py # Continual finetuning on Math -> Code -> CommonSense (Arc)
    model.py         # GPT model definition
```

## Datasets

### Permuted MNIST
Downloaded automatically by torchvision on first run.

### CORe50
Please refer to [CORe50 Project](https://vlomonaco.github.io/core50/index.html#dataset) and download the file shown below:
```
CORe50
├── core50_imgs.npz
├── labels.pkl
├── LUP.pkl
└── paths.pkl
```

### DomainNet
Please refer to the [DomainNet project page](http://ai.bu.edu/M3SDA/) and download the six domain splits. Organize as:
```
DomainNet/
├── clipart/
│   ├── train/
│   └── test/
├── infograph/
│   ├── train/
│   └── test/
├── painting/
│   ├── train/
│   └── test/
├── quickdraw/
│   ├── train/
│   └── test/
├── real/
│   ├── train/
│   └── test/
└── sketch/
    ├── train/
    └── test/
```
Run the training script from the `DomainNet/` directory (i.e. `cd vision && ln -s /path/to/DomainNet/* .` or set paths accordingly).

### Language (EN / DE / FR)
We used the [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) dataset from Hugging Face for language pretraining.

## Requirements

```
torch
torchvision      # mnist and vision tasks
numpy
timm             # vision tasks (ViT-B/16)
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

### Vision: CORe50 (8 sessions)

Place the CORe50 flat directory at `vision/core50_flat/` (or adjust `DATA_DIR` in the script).

```bash
# Single GPU
cd vision
python train_core50.py

# Multi-GPU (4 GPUs)
torchrun --standalone --nproc_per_node=4 train_core50.py

# Resume from task 3
python train_core50.py --start_task 3
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--lr_task0` | 1e-4 | Learning rate for task 0 |
| `--lr` | 5.798e-6 | Learning rate for tasks 1+ |
| `--ess` | 1.629e7 | Effective sample size |
| `--epochs` | 20 | Epochs per session |
| `--hess` | 0.005 | Initial Hessian value |
| `--hess2` | 1.517e-3 | Hessian reset value at task boundaries |
| `--gamma_m` | 0.312 | Prior mean interpolation factor |
| `--gamma_s` | 0.032 | Prior precision accumulation factor |

### Vision: DomainNet (6 domains)

Run from the directory containing the six domain folders.

```bash
# Single GPU
cd vision
python train_domainnet.py

# Multi-GPU (4 GPUs)
torchrun --standalone --nproc_per_node=4 train_domainnet.py

# Resume from task 3
python train_domainnet.py --start_task 3
```

Per-domain hyperparameters are pre-loaded from `DOMAIN_HPARAMS` in the script (from Bayesian search). Pass `--ignore_domain_hparams` to override with CLI arguments.

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

## Acknowledgement
Domain-Incremental Settings: We thank [PyCIL](https://github.com/G-U-N/PyCIL) and [S-Prompts](https://github.com/iamwangyabin/S-Prompts) for their wonderful framework and codes!  
Continual Pretraining and Finetuning of LLMs: We thank [IVON](https://github.com/team-approx-bayes/ivon-experiments) for their wonderful framework! 

## Citations 

@ONLINE{wikidump,
    author = "Wikimedia Foundation",
    title  = "Wikimedia Downloads",
    url    = "https://dumps.wikimedia.org"
}

