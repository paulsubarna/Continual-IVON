# Official Repository of CoVON: Fast and Slow Variational Continual Learning

** This repository is in progress **
### Abstract 
Continual learning remains a major challenge for modern deep networks, partly because commonly used optimizers lack inherent mechanisms for continual adaptation. 
One such natural mechanism is ‘fast and slow adaptation’ to balance stability and plasticity. 
This mechanism has deep roots in neuroscience and biology, but there is no consensus on how to
best incorporate it in commonly used optimizers. Here, we show that this can be easily done via the VCL framework, where past posteriors are used as priors in the future. Our key idea is to incorporate slow adaptation via merging of past posteriors to slow down the drift in the knowledge as learning progresses. The merged posterior is then used as the prior in the VCL update to implement the fast-weight updates. These steps can be seamlessly implemented in the IVON
optimizer, whose form and costs are nearly identical to that of Adam. We call this new optimizer the Continual IVON (CoVON) optimizer and show that
it not only consistently improves over existing VCL optimizers, but also performs better than other weight-regularization strategies across domain-incremental learning, continual pre-training, and fine-tuning of large language models.

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

## Acknowledgement
Domain-Incremental Settings: We thank [PyCIL](https://github.com/G-U-N/PyCIL),  [SPrompt](https://github.com/iamwangyabin/S-Prompts) for their wonderful framework and codes!  
Continual Pretraining and Finetuning of LLMs: We thank [IVON](https://github.com/team-approx-bayes/ivon-experiments) for their wonderful framework! 

## Citations

```bibtex
@ONLINE{wikidump,
    author = "Wikimedia Foundation",
    title  = "Wikimedia Downloads",
    url    = "https://dumps.wikimedia.org"
}

@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388}, 
}
```

