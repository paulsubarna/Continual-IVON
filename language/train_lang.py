"""
Continual language pretraining with IVON on EN -> DE -> FR.

Single GPU:
  python train_lang.py

Multi-GPU (4 GPUs):
  torchrun --standalone --nproc_per_node=4 train_lang.py

Resume from a per-language checkpoint:
  python train_lang.py --start_lang de --wandb_run_name <run>
"""

import math
import os
import pickle
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, "..")
from ivon_vcl import IVON_wprior
from model import GPTConfig, GPT

try:
    import wandb
except ImportError:
    wandb = None

# -----------------------------------------------------------------------------
# Default config
# -----------------------------------------------------------------------------
out_dir = './out'
eval_interval = 100
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

wandb_project = 'Lang_40M'
wandb_run_name = 'ivon-lang'

gradient_accumulation_steps = 3 * 8
batch_size = 20
block_size = 1024

n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.0
bias = False

learning_rate = 0.972
max_iters = 15000
weight_decay = 1e-6
beta1 = 0.9
beta2 = 0.99995
grad_clip = 1.0

decay_lr = True
warmup_iters = 500
lr_decay_iters = max_iters
min_lr = 0.0

backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# Continual learning settings
data_root = "./bins/L_40M"
langs = ["en", "de", "fr"]
iters_per_lang = 5000
hess2 = 0.0192

# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--out_dir',                     type=str,   default=out_dir)
parser.add_argument('--eval_interval',               type=int,   default=eval_interval)
parser.add_argument('--log_interval',                type=int,   default=log_interval)
parser.add_argument('--eval_iters',                  type=int,   default=eval_iters)
parser.add_argument('--eval_only',                   action='store_true')
parser.add_argument('--always_save_checkpoint',      action='store_true', default=always_save_checkpoint)
parser.add_argument('--no_save',                     action='store_true')
parser.add_argument('--init_from',                   type=str,   default=init_from,
                    choices=['scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
parser.add_argument('--wandb',                       action='store_true', help='Enable W&B logging')
parser.add_argument('--wandb_project',               type=str,   default=wandb_project)
parser.add_argument('--wandb_run_name',              type=str,   default=wandb_run_name)
parser.add_argument('--ckpt_run_name',               type=str,   default=None)
parser.add_argument('--gradient_accumulation_steps', type=int,   default=gradient_accumulation_steps)
parser.add_argument('--batch_size',                  type=int,   default=batch_size)
parser.add_argument('--block_size',                  type=int,   default=block_size)
parser.add_argument('--n_layer',                     type=int,   default=n_layer)
parser.add_argument('--n_head',                      type=int,   default=n_head)
parser.add_argument('--n_embd',                      type=int,   default=n_embd)
parser.add_argument('--dropout',                     type=float, default=dropout)
parser.add_argument('--bias',                        action='store_true', default=bias)
parser.add_argument('--learning_rate',               type=float, default=learning_rate)
parser.add_argument('--max_iters',                   type=int,   default=max_iters)
parser.add_argument('--weight_decay',                type=float, default=weight_decay)
parser.add_argument('--beta1',                       type=float, default=beta1)
parser.add_argument('--beta2',                       type=float, default=beta2)
parser.add_argument('--hess2',                       type=float, default=hess2)
parser.add_argument('--grad_clip',                   type=float, default=grad_clip)
parser.add_argument('--hess_init',                   type=float, default=0.001)
parser.add_argument('--ess',                         type=float, default=1e10)
parser.add_argument('--de_ess',                      type=float, default=1.44e13)
parser.add_argument('--fr_ess',                      type=float, default=1e15)
parser.add_argument('--clip_radius',                 type=float, default=0.0003)
parser.add_argument('--gamma_s',                     type=float, default=0.0124)
parser.add_argument('--gamma_m',                     type=float, default=0.00106)
parser.add_argument('--de_gamma_s',                  type=float, default=0.0124)
parser.add_argument('--de_gamma_m',                  type=float, default=0.00106)
parser.add_argument('--fr_gamma_s',                  type=float, default=0.0205)
parser.add_argument('--fr_gamma_m',                  type=float, default=0.0131)
parser.add_argument('--decay_lr',                    action='store_true', default=decay_lr)
parser.add_argument('--warmup_iters',                type=int,   default=warmup_iters)
parser.add_argument('--lr_decay_iters',              type=int,   default=lr_decay_iters)
parser.add_argument('--min_lr',                      type=float, default=min_lr)
parser.add_argument('--backend',                     type=str,   default=backend)
parser.add_argument('--device',                      type=str,   default=device)
parser.add_argument('--dtype',                       type=str,   default=dtype)
parser.add_argument('--compile',                     action='store_true', default=compile)
parser.add_argument('--data_root',                   type=str,   default=data_root)
parser.add_argument('--iters_per_lang',              type=int,   default=iters_per_lang)
parser.add_argument('--vocab_size',                  type=int,   default=None)
parser.add_argument('--start_lang',                  type=str,   default=None, choices=["en", "de", "fr"])
parser.add_argument('--ckpt_path',                   type=str,   default=None)

args = parser.parse_args()

out_dir                     = args.out_dir
eval_interval               = args.eval_interval
log_interval                = args.log_interval
eval_iters                  = args.eval_iters
eval_only                   = args.eval_only
always_save_checkpoint      = args.always_save_checkpoint and not args.no_save
init_from                   = args.init_from
wandb_log                   = args.wandb and wandb is not None
wandb_project               = args.wandb_project
wandb_run_name              = args.wandb_run_name
ckpt_run_name               = args.ckpt_run_name if args.ckpt_run_name is not None else wandb_run_name
gradient_accumulation_steps = args.gradient_accumulation_steps
batch_size                  = args.batch_size
block_size                  = args.block_size
hess2                       = args.hess2
n_layer                     = args.n_layer
n_head                      = args.n_head
n_embd                      = args.n_embd
dropout                     = args.dropout
bias                        = args.bias
learning_rate               = args.learning_rate
max_iters                   = args.max_iters
weight_decay                = args.weight_decay
beta1                       = args.beta1
beta2                       = args.beta2
grad_clip                   = args.grad_clip
decay_lr                    = args.decay_lr
warmup_iters                = args.warmup_iters
lr_decay_iters              = args.lr_decay_iters
min_lr                      = args.min_lr
backend                     = args.backend
device                      = args.device
dtype                       = args.dtype
compile                     = args.compile
data_root                   = args.data_root
iters_per_lang              = args.iters_per_lang
start_lang                  = args.start_lang
ckpt_path                   = args.ckpt_path
vocab_size_override         = args.vocab_size

config = vars(args)
print("training config:")
print(config)

# -----------------------------------------------------------------------------
# DDP init
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank       = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Language schedule with 10% replay of old languages
# -----------------------------------------------------------------------------
replay_ratio = 0.10
_replay_schedules: dict[int, list[str]] = {}

def _build_replay_schedule(lang_idx: int) -> list[str]:
    if lang_idx in _replay_schedules:
        return _replay_schedules[lang_idx]
    new_lang = langs[lang_idx]
    if lang_idx == 0:
        schedule = [new_lang] * iters_per_lang
    else:
        n_replay = int(iters_per_lang * replay_ratio)
        n_new = iters_per_lang - n_replay
        old_langs = langs[:lang_idx]
        replay_entries = [old_langs[i % len(old_langs)] for i in range(n_replay)]
        schedule = [new_lang] * n_new + replay_entries
        rng = np.random.RandomState(seed=42 + lang_idx)
        rng.shuffle(schedule)
    _replay_schedules[lang_idx] = schedule
    return schedule

def lang_at_iter(it: int) -> str:
    return langs[min(it // iters_per_lang, len(langs) - 1)]

def sample_training_lang(it: int) -> str:
    lang_idx = min(it // iters_per_lang, len(langs) - 1)
    local_it = it - lang_idx * iters_per_lang
    return _build_replay_schedule(lang_idx)[local_it % iters_per_lang]

def get_bin_path(lang: str, split: str) -> str:
    return os.path.join(data_root, f"{split}_{lang}.bin")

# -----------------------------------------------------------------------------
# Batch loader
# -----------------------------------------------------------------------------
def get_batch(split: str, lang: str):
    data = np.memmap(get_bin_path(lang, split), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Model init
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_root, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', None)
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                  block_size=block_size, bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if vocab_size_override is not None:
        model_args['vocab_size'] = vocab_size_override
    elif meta_vocab_size is not None:
        model_args['vocab_size'] = meta_vocab_size
    else:
        model_args['vocab_size'] = 50304
        print("meta.pkl not found; defaulting vocab_size to 50304")
    model = GPT(GPTConfig(**model_args))

elif init_from == 'resume':
    ckpt_path_resume = os.path.join(out_dir, ckpt_run_name, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path_resume, map_location=device)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint['model_args'][k]
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    model = GPT.from_pretrained(init_from, dict(dropout=dropout))
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

def configure_optim_groups(model, weight_decay):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return [
        {'params': decay_params,   'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]

assert dtype != "float16", "float16 training is not tested"
optim_groups = configure_optim_groups(model, weight_decay)
optimizer = IVON_wprior(
    optim_groups,
    lr=args.learning_rate,
    ess=args.ess,
    mc_samples=1,
    hess_init=args.hess_init,
    beta2=args.beta2,
    weight_decay=weight_decay,
    hess_approx='bonnet',
    sync=True,
    clip_radius=args.clip_radius,
    gamma_s=args.gamma_s,
    gamma_m=args.gamma_m,
)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# -----------------------------------------------------------------------------
# Compile + DDP
# -----------------------------------------------------------------------------
if compile:
    print("compiling the model... (takes ~1 minute)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

gamma_dict = {
    'en': (args.gamma_s,    args.gamma_m),
    'de': (args.de_gamma_s, args.de_gamma_m),
    'fr': (args.fr_gamma_s, args.fr_gamma_m),
}
ess_dict = {'de': args.de_ess, 'fr': args.fr_ess}

# -----------------------------------------------------------------------------
# Resume from a specific language
# -----------------------------------------------------------------------------
if start_lang is not None:
    lang_idx = langs.index(start_lang)
    iter_num = lang_idx * iters_per_lang
    if ckpt_path is None and lang_idx > 0:
        ckpt_path = os.path.join(out_dir, ckpt_run_name, f'ckpt_{langs[lang_idx - 1]}.pt')
    if ckpt_path is not None:
        print(f"start_lang={start_lang}: loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_val_loss = ckpt['best_val_loss']
        optimizer.gamma_s, optimizer.gamma_m = gamma_dict[start_lang]
        for pg in optimizer.param_groups:
            pg['lr'] = args.learning_rate
        if lang_idx > 0:
            optimizer.set_for_new_task(ess=ess_dict[start_lang], hess_init=hess2)
            print(f"Applied set_for_new_task (ess={ess_dict[start_lang]}, hess_init={hess2})")

# -----------------------------------------------------------------------------
# DDP mean reducer
# -----------------------------------------------------------------------------
def ddp_mean(x):
    if ddp and torch.distributed.is_initialized():
        y = x.clone()
        torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
        return y / ddp_world_size
    return x

# -----------------------------------------------------------------------------
# Eval
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_ppl_per_language(langs_list):
    out = {}
    model.eval()
    for lang in langs_list:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            Xv, Yv = get_batch('val', lang)
            with ctx:
                _, loss = model(Xv, Yv)
            losses[k] = loss
        mean_loss = ddp_mean(losses.mean())
        out[lang] = {"val_loss": float(mean_loss.item()),
                     "val_ppl":  float(torch.exp(mean_loss).item())}
    avg_loss = float(np.mean([v["val_loss"] for v in out.values()]))
    out["lang_avg"] = {"val_loss": avg_loss, "val_ppl": math.exp(avg_loss)}
    model.train()
    return out

# -----------------------------------------------------------------------------
# LR schedule (warmup then constant, reset per language)
# -----------------------------------------------------------------------------
def get_lr_wsd(local_it):
    if local_it < warmup_iters:
        return learning_rate * local_it / warmup_iters
    if local_it < iters_per_lang:
        return learning_rate
    return min_lr

# -----------------------------------------------------------------------------
# W&B
# -----------------------------------------------------------------------------
if wandb_log and master_process:
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
current_lang = lang_at_iter(iter_num)
prev_lang = current_lang
print(f"Initial language: {current_lang}")

optimizer.zero_grad(set_to_none=True)
X, Y = get_batch('train', sample_training_lang(iter_num))

t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
last_ppl_stats = None
langs_list = langs[:langs.index(start_lang) + 1] if start_lang is not None else [langs[0]]

while True:
    current_lang = lang_at_iter(iter_num)
    if current_lang != prev_lang:
        print(f"Switching training to language: {current_lang} at iter {iter_num}")
        local_iter_num = 0
        prev_lang = current_lang

    lr = get_lr_wsd(local_iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # Eval
    if iter_num % eval_interval == 0:
        ppl_stats = estimate_ppl_per_language(langs_list)
        last_ppl_stats = ppl_stats
        if master_process:
            msg = f"[eval] iter {iter_num} (lang={lang_at_iter(iter_num)})"
            for lg, v in ppl_stats.items():
                msg += f" | {lg}: ppl={v['val_ppl']:.2f} loss={v['val_loss']:.4f}"
            print(msg)
            if wandb_log:
                wandb.log({"iter": iter_num, "lr": lr,
                           "val/ppl":  ppl_stats["lang_avg"]["val_ppl"],
                           "val/loss": ppl_stats["lang_avg"]["val_loss"]})
            avg_val_loss = ppl_stats["lang_avg"]["val_loss"]
            if avg_val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = min(best_val_loss, avg_val_loss)
                if iter_num > 0:
                    ckpt_dir = os.path.join(out_dir, ckpt_run_name)
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save({
                        'model':      raw_model.state_dict(),
                        'optimizer':  optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num':   iter_num,
                        'best_val_loss': best_val_loss,
                        'config':     config,
                    }, os.path.join(ckpt_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward / backward with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        with optimizer.sampled_params(train=True):
            if ddp:
                model.require_backward_grad_sync = False
            with ctx:
                _, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train', sample_training_lang(iter_num))
            scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and (iter_num % eval_interval != 0) and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        if wandb_log:
            wandb.log({"iter": iter_num, "train/loss": lossf, "lr": lr})
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
              f"mfu {running_mfu*100:.2f}%, lang={lang_at_iter(iter_num)}")

    iter_num += 1
    local_iter_num += 1

    # End-of-language checkpoint + task transition
    if iter_num % iters_per_lang == 0 and iter_num > 0:
        completed_lang = langs[iter_num // iters_per_lang - 1]
        if master_process:
            print(f"Finished '{completed_lang}' at iter {iter_num}; saving checkpoint.")
            ckpt_dir = os.path.join(out_dir, ckpt_run_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'model':      raw_model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'model_args': model_args,
                'iter_num':   iter_num,
                'best_val_loss': best_val_loss,
                'config':     config,
            }, os.path.join(ckpt_dir, f'ckpt_{completed_lang}.pt'))

        next_lang_idx = iter_num // iters_per_lang
        if next_lang_idx < len(langs):
            next_lang = langs[next_lang_idx]
            optimizer.gamma_s, optimizer.gamma_m = gamma_dict[next_lang]
            optimizer.set_for_new_task(ess=ess_dict[next_lang], hess_init=hess2)
            langs_list.append(next_lang)
            print(f"Switching to '{next_lang}' "
                  f"(gamma_s={optimizer.gamma_s}, gamma_m={optimizer.gamma_m})")

    if iter_num >= max_iters:
        break

if master_process and last_ppl_stats is not None:
    s = last_ppl_stats
    print(f"FINAL | avg_ppl={s['lang_avg']['val_ppl']:.4f} "
          f"en={s['en']['val_ppl']:.4f} de={s['de']['val_ppl']:.4f} fr={s['fr']['val_ppl']:.4f}")

if ddp:
    destroy_process_group()
