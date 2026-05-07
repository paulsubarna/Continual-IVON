"""
CoVON continual learning on DomainNet (6 domains: clipart -> infograph -> painting
-> quickdraw -> real -> sketch).

Model: ViT-B/16 (pretrained via timm)

Single GPU:
    python train_domainnet.py

Multi-GPU:
    torchrun --nproc_per_node=4 train_domainnet.py

Resume from task 3:
    python train_domainnet.py --start_task 3 --checkpoint_dir ./outputs
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

sys.path.insert(0, "..")
from covon import CoVON_wprior

try:
    import wandb
except ImportError:
    wandb = None

# ---------------------------------------------------------------------------
# Best per-domain hyperparameters from Bayesian search
# ---------------------------------------------------------------------------
DOMAIN_HPARAMS = {
    0: dict(lr=9.956e-05, hess=5.421e-03, hess2=3.174e-03, beta2=0.999172, wd=1.596e-04, ess=3.892e+08, clip_radius=0.03260, epochs=30),  # clipart
    1: dict(lr=1.758e-04, hess=1.081e-02, hess2=5.151e-03, beta2=0.999348, wd=1.442e-04, ess=1.081e+08, clip_radius=0.02268, epochs=20),  # infograph
    2: dict(lr=1.536e-05, hess=5.091e-03, hess2=9.254e-02, beta2=0.999009, wd=5.36e-05,  ess=1.960e+10, clip_radius=0.08842, gamma_m=0.01715, gamma_s=0.03021, epochs=15),  # painting
    3: dict(lr=4.716e-05, hess=5.091e-03, hess2=8.627e-02, beta2=0.999009, wd=1.236e-04, ess=4.834e+09, clip_radius=0.06010, gamma_m=0.02051, gamma_s=0.02051, epochs=19),  # quickdraw
    4: dict(lr=1.162e-05, hess=8.580e-02, hess2=4.539e-02, beta2=0.999800, wd=1.245e-04, ess=1.568e+08, clip_radius=0.03305, gamma_m=0.1,     gamma_s=0.1,     epochs=19),  # real
    5: dict(lr=2.369e-05, hess=8.627e-02, hess2=2.352e-02, beta2=0.999593, wd=7.979e-05, ess=2.051e+08, clip_radius=0.00368, epochs=28),  # sketch
}

# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_initialized() else 0

def is_main_process():
    return get_rank() == 0

def get_device():
    if torch.cuda.is_available():
        if is_dist_initialized():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cuda")
    return torch.device("cpu")

def reduce_sum(value, device):
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if is_dist_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, get_device()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                            init_method="env://")
    return True, rank, world_size, get_device()

def cleanup_distributed():
    if is_dist_initialized():
        dist.barrier()
        dist.destroy_process_group()

def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model

def create_model(rank, distributed, num_classes):
    if distributed and rank != 0:
        dist.barrier()
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    if distributed and rank == 0:
        dist.barrier()
    return model

def optimizer_to(optim, device):
    for state in optim.state.values():
        if not isinstance(state, dict):
            continue
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    for group in optim.param_groups:
        for k, v in list(group.items()):
            if torch.is_tensor(v):
                group[k] = v.to(device)
            elif isinstance(v, list):
                group[k] = [x.to(device) if torch.is_tensor(x) else x for x in v]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class IndxImageFolder(ImageFolder):
    def __init__(self, root, transform=None, num_classes=None):
        super().__init__(root, transform)
        self.num_classes = num_classes

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path

# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(loader, model, optimizer, num_epochs, task_idx, criterion=nn.CrossEntropyLoss()):
    model.train()
    device = get_device()
    task_losses = []
    for epoch in range(num_epochs):
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        total_loss = total_samples = correct = 0
        batch_iter = tqdm(loader, desc=f"Task {task_idx+1} | Epoch {epoch+1}/{num_epochs}",
                          leave=False, dynamic_ncols=True) if is_main_process() else loader
        for data, target, _ in batch_iter:
            data, target = data.to(device), target.to(device)
            with optimizer.sampled_params(train=True):
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    output = model(data)
                    loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
            optimizer.step()
            bs = data.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            correct += output.argmax(1).eq(target).sum().item()
            if is_main_process() and hasattr(batch_iter, 'set_postfix'):
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")
        total_loss    = reduce_sum(total_loss, device)
        total_samples = reduce_sum(total_samples, device)
        correct       = reduce_sum(correct, device)
        avg_loss  = total_loss / total_samples
        train_acc = 100.0 * correct / total_samples
        task_losses.append(avg_loss)
        if is_main_process():
            print(f"  epoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}  acc={train_acc:.2f}%")
    return train_acc, task_losses


def evaluate(loader, model, criterion=nn.CrossEntropyLoss()):
    model.eval()
    device = get_device()
    total_loss = total_samples = correct = 0
    with torch.no_grad():
        for data, target, _ in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            bs = data.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            correct += output.argmax(1).eq(target).sum().item()
    total_loss    = reduce_sum(total_loss, device)
    total_samples = reduce_sum(total_samples, device)
    correct       = reduce_sum(correct, device)
    return total_loss / total_samples, 100.0 * correct / total_samples


def test_all(data_test, task_idx, model):
    accs, losses = [], []
    for i, loader in enumerate(data_test.values()):
        if i >= task_idx:
            break
        loss, acc = evaluate(loader, model)
        accs.append(acc)
        losses.append(loss)
        if is_main_process():
            print(f"  task {i} acc={acc:.2f}%  loss={loss:.4f}")
    avg = float(np.mean(accs))
    if is_main_process():
        print(f"  avg acc: {avg:.2f}%")
        if wandb is not None:
            wandb.log({"test/avg_acc": avg, "task": task_idx})
    return avg

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_training(args):
    distributed, rank, world_size, device = setup_distributed()
    set_seed(args.seed)
    cudnn.benchmark = True

    if is_main_process():
        print(args, end="\n\n")

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    domains = all_domains[:args.num_tasks]

    data_train, data_test = {}, {}
    for i, name in enumerate(domains):
        if is_main_process():
            print(f"Loading {name}...")
        ds = IndxImageFolder(root=f"./{name}/train", transform=transform_train, num_classes=345)
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        data_train[f"task_{i}"] = DataLoader(ds, batch_size=args.batch_size,
                                              shuffle=(sampler is None), sampler=sampler,
                                              num_workers=4, pin_memory=torch.cuda.is_available())
        ds_test = IndxImageFolder(root=f"./{name}/test", transform=transform_test, num_classes=345)
        data_test[f"task_{i}"] = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                                             num_workers=2, pin_memory=torch.cuda.is_available())

    model = create_model(rank, distributed, num_classes=345)
    model.to(device)
    if distributed:
        dist.barrier()
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
        dist.barrier()

    base_model = unwrap_model(model)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if is_main_process() and wandb is not None and args.wandb:
        wandb.init(project="DNet-CoVON", config=vars(args))

    if args.start_task > 0:
        ckpt = os.path.join(args.checkpoint_dir, f"model_task_{args.start_task - 1}.pth")
        base_model.load_state_dict(torch.load(ckpt, map_location=device))
        if is_main_process():
            print(f"Resumed model from {ckpt}")

    optimizer = None
    avg_acc_all = []

    for task_idx, train_loader in enumerate(data_train.values()):
        if task_idx >= args.num_tasks:
            break
        if task_idx < args.start_task:
            continue

        hp = {} if args.ignore_domain_hparams else DOMAIN_HPARAMS.get(task_idx, {})
        task_lr          = hp.get("lr",          args.lr)
        task_hess        = hp.get("hess",        args.hess)
        task_hess2       = hp.get("hess2",       args.hess2)
        task_beta2       = hp.get("beta2",       args.beta2)
        task_wd          = hp.get("wd",          args.wd)
        task_ess         = hp.get("ess",         args.ess)
        task_clip_radius = hp.get("clip_radius", args.clip_radius)
        task_gamma_m     = hp.get("gamma_m",     args.gamma_m)
        task_gamma_s     = hp.get("gamma_s",     args.gamma_s)
        task_epochs      = hp.get("epochs",      args.epochs)

        if is_main_process():
            print(f"\n[Task {task_idx} — {domains[task_idx]}] "
                  f"lr={task_lr:.3e} hess={task_hess:.3e} ess={task_ess:.3e} epochs={task_epochs}")

        if task_idx == 0:
            optimizer = CoVON_wprior(model.parameters(), lr=task_lr, ess=task_ess, mc_samples=1,
                                    hess_init=task_hess, beta2=task_beta2, weight_decay=task_wd,
                                    hess_approx='bonnet', sync=True, clip_radius=task_clip_radius,
                                    gamma_m=task_gamma_m, gamma_s=task_gamma_s)
        elif task_idx == args.start_task and args.start_task > 0:
            optimizer = CoVON_wprior(model.parameters(), lr=task_lr, ess=task_ess, mc_samples=1,
                                    hess_init=task_hess, beta2=task_beta2, weight_decay=task_wd,
                                    hess_approx='bonnet', sync=True, clip_radius=task_clip_radius,
                                    gamma_m=task_gamma_m, gamma_s=task_gamma_s)
            ckpt_opt = os.path.join(args.checkpoint_dir, f"optimizer_task_{args.start_task - 1}.pth")
            optimizer.load_state_dict(torch.load(ckpt_opt, map_location="cpu"))
            optimizer_to(optimizer, device)
            optimizer.set_for_new_task(ess=task_ess, hess_init=task_hess2, model=base_model)
        else:
            optimizer.gamma_m = task_gamma_m
            optimizer.gamma_s = task_gamma_s
            for group in optimizer.param_groups:
                group['lr'] = task_lr
            optimizer.set_for_new_task(ess=task_ess, hess_init=task_hess2, model=base_model)

        t0 = time.time()
        train(train_loader, model, optimizer, task_epochs, task_idx)

        if is_dist_initialized():
            dist.barrier()

        avg_acc = test_all(data_test, task_idx + 1, model)
        avg_acc_all.append(avg_acc)

        if is_main_process():
            print(f"Task {task_idx+1} done in {time.time()-t0:.1f}s  avg_acc={avg_acc:.2f}%")
            torch.save(base_model.state_dict(),
                       os.path.join(args.checkpoint_dir, f"model_task_{task_idx}.pth"))
            torch.save(optimizer.state_dict(),
                       os.path.join(args.checkpoint_dir, f"optimizer_task_{task_idx}.pth"))

        if is_dist_initialized():
            dist.barrier()

    if is_main_process():
        print(f"\nDone. Per-task avg acc: {avg_acc_all}")
        print(f"HPARAM_SEARCH_METRIC: {float(np.mean(avg_acc_all)):.6f}")
        if wandb is not None and args.wandb:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",                   type=float, default=1.054e-05)
    parser.add_argument("--hess",                 type=float, default=5.091e-03)
    parser.add_argument("--hess2",                type=float, default=3.064e-03)
    parser.add_argument("--beta2",                type=float, default=0.999923)
    parser.add_argument("--wd",                   type=float, default=1.919e-04)
    parser.add_argument("--ess",                  type=float, default=878213573)
    parser.add_argument("--clip_radius",          type=float, default=0.03676)
    parser.add_argument("--gamma_m",              type=float, default=0.1)
    parser.add_argument("--gamma_s",              type=float, default=0.1)
    parser.add_argument("--epochs",               type=int,   default=20)
    parser.add_argument("--batch_size",           type=int,   default=32)
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--num_tasks",            type=int,   default=6)
    parser.add_argument("--checkpoint_dir",       type=str,   default="./outputs/")
    parser.add_argument("--start_task",           type=int,   default=0)
    parser.add_argument("--ignore_domain_hparams", action="store_true",
                        help="Use CLI args for all tasks instead of per-domain best hparams")
    parser.add_argument("--wandb",                action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    run_training(args)
