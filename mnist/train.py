"""
train_ivon.py — CoVON continual learning on Permuted MNIST.

"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F

from covon import CoVON_wprior as IVON_wprior  # CoVON is a drop-in replacement for IVON
from data import get_permuted_mnist_loaders


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400,
                 num_layers: int = 2, num_classes: int = 10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_epoch(model, optimizer, loader, criterion, device):
    model.train()
    correct = 0
    total_loss = 0
    total_samples = 0
    for data, target, _ in loader:
        data, target = data.to(device), target.to(device)
        with optimizer.sampled_params(train=True):
            output = model(data)
            loss = F.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size  # scale loss by batch size
        total_samples += batch_size

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / total_samples
    train_acc = 100.0 * correct / total_samples
    return avg_loss, train_acc


def evaluate(model, loader, criterion, device):
    """Eval using the stored mean parameters (no sampling needed)."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for data, target, _ in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.cross_entropy(output, target).item() * data.size(0)
            total_samples += data.size(0)
            total_correct += output.argmax(1).eq(target).sum().item()

    return total_loss / total_samples, 100.0 * total_correct / total_samples


def evaluate_all_tasks(model, test_loaders, criterion, device, num_seen):
    """Evaluate on all seen tasks; print per-task and return average accuracy."""
    accs = []
    for i in range(num_seen):
        _, acc = evaluate(model, test_loaders[i], criterion, device)
        accs.append(acc)
        print(f"  Task {i} acc: {acc:.2f}%")
    avg = float(np.mean(accs))
    print(f"  Average accuracy: {avg:.2f}%")
    return avg


# ── Training loop ─────────────────────────────────────────────────────────────

def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")
    train_loaders, test_loaders, _ = get_permuted_mnist_loaders(
        cache_dir=args.cache_dir,
        num_tasks=args.num_tasks,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )

    model = MLP(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    criterion = F.cross_entropy

    optimizer = IVON_wprior(
        model.parameters(),
        lr=args.lr_task1,
        ess=args.ess_task1,
        hess_init=args.hess_init_task1,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        clip_radius=args.clip_radius,
        gamma_m=args.gamma_m,
        gamma_s=args.gamma_s,
    )

    wandb.init(
        project="MNIST-CIL",
        name=f"covon_ess{args.ess:.0e}_hi{args.hess_init}",
        config=vars(args),
    )

    all_avg_accs = []
    t_start = time.time()

    for task_idx, train_loader in enumerate(train_loaders):
        print(f"\n{'='*60}\nTask {task_idx + 1}/{args.num_tasks}\n{'='*60}")

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, optimizer, train_loader, criterion, device)
            print(f"  Epoch {epoch+1:02d}/{args.epochs}  "
                  f"loss={train_loss:.4f}  acc={train_acc:.1f}%", end="\r")

        print()

        # Evaluate on all tasks seen so far
        avg_acc = evaluate_all_tasks(model, test_loaders, criterion, device,
                                     num_seen=task_idx + 1)
        all_avg_accs.append(avg_acc)

        wandb.log({
            "task": task_idx + 1,
            "avg_acc_seen": avg_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
        })

        if task_idx < args.num_tasks - 1:
            if task_idx == args.num_tasks - 2:
                ess_ = 5e5
            else:
                scale = args.prior_ess_scale_t1 if task_idx == 0 else args.prior_ess_scale
                ess_ = args.ess * scale

            optimizer.set_for_new_task(ess=ess_, hess_init=args.hess_2)
            if task_idx == 0:
                for g in optimizer.param_groups:
                    g["lr"] = args.lr

    elapsed = time.time() - t_start
    final_avg = float(all_avg_accs[-1])
    print(f"\nFinal average accuracy (all {args.num_tasks} tasks): {final_avg:.2f}%")
    print(f"Total time: {elapsed:.1f}s")
    wandb.log({"final_avg_acc": final_avg})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IVON on Permuted MNIST")
    parser.add_argument("--num_tasks",          type=int,   default=10)
    parser.add_argument("--epochs",             type=int,   default=40)
    parser.add_argument("--batch_size",         type=int,   default=256)
    parser.add_argument("--test_batch_size",    type=int,   default=256)
    parser.add_argument("--lr",                 type=float, default=0.02305726806421391,
                        help="LR for tasks 2+")
    parser.add_argument("--lr_task1",           type=float, default=0.0045082667593017885,
                        help="LR for task 1")
    parser.add_argument("--ess",                type=float, default=35017987.29981349)
    parser.add_argument("--ess_task1",          type=float, default=8535632.142875116,
                        help="ESS for task 1")
    parser.add_argument("--hess_init",          type=float, default=0.01137719048726074)
    parser.add_argument("--hess_init_task1",    type=float, default=0.00849137804278021,
                        help="hess_init for task 1")
    parser.add_argument("--hess_2",             type=float, default=0.002607542738206092)
    parser.add_argument("--beta1",              type=float, default=0.9)
    parser.add_argument("--beta2",              type=float, default=0.9999183952756989)
    parser.add_argument("--weight_decay",       type=float, default=9.78493388866844e-07)
    parser.add_argument("--prior_ess_scale",    type=float, default=0.0635390831329116)
    parser.add_argument("--prior_ess_scale_t1", type=float, default=0.0981111812780596)
    parser.add_argument("--hidden_dim",         type=int,   default=200)
    parser.add_argument("--num_layers",         type=int,   default=2)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--gamma_m",            type=float, default=0.882573610419212)
    parser.add_argument("--gamma_s",            type=float, default=0.8583386063524316)
    parser.add_argument("--cache_dir",          type=str,   default="./data/perm_mnist_cache")
    parser.add_argument("--wandb",              action="store_true", help="Enable W&B logging")
    parser.add_argument("--clip_radius",        type=float, default=0.01235509595907778)
    args = parser.parse_args()

    run(args)