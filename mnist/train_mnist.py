"""
IVON continual learning on Permuted MNIST.

10 tasks: plain MNIST (task 0) + 9 permuted variants.
Model: MLP  784 -> hidden -> hidden -> 10

IVON accumulates a diagonal Hessian estimate during training. At each task
boundary, set_for_new_task() snapshots the Hessian as a Bayesian prior for
the next task -- no separate Fisher computation needed.
"""

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, "..")
from ivon_vcl import IVON_wprior
from data import get_permuted_mnist_loaders

try:
    import wandb
except ImportError:
    wandb = None


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, num_layers=3, num_classes=10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.log_softmax(self.net(x.view(x.size(0), -1)), dim=1)


def set_seed(seed=42):
    torch.manual_seed(seed)


def train_epoch(model, optimizer, loader, device):
    model.train()
    correct = total_loss = total_samples = 0
    for data, target, _ in loader:
        data, target = data.to(device), target.to(device)
        with optimizer.sampled_params(train=True):
            output = model(data)
            loss = F.nll_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
        optimizer.step()
        bs = data.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        correct += output.argmax(1).eq(target).sum().item()
    return total_loss / total_samples, 100.0 * correct / total_samples


def evaluate(model, loader, device):
    model.eval()
    total_loss = total_correct = total_samples = 0
    with torch.no_grad():
        for data, target, _ in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target).item() * data.size(0)
            total_samples += data.size(0)
            total_correct += output.argmax(1).eq(target).sum().item()
    return total_loss / total_samples, 100.0 * total_correct / total_samples


def evaluate_all_tasks(model, test_loaders, device, num_seen):
    accs = []
    for i in range(num_seen):
        _, acc = evaluate(model, test_loaders[i], device)
        accs.append(acc)
        print(f"  Task {i} acc: {acc:.2f}%")
    avg = float(np.mean(accs))
    print(f"  Average accuracy: {avg:.2f}%")
    return avg


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

    optimizer = IVON_wprior(
        model.parameters(),
        lr=args.lr_task1,
        ess=args.ess_task1,
        hess_init=args.hess_init_task1,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        clip_radius=0.01,
        gamma_m=args.gamma_m,
        gamma_s=args.gamma_s,
    )

    if wandb is not None and args.wandb:
        wandb.init(project="MNIST-CIL", config=vars(args))

    all_avg_accs = []
    t_start = time.time()

    for task_idx, train_loader in enumerate(train_loaders):
        print(f"\n{'='*60}\nTask {task_idx + 1}/{args.num_tasks}\n{'='*60}")

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, optimizer, train_loader, device)
            print(f"  Epoch {epoch+1:02d}/{args.epochs}  "
                  f"loss={train_loss:.4f}  acc={train_acc:.1f}%", end="\r")
        print()

        avg_acc = evaluate_all_tasks(model, test_loaders, device, num_seen=task_idx + 1)
        all_avg_accs.append(avg_acc)

        if wandb is not None and args.wandb:
            wandb.log({"task": task_idx + 1, "avg_acc_seen": avg_acc,
                       "train_loss": train_loss, "train_acc": train_acc})

        if task_idx < args.num_tasks - 1:
            scale = args.prior_ess_scale_t1 if task_idx == 0 else args.prior_ess_scale
            optimizer.set_for_new_task(ess=args.ess * scale, hess_init=args.hess_2)
            if task_idx == 0:
                for g in optimizer.param_groups:
                    g["lr"] = args.lr

    elapsed = time.time() - t_start
    final_avg = float(all_avg_accs[-1])
    print(f"\nFinal average accuracy (all {args.num_tasks} tasks): {final_avg:.2f}%")
    print(f"Total time: {elapsed:.1f}s")

    if wandb is not None and args.wandb:
        wandb.log({"final_avg_acc": final_avg})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IVON on Permuted MNIST")
    parser.add_argument("--num_tasks",          type=int,   default=10)
    parser.add_argument("--epochs",             type=int,   default=30)
    parser.add_argument("--batch_size",         type=int,   default=256)
    parser.add_argument("--test_batch_size",    type=int,   default=512)
    parser.add_argument("--lr",                 type=float, default=0.021601,
                        help="LR for tasks 2+")
    parser.add_argument("--lr_task1",           type=float, default=5.311e-3,
                        help="LR for task 1")
    parser.add_argument("--ess",                type=float, default=39826315.35)
    parser.add_argument("--ess_task1",          type=float, default=1e7,
                        help="ESS for task 1")
    parser.add_argument("--hess_init",          type=float, default=0.01622)
    parser.add_argument("--hess_init_task1",    type=float, default=0.01061,
                        help="hess_init for task 1")
    parser.add_argument("--hess_2",             type=float, default=0.002968)
    parser.add_argument("--beta1",              type=float, default=0.9)
    parser.add_argument("--beta2",              type=float, default=0.99995)
    parser.add_argument("--weight_decay",       type=float, default=1e-6)
    parser.add_argument("--prior_ess_scale",    type=float, default=0.088413)
    parser.add_argument("--prior_ess_scale_t1", type=float, default=0.095762)
    parser.add_argument("--hidden_dim",         type=int,   default=400)
    parser.add_argument("--num_layers",         type=int,   default=3)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--gamma_m",            type=float, default=0.911427)
    parser.add_argument("--gamma_s",            type=float, default=0.797672)
    parser.add_argument("--cache_dir",          type=str,   default="./data/perm_mnist_cache")
    parser.add_argument("--wandb",              action="store_true", help="Enable W&B logging")
    args = parser.parse_args()
    run(args)
