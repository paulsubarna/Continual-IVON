import argparse
import sys
import os
import time
import random
sys.path.insert(0, "..")
from covon import CoVON_wprior
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import transforms
from torchvision.datasets import ImageFolder

import timm

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import wandb
except ImportError:
    wandb = None

DATA_DIR = "./core50_flat"
TRAIN_SESSIONS = [f"s{i}" for i in range(1, 9)]   # s1-s8
TEST_SESSIONS  = ["s9", "s10", "s11"]
NUM_CLASSES    = 50


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_initialized() else 0

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0

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
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                            init_method="env://")
    if wandb is not None and rank != 0:
        os.environ["WANDB_MODE"] = "disabled"
    return True, rank, world_size, get_device()

def cleanup_distributed():
    if is_dist_initialized():
        dist.barrier()
        dist.destroy_process_group()

def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model

def create_model_for_rank(rank, distributed, num_classes):
    if distributed and rank != 0:
        dist.barrier()
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    if distributed and rank == 0:
        dist.barrier()
    return model

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IndxImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


def train(loader, model, optimizer, num_epochs, task_idx, criterion=nn.CrossEntropyLoss()):
    model.train()
    device = get_device()
    task_losses = []
    for epoch in range(num_epochs):
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        total_loss = total_samples = correct = 0
        batch_iter = tqdm(loader, desc=f"Task {task_idx+1} | Epoch {epoch+1}/{num_epochs}",
                          leave=False, dynamic_ncols=True) if (is_main_process() and tqdm) else loader
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
            if tqdm and isinstance(batch_iter, tqdm):
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")
        total_loss    = reduce_sum(total_loss, device)
        total_samples = reduce_sum(total_samples, device)
        correct       = reduce_sum(correct, device)
        avg_loss  = total_loss / total_samples
        train_acc = 100.0 * correct / total_samples
        task_losses.append(avg_loss)
        if is_main_process():
            print(f"  epoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}  acc={train_acc:.2f}%")
            if wandb is not None:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/acc": train_acc,
                    "train/epoch": task_idx * num_epochs + epoch + 1,
                    "task": task_idx + 1,
                })
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
    for name, loader in data_test.items():
        loss, acc = evaluate(loader, model)
        accs.append(acc)
        losses.append(loss)
        if is_main_process():
            print(f"  test [{name}] after task {task_idx+1}: {acc:.2f}%  loss={loss:.4f}")
    avg_loss = float(np.mean(losses))
    avg = float(np.mean(accs))
    if is_main_process():
        print(f"  avg test acc: {avg:.2f}%")
        if wandb is not None:
            wandb.log({
                "test/avg_loss": avg_loss,
                "test/avg_acc": avg,
                "task": task_idx,
            })
    return avg


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

    train_sessions = TRAIN_SESSIONS[:args.num_tasks]
    data_train, data_test = {}, {}

    for i, sess in enumerate(train_sessions):
        if is_main_process():
            print(f"Loading train session {sess}...")
        ds = IndxImageFolder(root=os.path.join(DATA_DIR, sess), transform=transform_train)
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        data_train[f"task_{i}"] = DataLoader(ds, batch_size=args.batch_size,
                                              shuffle=(sampler is None), sampler=sampler,
                                              num_workers=4, pin_memory=torch.cuda.is_available())

    for sess in TEST_SESSIONS:
        if is_main_process():
            print(f"Loading test session {sess}...")
        ds = IndxImageFolder(root=os.path.join(DATA_DIR, sess), transform=transform_test)
        data_test[sess] = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=2, pin_memory=torch.cuda.is_available())

    model = create_model_for_rank(rank, distributed, NUM_CLASSES)
    model.to(device)
    if distributed:
        dist.barrier()
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
        dist.barrier()

    base_model = unwrap_model(model)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if is_main_process() and wandb is not None:
        wandb.init(
            project="core50-covon",
            config=dict(
                lr=args.lr, epochs=args.epochs, hess_init=args.hess,
                weight_decay=args.wd, beta=args.beta2, ess=args.ess,
                hess2=args.hess2, batch_size=args.batch_size,
                world_size=world_size,
            ),
        )

    # Resume
    if args.start_task > 0:
        ckpt = os.path.join(args.checkpoint_dir, f"model_task_{args.start_task - 1}.pth")
        base_model.load_state_dict(torch.load(ckpt, map_location=device))
        if is_main_process():
            print(f"Resumed model from {ckpt}")

    optimizer = None
    avg_acc_all = []

    for task_idx, (_, train_loader) in enumerate(data_train.items()):
        if task_idx < args.start_task:
            continue

        hp = dict(lr=args.lr, hess=args.hess, hess2=args.hess2, beta2=args.beta2,
                  wd=args.wd, ess=args.ess, clip_radius=args.clip_radius,
                  epochs=args.epochs, gamma_m=args.gamma_m, gamma_s=args.gamma_s)

        if is_main_process():
            print(f"\n[Task {task_idx+1}/{len(data_train)}  session={train_sessions[task_idx]}]  "
                  f"lr={hp['lr']:.3e}  hess={hp['hess']:.3e}  ess={hp['ess']:.3e}  epochs={hp['epochs']}")

        if task_idx == 0:
            optimizer = CoVON_wprior(model.parameters(),
                                    lr=args.lr_task0, ess=hp['ess'], mc_samples=1,
                                    hess_init=hp['hess'], beta2=hp['beta2'],
                                    weight_decay=hp['wd'], hess_approx='bonnet', sync=True,
                                    clip_radius=hp['clip_radius'],
                                    gamma_m=hp['gamma_m'], gamma_s=hp['gamma_s'])
        elif task_idx == args.start_task and args.start_task > 0:
            optimizer = CoVON_wprior(model.parameters(),
                                    lr=hp['lr'], ess=hp['ess'], mc_samples=1,
                                    hess_init=hp['hess'], beta2=hp['beta2'],
                                    weight_decay=hp['wd'], hess_approx='bonnet', sync=True,
                                    clip_radius=hp['clip_radius'],
                                    gamma_m=hp['gamma_m'], gamma_s=hp['gamma_s'])
            ckpt_opt = os.path.join(args.checkpoint_dir, f"optimizer_task_{args.start_task - 1}.pth")
            optimizer.load_state_dict(torch.load(ckpt_opt, map_location="cpu"))
            for state in optimizer.state.values():
                if not isinstance(state, dict):
                    continue
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            for group in optimizer.param_groups:
                for k, v in list(group.items()):
                    if torch.is_tensor(v):
                        group[k] = v.to(device)
                    elif isinstance(v, list):
                        group[k] = [x.to(device) if torch.is_tensor(x) else x for x in v]
            optimizer.set_for_new_task(ess=hp['ess'], hess_init=hp['hess2'], model=base_model)
        else:
            optimizer.gamma_m = hp['gamma_m']
            optimizer.gamma_s = hp['gamma_s']
            for group in optimizer.param_groups:
                group['lr'] = hp['lr']
            optimizer.set_for_new_task(ess=hp['ess'], hess_init=hp['hess2'], model=base_model)

        t0 = time.time()
        _, _ = train(train_loader, model, optimizer, hp['epochs'], task_idx)

        if is_dist_initialized():
            dist.barrier()

        avg_acc = test_all(data_test, task_idx, model)
        avg_acc_all.append(avg_acc)

        if is_main_process():
            print(f"Task {task_idx+1} done in {time.time()-t0:.1f}s  avg_test_acc={avg_acc:.2f}%")
            torch.save(base_model.state_dict(),
                       os.path.join(args.checkpoint_dir, f"model_task_{task_idx}.pth"))
            torch.save(optimizer.state_dict(),
                       os.path.join(args.checkpoint_dir, f"optimizer_task_{task_idx}.pth"))

        if is_dist_initialized():
            dist.barrier()

    if is_main_process():
        print(f"\nDone. Per-task avg_test_acc: {avg_acc_all}")
        print(f"HPARAM_SEARCH_METRIC: {float(np.mean(avg_acc_all)):.6f}")
        if wandb is not None:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",           type=float, default=5.798072701368654e-06)
    parser.add_argument("--lr_task0",     type=float, default=1e-4)
    parser.add_argument("--hess",         type=float, default=0.005)
    parser.add_argument("--hess2",        type=float, default=0.0015172863354075696)
    parser.add_argument("--beta2",        type=float, default=0.999)
    parser.add_argument("--wd",           type=float, default=1e-4)
    parser.add_argument("--ess",          type=float, default=16290309.780104343)
    parser.add_argument("--clip_radius",  type=float, default=0.02306187425100503)
    parser.add_argument("--gamma_m",      type=float, default=0.31201480297685247)
    parser.add_argument("--gamma_s",      type=float, default=0.03207585711845409)
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--num_tasks",    type=int,   default=8,
                        help="Number of train sessions to use (1-8)")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs_core50/")
    parser.add_argument("--start_task",   type=int,   default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    run_training(args)
