"""
data.py -- Permuted MNIST data pipeline for continual learning.

10 tasks:
  Task 0 : plain MNIST (no permutation)
  Task 1-9: MNIST with a fixed random pixel permutation (different per task)

Each DataLoader yields (image, label, index) to match the training loop.
Tasks are cached to disk on first run so subsequent loads are instant.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class TensorMNIST(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], idx


def _make_permutation(seed: int) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randperm(784, generator=rng)


def _build_tensors(mnist_root, train, permutation):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(root=mnist_root, train=train, transform=transform, download=True)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=2)
    all_images, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.view(len(imgs), -1)
        if permutation is not None:
            imgs = imgs[:, permutation]
        all_images.append(imgs.view(len(imgs), 1, 28, 28))
        all_labels.append(labels)
    return torch.cat(all_images), torch.cat(all_labels)


def _save_tasks(cache_dir, mnist_root, num_tasks, perm_seed_offset):
    os.makedirs(cache_dir, exist_ok=True)
    permutations = []
    for task_idx in range(num_tasks):
        perm = None if task_idx == 0 else _make_permutation(perm_seed_offset + task_idx)
        permutations.append(perm)
        for split, train_flag in [("train", True), ("test", False)]:
            path = os.path.join(cache_dir, f"task_{task_idx}_{split}.pt")
            images, labels = _build_tensors(mnist_root, train_flag, perm)
            torch.save({"images": images, "labels": labels}, path)
        print(f"  saved task {task_idx} ({'plain' if perm is None else f'seed {perm_seed_offset + task_idx}'})")
    torch.save(permutations, os.path.join(cache_dir, "permutations.pt"))
    return permutations


def get_permuted_mnist_loaders(
    cache_dir="./data/perm_mnist_cache",
    mnist_root="./data",
    num_tasks=10,
    batch_size=256,
    test_batch_size=512,
    num_workers=2,
    perm_seed_offset=1000,
):
    perm_file = os.path.join(cache_dir, "permutations.pt")
    all_cached = all(
        os.path.exists(os.path.join(cache_dir, f"task_{t}_{s}.pt"))
        for t in range(num_tasks) for s in ("train", "test")
    ) and os.path.exists(perm_file)

    if all_cached:
        print(f"Loading tasks from cache: {cache_dir}")
        permutations = torch.load(perm_file, weights_only=False)
    else:
        print(f"Cache not found -- building {num_tasks} tasks and saving to {cache_dir}")
        permutations = _save_tasks(cache_dir, mnist_root, num_tasks, perm_seed_offset)

    train_loaders, test_loaders = [], []
    for task_idx in range(num_tasks):
        for split, loaders, shuffle, bs in [
            ("train", train_loaders, True,  batch_size),
            ("test",  test_loaders,  False, test_batch_size),
        ]:
            data = torch.load(
                os.path.join(cache_dir, f"task_{task_idx}_{split}.pt"),
                weights_only=True,
            )
            ds = TensorMNIST(data["images"], data["labels"])
            loaders.append(DataLoader(
                ds, batch_size=bs, shuffle=shuffle,
                num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            ))

    return train_loaders, test_loaders, permutations
