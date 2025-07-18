"""
preprocessing/dataset.py
────────────────────────
Generic Dataset/DataModule helpers + a memory-efficient loader
for the custom MY_NQ token/label files.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset as TorchDataset, DataLoader, WeightedRandomSampler

import constants as cst
from utils.utils_data import one_hot_encoding_type, tanh_encoding_type   # noqa: F401


# ──────────────────────────────────────────────────────────
# 1. Classic in-RAM Dataset
# ──────────────────────────────────────────────────────────
class Dataset(TorchDataset):
    """Holds an (N, F) tensor and returns fixed-length windows stored in RAM."""
    def __init__(self, x, y, seq_size: int):
        self.seq_size = seq_size
        self.length   = y.shape[0]
        self.x = torch.as_tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x
        self.y = torch.as_tensor(y, dtype=torch.long)    if isinstance(y, np.ndarray) else y

    def __len__(self):              return self.length
    def __getitem__(self, idx):
        return ( self.x[idx:idx+self.seq_size],
                 self.y[idx] )


# ──────────────────────────────────────────────────────────
# 2. Streaming Dataset for big files
# ──────────────────────────────────────────────────────────
class WindowDataset(TorchDataset):
    """Overlapping windows from a memory-mapped (N, 41) token array."""
    def __init__(self, X_mmap, y_mmap,
                 start, end, seq_len: int = 128, step: int = 1):
        self.X, self.y = X_mmap, y_mmap          # mmaps
        self.start, self.end = start, end
        self.seq, self.step  = seq_len, step
        self.length = (end - start - seq_len) // step + 1

    def __len__(self):              return self.length
    def __getitem__(self, idx):
        i = self.start + idx * self.step
        window = self.X[i:i+self.seq, 1:]
        label  = self.y[i + self.seq - 1]
        return ( torch.as_tensor(window, dtype=torch.float32),
                 torch.tensor(label, dtype=torch.long) )


# ──────────────────────────────────────────────────────────
# 3. Lightning DataModule with balanced sampler
# ──────────────────────────────────────────────────────────
class DataModule(pl.LightningDataModule):
    """
    • Training loader uses a WeightedRandomSampler so every mini-batch is
      close to class-balanced (useful when positive class ≈ 20 %).
    • Val / Test loaders unchanged.
    """
    def __init__(self,
                 train_set: TorchDataset,
                 val_set:   TorchDataset,
                 batch_size: int,
                 test_set: TorchDataset = None,
                 test_batch_size: int = None,
                 is_shuffle_train: bool = True,
                 num_workers: int = 8):
        super().__init__()
        self.train_set = train_set
        self.val_set   = val_set
        self.test_set  = test_set
        self.batch_size       = batch_size
        self.test_batch_size  = test_batch_size or batch_size
        self.is_shuffle_train = is_shuffle_train
        self.pin_memory       = cst.DEVICE == "cuda"
        self.num_workers      = num_workers

    # ───────── balanced train loader ─────────
    def train_dataloader(self):
        # 1) grab labels as a numpy array
        if hasattr(self.train_set, 'y'):
            labels = np.asarray(self.train_set.y)
        else:
            raise AttributeError("train_set has no attribute 'y'")

        # 2) compute per-class weights  (inverse frequency)
        class_counts  = np.bincount(labels, minlength=int(labels.max())+1)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights      = torch.as_tensor(sample_weights, dtype=torch.float),
            num_samples  = len(sample_weights),
            replacement  = True
        )

        return DataLoader(self.train_set,
                          batch_size   = self.batch_size,
                          sampler      = sampler,          # ← replaces shuffle
                          pin_memory   = self.pin_memory,
                          num_workers  = self.num_workers,
                          drop_last    = False)

    # ───────── val / test loaders ─────────
    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size   = self.batch_size,
                          shuffle      = False,
                          pin_memory   = self.pin_memory,
                          num_workers  = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size   = self.test_batch_size,
                          shuffle      = False,
                          pin_memory   = self.pin_memory,
                          num_workers  = self.num_workers)


# ──────────────────────────────────────────────────────────
# 4. Helper for MY_NQ token files
# ──────────────────────────────────────────────────────────
def load_pre_split_dataset(tokens_path: str,
                           label_path:  str,
                           seq_len: int = 128,
                           step: int = 1,
                           horizon: int = 30,              # kept for API
                           split_rates = (0.7, 0.15, 0.15)):
    """Return (train_ds, val_ds, test_ds) as WindowDataset objects."""
    X = np.load(tokens_path, mmap_mode="r")
    y = np.load(label_path,  mmap_mode="r")
    assert len(X) == len(y), "tokens and labels length mismatch"

    N       = len(X)
    n_train = int(split_rates[0] * N)
    n_val   = int(split_rates[1] * N)
    bounds  = [0, n_train, n_train + n_val, N]

    train_ds = WindowDataset(X, y, bounds[0], bounds[1], seq_len, step)
    val_ds   = WindowDataset(X, y, bounds[1], bounds[2], seq_len, step)
    test_ds  = WindowDataset(X, y, bounds[2], bounds[3], seq_len, step)

    return train_ds, val_ds, test_ds
