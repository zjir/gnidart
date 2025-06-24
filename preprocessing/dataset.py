"""
preprocessing/dataset.py
────────────────────────
Generic Dataset/DataModule helpers + a memory-efficient loader
for the custom MY_NQ token/label files.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset as TorchDataset, DataLoader

import constants as cst
# utils.utils_data is imported only to keep previous API intact
from utils.utils_data import one_hot_encoding_type, tanh_encoding_type   # noqa: F401


# ──────────────────────────────────────────────────────────
# 1. Classic in-RAM Dataset (unchanged)
# ──────────────────────────────────────────────────────────
class Dataset(TorchDataset):
    """
    Holds an (N, F) tensor and returns fixed-length windows stored in RAM.
    """
    def __init__(self, x, y, seq_size: int):
        self.seq_size = seq_size
        self.length   = y.shape[0]
        self.x = torch.as_tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x
        self.y = torch.as_tensor(y, dtype=torch.long)    if isinstance(y, np.ndarray) else y

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            self.x[idx : idx + self.seq_size],     # (seq, features)
            self.y[idx],
        )


# ──────────────────────────────────────────────────────────
# 2. Streaming Dataset for big files
# ──────────────────────────────────────────────────────────
class WindowDataset(TorchDataset):
    """
    Provides overlapping windows from a memory-mapped (N, 41) token array
    without creating a giant interim tensor.
    """
    def __init__(
        self,
        X_mmap: np.ndarray,
        y_mmap: np.ndarray,
        start: int,
        end: int,
        seq_len: int = 128,
        step: int = 1,
    ):
        self.X      = X_mmap            # int16, mmap'd
        self.y      = y_mmap            # uint8, mmap'd
        self.start  = start
        self.end    = end
        self.seq    = seq_len
        self.step   = step
        self.length = (end - start - seq_len) // step + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = self.start + idx * self.step
        window = self.X[i : i + self.seq, 1:]          # drop ts
        label  = self.y[i + self.seq - 1]

        # convert just this slice to tensors  ↓  use float32 for model input
        return torch.as_tensor(window, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.long)       # ← keep targets long

# ──────────────────────────────────────────────────────────
# 3. Lightning DataModule (handles either dataset)
# ──────────────────────────────────────────────────────────
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_set: TorchDataset,
        val_set:   TorchDataset,
        batch_size: int,
        test_set: TorchDataset = None,
        test_batch_size: int = None,
        is_shuffle_train: bool = True,
        num_workers: int = 8,
    ):
        super().__init__()
        self.train_set = train_set
        self.val_set   = val_set
        self.test_set  = test_set
        self.batch_size       = batch_size
        self.test_batch_size  = test_batch_size or batch_size
        self.is_shuffle_train = is_shuffle_train
        self.pin_memory       = cst.DEVICE == "cuda"
        self.num_workers      = num_workers

    # dataloaders -------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, self.is_shuffle_train,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, False,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.test_batch_size, False,
                          pin_memory=self.pin_memory, num_workers=self.num_workers)


# ──────────────────────────────────────────────────────────
# 4. Helper for MY_NQ tokens/labels
# ──────────────────────────────────────────────────────────
def load_pre_split_dataset(
    tokens_path: str,
    label_path:  str,
    seq_len: int = 128,
    step: int = 1,
    horizon: int = 30,                # kept for signature compatibility
    split_rates = (0.7, 0.15, 0.15),
):
    """
    Reads bucketed_lob.npy + labels.npy and returns
    (train_ds, val_ds, test_ds) as WindowDataset instances.
    """
    X = np.load(tokens_path, mmap_mode="r")   # (N, 41) int16
    y = np.load(label_path,  mmap_mode="r")   # (N,)    uint8
    assert len(X) == len(y), "tokens and labels length mismatch"

    N = len(X)
    n_train = int(split_rates[0] * N)
    n_val   = int(split_rates[1] * N)
    bounds  = [0, n_train, n_train + n_val, N]

    train_ds = WindowDataset(X, y, bounds[0], bounds[1], seq_len, step)
    val_ds   = WindowDataset(X, y, bounds[1], bounds[2], seq_len, step)
    test_ds  = WindowDataset(X, y, bounds[2], bounds[3], seq_len, step)

    return train_ds, val_ds, test_ds
