"""TLOB‑compatible builder for Sierra‑Chart DOM dumps (NQ)
================================================================
This builder ingests **one or many** CSV files produced by the
`DumpDomCSV` ACSIL study when the study is configured to output the DOM
columns in the following *block* order:

```
 ts_ms ,  ask_px0…9 , bid_px0…9 , ask_sz0…9 , bid_sz0…9
```

Internally the pipeline needs the canonical **inter‑leaved** layout used
throughout the original TLOB repo:

```
 ask_px0 , ask_sz0 , bid_px0 , bid_sz0 ,  ask_px1 , ask_sz1 , … , bid_sz9
          ^even idx = price                ^odd idx  = size
```

This class therefore **reorders** every DataFrame **as soon as it is
loaded**, before any labeling or normalization is performed, ensuring
that all downstream utilities (`labeling`, `z_score_orderbook`, sparse
encoders, etc.) operate on the expected schema.

The final artefacts are the same three `.npy` blobs expected by the
Leonardo‑Berti TLOB fork:

```
<data_dir>/NQ/train.npy   (40 features | ts | 4 label cols)
<data_dir>/NQ/val.npy
<data_dir>/NQ/test.npy
```

The timestamp column is preserved at index 0 of every row so model
predictions can be synchronised back to real‑time ticks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

import constants as cst
from constants import SamplingType

# ---------------------------------------------------------------------------
# helper: robust timestamp → pandas.Timestamp (UTC)
# ---------------------------------------------------------------------------

def _to_timestamp(col: pd.Series) -> pd.Series:
    """Handle **millisecond integers** *or* ISO‑8601 strings (+timezone)."""
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_datetime(col, unit="ms", utc=True, errors="coerce")
    # else assume string‑like
    return pd.to_datetime(col, utc=True, errors="coerce", format="ISO8601")

# ---------------------------------------------------------------------------
# quick loader (mirrors btc_load)
# ---------------------------------------------------------------------------

def nq_load(path: str | Path, len_smooth: int, h: int, seq_size: int):
    arr = np.load(path)
    col = {2: -4, 3: -3, 4: -2, 5: -1}[h]
    labels = arr[seq_size - len_smooth :, col]
    labels = torch.from_numpy(labels[np.isfinite(labels)]).long()
    inputs = torch.from_numpy(arr[:, 1 : 1 + 4 * cst.N_LOB_LEVELS]).float()
    timestamps = torch.from_numpy(arr[:, 0])  # keep if you need it
    return inputs, labels, timestamps

# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------

class NQDataBuilder:
    """Build and save train/val/test .npy files for NQ DOM dumps."""

    # input‑file columns (block order)
    _COLS = (
        ["timestamp"]
        + [f"ask_px{i}" for i in range(10)]
        + [f"bid_px{i}" for i in range(10)]
        + [f"ask_sz{i}" for i in range(10)]
        + [f"bid_sz{i}" for i in range(10)]
    )

    # target inter‑leaved feature order (len = 40)
    _IL_COLS = [
        f"ask_px{i}" for i in range(10)
        for _ in ()  # placeholder to allow comprehension trick
    ]  # will be rebuilt in __init__

    def __init__(
        self,
        data_dir: str | Path,
        csv_files: str | Path | Iterable[str | Path],
        split_rates=(0.8, 0.1, 0.1),  # by *trading‑day*
        sampling_type: SamplingType = SamplingType.TIME,
        sampling_time: str = "1S",
    ):
        # absolute, OS‑safe paths ------------------------------------------
        self.data_dir = Path(data_dir).expanduser().resolve()
        if isinstance(csv_files, (str, Path)):
            self.csv_files: List[Path] = [Path(csv_files).expanduser().resolve()]
        else:
            self.csv_files = [Path(p).expanduser().resolve() for p in csv_files]

        self.split_rates    = split_rates
        self.sampling_type  = sampling_type
        self.sampling_time  = sampling_time
        self.n_lob_levels   = cst.N_LOB_LEVELS

        # build canonical inter‑leaved column list once --------------------
        self._IL_COLS = []
        for lvl in range(self.n_lob_levels):
            self._IL_COLS += [f"ask_px{lvl}", f"ask_sz{lvl}",
                              f"bid_px{lvl}", f"bid_sz{lvl}"]

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def prepare_save_datasets(self):
        day_dir = self._split_by_day()        # 1 CSV per date
        self._load_concat_splits(day_dir)     # three big DFs
        self._append_labels()                 # four horizon cols
        self._normalize_dataframes()          # z‑score 40 features

        out_dir = self.data_dir / "NQ"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "train.npy", self.train_set)
        np.save(out_dir / "val.npy",   self.val_set)
        np.save(out_dir / "test.npy",  self.test_set)
        print("Saved .npy files in", out_dir)

    # ------------------------------------------------------------------
    # INTERNAL: UTIL ----------------------------------------------------
    # ------------------------------------------------------------------
    def _apply_interleaved(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *view* of df with timestamp + inter‑leaved 40 features."""
        return df[["timestamp"] + self._IL_COLS]

    # ------------------------------------------------------------------
    # step 1: split raw CSV(s) into one file per trading day -------------
    # ------------------------------------------------------------------
    def _split_by_day(self) -> Path:
        day_dir = self.data_dir / "NQ" / "day_csv"
        day_dir.mkdir(parents=True, exist_ok=True)
        if any(day_dir.iterdir()):
            print("Daily CSVs already exist – skipping re‑split.")
            return day_dir

        for csv_path in self.csv_files:
            print("Reading", csv_path)
            df = pd.read_csv(csv_path, header=None)
            df.columns = self._COLS
            df["timestamp"] = _to_timestamp(df["timestamp"])
            df = df.dropna(subset=["timestamp"])

            for date, day_df in df.groupby(df["timestamp"].dt.date):
                fname = f"NQ_{date}.csv"
                day_df.to_csv(day_dir / fname, index=False, header=False)
                print("  wrote", fname, f"({len(day_df)} rows)")
        return day_dir

    # ------------------------------------------------------------------
    # step 2: concat day files into train / val / test DFs ---------------
    # ------------------------------------------------------------------
    def _load_concat_splits(self, day_dir: Path):
        files = sorted(day_dir.glob("NQ_*.csv"))
        n_days = len(files)
        n_train = int(n_days * self.split_rates[0])
        n_val   = int(n_days * self.split_rates[1]) + n_train

        def _read_one(path: Path):
            df = pd.read_csv(path, names=self._COLS)
            # cast to int64 (ms) right here → guarantees numeric dtype downstream
            df["timestamp"] = (
            _to_timestamp(df["timestamp"]).dropna()
                                          .astype("int64") // 1_000_000            # ns → ms
            )

            # OPTIONAL: resampling — commented out for 1 second native DOM
            # if self.sampling_type == SamplingType.TIME:
            #     df = (
            #         df.set_index("timestamp")
            #           .resample(self.sampling_time)
            #           .last()
            #           .ffill()
            #           .dropna()
            #           .reset_index()
            #     )

            # ---- reorder to canonical inter‑leaved layout --------------
            return self._apply_interleaved(df)

        train_df = pd.concat([_read_one(f) for f in files[:n_train]], ignore_index=True)
        val_df   = pd.concat([_read_one(f) for f in files[n_train:n_val]], ignore_index=True)
        test_df  = pd.concat([_read_one(f) for f in files[n_val:]], ignore_index=True)

        # keep timestamp column for later
        self.dataframes = [train_df, val_df, test_df]

    # ------------------------------------------------------------------
    # step 3: generate four horizon label columns and concat ------------
    # ------------------------------------------------------------------
    def _append_labels(self):
        train_vals = self.dataframes[0].values.astype("int64")
        val_vals   = self.dataframes[1].values.astype("int64")
        test_vals  = self.dataframes[2].values.astype("int64")

        for h in cst.LOBSTER_HORIZONS:  # [10,20,50,100]
            for name, arr_in in zip(("train", "val", "test"), (train_vals, val_vals, test_vals)):
                labs = NQDataBuilder.labeling_tick(arr_in[:, 1:], cst.LEN_SMOOTH, h, 0.25,4)  # skip ts for features
                pad  = np.full(arr_in.shape[0] - labs.shape[0], np.inf)
                labs = np.concatenate([labs, pad])
                target = getattr(self, f"{name}_labels_horizons", None)
                col = f"label_h{h}"
                if target is None:
                    setattr(self, f"{name}_labels_horizons", pd.DataFrame(labs, columns=[col]))
                else:
                    target[col] = labs

        self.train_set = np.concatenate([train_vals, self.train_labels_horizons.values], axis=1)
        self.val_set   = np.concatenate([val_vals,   self.val_labels_horizons.values],   axis=1)
        self.test_set  = np.concatenate([test_vals,  self.test_labels_horizons.values],  axis=1)

    # ------------------------------------------------------------------
    # step 4: z‑score the 40 numeric features (timestamp untouched) ------
    # ------------------------------------------------------------------
    def _normalize_dataframes(self):
        for i, df in enumerate(self.dataframes):
            ts = df["timestamp"].astype("int64")
            feats = df.drop(columns=["timestamp"])
            # feats are already inter‑leaved from _load_concat_splits
            if i == 0:
                feats, m_sz, m_px, s_sz, s_px = NQDataBuilder.z_score_orderbook(feats)
            else:
                feats, *_ = NQDataBuilder.z_score_orderbook(feats, m_sz, m_px, s_sz, s_px)
            self.dataframes[i] = pd.concat([ts, feats], axis=1)

    @staticmethod
    def labeling_tick(X, len_smooth, h, tick=0.25, alpha_ticks=4):
        """
        Three-class label generator (up / stay / down) based on a fixed
        tick threshold.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
            Inter-leaved LOB features; column 0 = ask_px0, column 2 = bid_px0.
        len_smooth : int
            Rolling window length used to smooth mid-prices.
        h : int
            Prediction horizon (in rows).
        tick : float, default 0.25
            Tick size of the instrument (NQ = 0.25 points).
        alpha_ticks : int, default 4
            Number of ticks beyond which the move counts as up / down.

        Returns
        -------
        labels : ndarray (n_samples-h,), dtype=int
            0 = up, 1 = stay, 2 = down
        """
        # ------------------------------------------------------------------
        # Align smoothing window and horizon
        # ------------------------------------------------------------------
        if h < len_smooth:
            len_smooth = h

        # ------------------------------------------------------------------
        # Rolling mid-prices (ask0 + bid0) / 2
        # ------------------------------------------------------------------
        prev_ask = np.lib.stride_tricks.sliding_window_view(
            X[:, 0], window_shape=len_smooth
        )[:-h]
        prev_bid = np.lib.stride_tricks.sliding_window_view(
            X[:, 2], window_shape=len_smooth
        )[:-h]
        fut_ask  = np.lib.stride_tricks.sliding_window_view(
            X[:, 0], window_shape=len_smooth
        )[h:]
        fut_bid  = np.lib.stride_tricks.sliding_window_view(
            X[:, 2], window_shape=len_smooth
        )[h:]

        prev_mid = (prev_ask + prev_bid) / 2.0
        fut_mid  = (fut_ask  + fut_bid)  / 2.0
        prev_mid = prev_mid.mean(axis=1)
        fut_mid  = fut_mid.mean(axis=1)

        # ------------------------------------------------------------------
        # Threshold in absolute points (alpha = ticks × tick_size)
        # ------------------------------------------------------------------
        delta = fut_mid - prev_mid
        alpha = alpha_ticks * tick

        labels = np.where(delta < -alpha, 2,
                np.where(delta >  alpha, 0, 1))

        # diagnostics: ensure every class percentage is printed
        counts = np.bincount(labels, minlength=3)
        pct    = counts / counts.sum()
        print(f"tick size           : {tick}")
        print(f"alpha (ticks)       : {alpha_ticks}")
        print(f"alpha (points)      : {alpha}")
        print(f"Label distribution  : {{0: {counts[0]}, 1: {counts[1]}, 2: {counts[2]}}}")
        print(f"Label percentages   : {{0: {pct[0]:.4f}, 1: {pct[1]:.4f}, 2: {pct[2]:.4f}}}")
        return labels

        return labels
    
    @staticmethod
    def z_score_orderbook(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
        """ DONE: remember to use the mean/std of the training set, to z-normalize the test set. """
        if (mean_size is None) or (std_size is None):
            mean_size = data.iloc[:, 1::2].stack().mean()
            std_size = data.iloc[:, 1::2].stack().std()

        #do the same thing for prices
        if (mean_prices is None) or (std_prices is None):
            mean_prices = data.iloc[:, 0::2].stack().mean() #price
            std_prices = data.iloc[:, 0::2].stack().std() #price

        # apply the z score to the original data using .loc with explicit float cast
        price_cols = data.columns[0::2]
        size_cols = data.columns[1::2]

        #apply the z score to the original data
        for col in size_cols:
            data[col] = data[col].astype("float64")
            data[col] = (data[col] - mean_size) / std_size

        for col in price_cols:
            data[col] = data[col].astype("float64")
            data[col] = (data[col] - mean_prices) / std_prices

        # check if there are null values, then raise value error
        if data.isnull().values.any():
            raise ValueError("data contains null value")

        return data, mean_size, mean_prices, std_size,  std_prices

