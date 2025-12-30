"""
bci_code/train_cv.py
Braindecode EEG Conformer Training Script for BCI Competition 2020 Track 3 (Language Decoding)
Compatible with Docker environment 'bci'
"""
import argparse
import sys
import logging
import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset


class _IpywidgetsStub:
    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):  # pragma: no cover - defensive shim
        return self


sys.modules.setdefault("ipywidgets", _IpywidgetsStub())

from braindecode.models import EEGConformer
from braindecode.classifier import EEGClassifier
from skorch.callbacks import LRScheduler, EarlyStopping
from skorch.helper import predefined_split

# Logging Setup
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="EEG Conformer Training")
    # Allowed overrides as per README.md
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--subject_ids", type=int, nargs='+', default=[1])
    parser.add_argument("--cache_root", type=str, default="/mnt/bci_source/data/preprocessed_mixed_256hz_v2")
    parser.add_argument("--aug_prob", type=float, default=0.5, help="Augmentation probability")
    parser.add_argument("--config", type=str, help="Path to JSON config produced by tools.eeg_experiment")
    parser.add_argument("--data_root", type=str, default="/mnt/bci_source/data/preprocessed_mixed_256hz_v2", help="Root directory for language decoding data")
    parser.add_argument("--classes", type=int, nargs='+', help="Optional class ids for language decoding labels")

    # Internal flags
    parser.add_argument("--dry_run", action="store_true", help="Run in fast check mode")
    parser.add_argument("--output_dir", type=str, default="./runs/debug", help="Directory to save artifacts")

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "config", None):
        return args
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                log.debug(f"Ignoring unknown config key: {k}")
        log.info(f"Loaded config overrides from {args.config}")
    except Exception as e:
        log.error(f"Failed to load config {args.config}: {e}")
        sys.exit(1)
    return args


def _select_npz(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Pick the first NPZ with X/y keys if available."""
    if not data_root.exists():
        log.warning(f"Data root does not exist: {data_root}")
        return None, None  # type: ignore[return-value]
    for path in sorted(data_root.rglob("*.npz")):
        try:
            loaded = np.load(path)
            if "X" in loaded and "y" in loaded:
                X = loaded["X"]
                y = loaded["y"]
                log.info(f"Loaded NPZ dataset: {path}")
                return X, y
        except Exception as exc:  # pragma: no cover (best-effort)
            log.warning(f"Failed to load {path}: {exc}")
            continue
    return None, None  # type: ignore[return-value]


def load_or_synthesize_data(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load language decoding data if present; otherwise synthesize."""
    data_root = Path(args.data_root)

    if not args.dry_run:
        loaded_subjects = []
        for sid in args.subject_ids:
            fname = f"S{sid:02d}_preprocessed_with_test.pkl"
            path = data_root / fname
            if not path.exists():
                log.warning(f"Subject file missing: {path}")
                continue
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                X = np.concatenate([payload["X_train"], payload["X_val"]], axis=0)
                y = np.concatenate([payload["y_train"], payload["y_val"]], axis=0)
                sfreq = float(payload.get("sfreq", 250.0))
                loaded_subjects.append((X.astype(np.float32), y.astype(np.int64), sfreq))
                log.info(f"Loaded subject {sid:02d} from {path}")
            except Exception as exc:
                log.error(f"Failed to load {path}: {exc}")
                continue

        if loaded_subjects:
            X_all = np.concatenate([item[0] for item in loaded_subjects], axis=0)
            y_all = np.concatenate([item[1] for item in loaded_subjects], axis=0)
            sfreq = loaded_subjects[0][2]
            log.info(f"Using mounted dataset at {data_root}; subjects={len(loaded_subjects)}; trials={X_all.shape[0]}")
            return X_all, y_all, sfreq

        X_npz, y_npz = _select_npz(data_root)
        if X_npz is not None and y_npz is not None:
            X_npz = np.asarray(X_npz, dtype=np.float32)
            y_npz = np.asarray(y_npz, dtype=np.int64)
            log.info(f"Using NPZ dataset at {data_root}")
            return X_npz, y_npz, 250.0

    rng = np.random.default_rng(args.seed)
    synth_trials = 12 if args.dry_run else 64
    synth_channels = 64
    synth_times = 1000
    synth_classes = args.classes if args.classes else list(range(8))
    n_classes = len(synth_classes)

    log.warning("WARNING: Using Synthetic Data for Language Decoding Test")
    X = rng.standard_normal((synth_trials, synth_channels, synth_times), dtype=np.float32)
    y = rng.integers(0, n_classes, size=synth_trials, endpoint=False, dtype=np.int64)
    return X, y, 250.0


def make_splits(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2) -> Tuple[TensorDataset, TensorDataset]:
    """Simple hold-out split with deterministic shuffling."""
    n_trials = X.shape[0]
    indices = np.arange(n_trials)
    np.random.shuffle(indices)
    split = int(n_trials * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    return train_ds, val_ds


def normalize_labels(y: np.ndarray) -> Tuple[np.ndarray, list]:
    classes = sorted(np.unique(y).tolist())
    mapping = {c: i for i, c in enumerate(classes)}
    y_norm = np.vectorize(mapping.get)(y).astype(np.int64)
    return y_norm, classes


def main():
    args = load_config(parse_args())

    # Ensure reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log.info(f"Starting Experiment with config: {vars(args)}")
    log.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. Dataset Loading (Language Decoding) ---
    X, y_raw, sfreq = load_or_synthesize_data(args)
    y, classes = normalize_labels(y_raw)
    n_trials, n_channels, n_times = X.shape
    target_times = 1000 if n_times < 1000 else n_times
    if target_times != n_times:
        pad = target_times - n_times
        X = np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="constant")
        n_times = target_times
        log.info(f"Padded time dimension to {n_times} samples for model compatibility")
    n_classes = len(classes)
    log.info(f"Data prepared: trials={n_trials}, channels={n_channels}, time={n_times}, classes={n_classes}")

    # --- 2. Model Definition (EEG Conformer) ---
    log.info(f"Building EEG Conformer (Ch={n_channels}, Classes={n_classes}, Sfreq={sfreq})")
    model = EEGConformer(
        n_outputs=n_classes,
        n_chans=n_channels,
        n_times=n_times,
        input_window_seconds=float(n_times) / sfreq,
        sfreq=sfreq,
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # --- 3. Training Loop ---
    train_ds, valid_ds = make_splits(X, y)

    t_max = max(1, args.epochs - 1)

    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_ds),
        optimizer__lr=args.lr,
        optimizer__weight_decay=0.01,
        batch_size=args.batch_size,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=t_max)),
            ("early_stopping", EarlyStopping(monitor='valid_loss', patience=args.patience)),
        ],
        device=device,
        classes=list(range(n_classes)),
    )

    log.info("Starting Training...")
    try:
        clf.fit(train_ds, y=None, epochs=args.epochs)
    except Exception as e:
        log.error(f"Training failed: {e}")
        if not args.dry_run:
            raise e

    # --- 4. Evaluation & Reporting ---
    train_acc = clf.history[-1, 'train_accuracy']
    valid_acc = clf.history[-1, 'valid_accuracy']

    log.info("Training Completed.")
    log.info(f"Final Train Acc: {train_acc:.4f}")
    log.info(f"Final Valid Acc: {valid_acc:.4f}")

    # IMPORTANT: The Agent parses this specific string format
    print(f"overall_mean_acc: {valid_acc:.4f}")


if __name__ == "__main__":
    main()