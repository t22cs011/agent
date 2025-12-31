import sys
import argparse
import os
import pickle
import warnings
import faulthandler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from braindecode.models import EEGConformer, ShallowFBCSPNet, Deep4Net, EEGNetv4
import pandas as pd
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

warnings.filterwarnings("ignore", message=".*Tensorflow.*")

def _resolve_data_root() -> str:
    env_root = os.environ.get("BCI_DATA_ROOT")
    candidates = []
    if env_root:
        candidates.append(env_root)
    candidates.extend([
        "/mnt/bci_source/data/preprocessed_mixed_256hz_v2",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed_mixed_256hz_v2")),
        os.path.abspath(os.path.join(os.getcwd(), "data", "preprocessed_mixed_256hz_v2")),
    ])
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            print(f"[info] Using dataset root: {candidate}")
            return candidate
    fallback = candidates[0] if candidates else "/mnt/bci_source/data/preprocessed_mixed_256hz_v2"
    print(f"[warn] No dataset root exists; falling back to {fallback}")
    return fallback


def _parse_args():
    parser = argparse.ArgumentParser(description="Train CV runner with lightweight debug options")
    parser.add_argument("--subjects", "-s", help="Comma-separated subject IDs or ranges (e.g. 1,3-5)", default=None)
    parser.add_argument("--n-folds", type=int, help="Number of folds to use (overrides N_FOLDS)", default=None)
    parser.add_argument("--max-epochs", type=int, help="Max epochs (overrides MAX_EPOCHS)", default=None)
    parser.add_argument("--results-root", help="Override RESULTS_ROOT directory", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run a very small quick check (1 subject, few epochs)")
    return parser.parse_args()

# --- Configuration ---
DATA_ROOT = _resolve_data_root()
# Use a workspace-local results folder by default (avoid creating '/runs')
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "runs")

# Ensure base results directory exists early (used by faulthandler)
os.makedirs(RESULTS_ROOT, exist_ok=True)
SUBJECT_IDS = range(1, 16)  # S01 to S15
N_FOLDS = 10
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 16
LR = 0.000625  # 一般的なBraindecodeのデフォルト
WEIGHT_DECAY = 0
# When True, do not load existing checkpoints (useful for dry-run / quick checks)
SKIP_CHECKPOINTS = False

# Data Shapes
N_TIME_POINTS = 512  # 1 trial = 512 samples (2s @256Hz)
# Cropped decoding: 464-sample window with stride 6 -> 9 crops per trial
CROP_WINDOW = 464
CROP_STRIDE = 6
TEST_STRIDE = 4
N_CROPS_TEST = 1

def _build_shallow(n_outputs, n_chans, n_times):
    """ShallowFBCSP tuned for 464-sample crops; keep kernels <= input."""
    filter_len = 20  # shorter temporal kernel to stay < window
    conv_out = max(1, n_times - filter_len + 1)
    pool_len = max(4, min(64, conv_out))
    pool_stride = max(2, min(8, pool_len))
    return ShallowFBCSPNet(
        n_outputs=n_outputs,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=256,
        n_filters_time=20,
        n_filters_spat=20,
        filter_time_length=filter_len,
        pool_time_length=pool_len,
        pool_time_stride=pool_stride,
        drop_prob=0.5,
    )


def _build_eegconformer(n_outputs, n_chans, n_times):
    # Smaller Conformer with explicit pooling to stabilize dense layers
    return EEGConformer(
        n_outputs=n_outputs,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=256,
        pool_time_length=8,
        pool_time_stride=4,
        n_filters_time=16,
        att_depth=2,
        att_heads=4,
        att_drop_prob=0.5,
        drop_prob=0.5,
        final_fc_length="auto",
    )


MODELS = {
    "EEGConformer": _build_eegconformer,
    "ShallowFBCSPNet": _build_shallow,
    "DeepNet": lambda n_outputs, n_chans, n_times: Deep4Net(
        n_outputs=n_outputs,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=256,
        drop_prob=0.5,
    ),
    "EEGNet": lambda n_outputs, n_chans, n_times: EEGNetv4(
        n_outputs=n_outputs,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=256,
        F1=8,
        D=2,
        F2=16,
        drop_prob=0.25,
    ),
}

# Device selection: allow forcing CPU via env `BCI_FORCE_CPU` or fallback to CPU when CUDA unavailable.
force_cpu_env = os.environ.get("BCI_FORCE_CPU", "0").lower()
FORCE_CPU = force_cpu_env in ("1", "true", "yes")
if FORCE_CPU:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("[warn] Using CPU device (BCI_FORCE_CPU=%s)" % (force_cpu_env,))
else:
    print(f"[info] Using device: {DEVICE}")

# Limit thread count to reduce instability on noisy hardware
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

# Enable faulthandler to capture native tracebacks on crash
try:
    faulthandler.enable()
    _fh_log = open(os.path.join(RESULTS_ROOT, "faulthandler.log"), "a", encoding="utf-8")
    faulthandler.enable(file=_fh_log)
except Exception:
    pass


def ensure_results_dir(preferred_dir=RESULTS_ROOT, fallback_dir="/tmp/results"):
    """Choose a writable directory for results, preferring configured RESULTS_ROOT."""
    for candidate in (preferred_dir, fallback_dir):
        try:
            os.makedirs(candidate, exist_ok=True)
            test_path = os.path.join(candidate, ".write_test")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_path)
            print(f"[info] Using results directory: {candidate}")
            return candidate
        except Exception as e:
            print(f"[warn] Cannot use {candidate}: {e}")
    print("[warn] Falling back to current directory for results")
    return "."


def load_subject_data(subject_id):
    """Load subject data with flexible key names.

    Returns:
        X_trainval, y_trainval: concatenated train+val
        X_test, y_test: optional test split (may be None if missing)
    """
    file_path = os.path.join(DATA_ROOT, f"S{subject_id:02d}_preprocessed_with_test.pkl")
    print(f"[debug][S{subject_id:02d}] Looking for file: {file_path}")
    if not os.path.exists(file_path):
        print(f"[warn][S{subject_id:02d}] File not found: {file_path}")
        return None, None

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[error][S{subject_id:02d}] Failed to load pickle: {e}")
        return None, None

    print(f"[debug][S{subject_id:02d}] Loaded object type: {type(data)}")

    if not isinstance(data, dict):
        print(f"[warn][S{subject_id:02d}] Unexpected data structure; expected dict but got {type(data)}")
        return None, None

    keys = list(data.keys())
    print(f"[debug][S{subject_id:02d}] Dict keys: {keys}")

    # Flexible key candidates for each split
    split_key_options = [
        ("train", ["X_train", "x_train", "train_x"], ["y_train", "Y_train", "train_y", "y"]),
        ("val", ["X_val", "x_val", "val_x", "X_valid", "x_valid", "valid_x"], ["y_val", "Y_val", "val_y", "y_valid", "valid_y"]),
        ("test", ["X_test", "x_test", "test_x"], ["y_test", "Y_test", "test_y"]),
    ]

    def pick_first(options):
        for k in options:
            if k in data:
                return k, data[k]
        return None, None

    xs_trainval = []
    ys_trainval = []
    xs_test = []
    ys_test = []

    for split_name, x_keys, y_keys in split_key_options:
        x_key, x_arr = pick_first(x_keys)
        y_key, y_arr = pick_first(y_keys)

        if x_key and y_key:
            x_shape = getattr(x_arr, "shape", "n/a")
            y_shape = getattr(y_arr, "shape", "n/a")
            print(f"    [debug][S{subject_id:02d}] {split_name}: {x_key} shape={x_shape}, {y_key} shape={y_shape}")
            if split_name == "test":
                xs_test.append(x_arr)
                ys_test.append(y_arr)
            else:
                xs_trainval.append(x_arr)
                ys_trainval.append(y_arr)
        elif x_key and not y_key:
            print(f"    [warn][S{subject_id:02d}] Found {x_key} but missing label key among {y_keys}")
        elif y_key and not x_key:
            print(f"    [warn][S{subject_id:02d}] Found {y_key} but missing feature key among {x_keys}")
        else:
            print(f"    [debug][S{subject_id:02d}] No data found for split '{split_name}'")

    if not xs_trainval or not ys_trainval:
        print(f"[warn][S{subject_id:02d}] No feature/label arrays collected; skipping subject")
        return None, None, None, None

    try:
        X = np.concatenate(xs_trainval, axis=0)
        y = np.concatenate(ys_trainval, axis=0)
    except Exception as e:
        print(f"[error][S{subject_id:02d}] Failed to concatenate arrays: {e}")
        return None, None, None, None

    if X.ndim != 3:
        print(f"[warn][S{subject_id:02d}] X has unexpected ndim={X.ndim}; expected 3")
        return None, None, None, None
    if y.ndim != 1:
        print(f"[warn][S{subject_id:02d}] y has unexpected ndim={y.ndim}; expected 1")
        return None, None, None, None
    if X.shape[0] != y.shape[0]:
        print(f"[warn][S{subject_id:02d}] Mismatch between samples and labels: X={X.shape}, y={y.shape}")
        return None, None, None, None

    print(f"[debug][S{subject_id:02d}] Combined X shape: {X.shape}, y shape: {y.shape}")

    if X.shape[2] < N_TIME_POINTS:
        pad_len = N_TIME_POINTS - X.shape[2]
        print(f"[debug][S{subject_id:02d}] Padding time axis by {pad_len}")
        X = np.pad(X, ((0, 0), (0, 0), (0, pad_len)), "constant")

    X = X[:, :, :N_TIME_POINTS]

    X_test = np.concatenate(xs_test, axis=0) if xs_test else None
    y_test = np.concatenate(ys_test, axis=0) if ys_test else None

    return X, y, X_test, y_test


def create_crops(X, y, window_size, stride, is_train=True):
    """
    X: (N_trials, n_chans, n_times)
    Returns:
       X_crops: (N_trials * n_crops, n_chans, window_size)
       y_crops: (N_trials * n_crops,)
       trial_indices: (N_trials * n_crops,) - 元の試行ID（Voting用）
    """
    n_trials, n_chans, n_times = X.shape
    crops = []
    labels = []
    trial_idxs = []

    # Calculate starting points (e.g., 512-464 with stride 6 => 9 crops)
    max_start = max(0, n_times - window_size)
    starts = range(0, max_start + 1, stride)

    for i in range(n_trials):
        for start in starts:
            end = start + window_size
            crops.append(X[i, :, start:end])
            labels.append(y[i])
            trial_idxs.append(i)

    return np.array(crops), np.array(labels), np.array(trial_idxs)


def _aggregate_probs_and_loss(model, loader, criterion):
    model.eval()
    probs_by_tid = {}
    targets = {}
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for X_b, y_b, t_idx_b in loader:
            X_b = X_b.to(DEVICE)
            outputs = model(X_b)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.dim() > 2:
                outputs = outputs.mean(dim=tuple(range(2, outputs.dim())))
            loss = criterion(outputs, y_b.to(DEVICE))
            total_loss += loss.item() * y_b.size(0)
            total_count += y_b.size(0)
            prob = torch.softmax(outputs, dim=1).cpu().numpy()
            y_np = y_b.numpy()
            t_np = t_idx_b.numpy()
            for i in range(len(y_np)):
                tid = int(t_np[i])
                if tid not in probs_by_tid:
                    probs_by_tid[tid] = []
                    targets[tid] = int(y_np[i])
                probs_by_tid[tid].append(prob[i])
    preds = []
    gts = []
    for tid in sorted(probs_by_tid.keys()):
        mean_prob = np.mean(probs_by_tid[tid], axis=0)
        preds.append(int(np.argmax(mean_prob)))
        gts.append(int(targets[tid]))
    mean_loss = total_loss / max(total_count, 1)
    return preds, gts, mean_loss


def _log_epoch_metrics(epoch, train_loss, train_acc, val_loss, val_acc):
    if not mlflow.active_run():
        return
    mlflow.log_metrics(
        {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        step=epoch,
    )


def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, history, patience_counter):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_acc": best_acc,
            "best_model_state": getattr(model, "_best_state", None),
            "history": history,
            "patience_counter": patience_counter,
        },
        path,
    )


def _load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler):
    # Respect global SKIP_CHECKPOINTS for quick/dry-run modes
    try:
        if SKIP_CHECKPOINTS:
            return 1, 0.0, None, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}, 0
    except NameError:
        pass
    if not os.path.isdir(ckpt_dir):
        return 1, 0.0, None, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}, 0
    candidates = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not candidates:
        return 1, 0.0, None, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}, 0
    latest = sorted(candidates)[-1]
    ckpt_path = os.path.join(ckpt_dir, latest)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state.get("model_state", {}))
    if optimizer is not None and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state"):
        scheduler_state = state["scheduler_state"]
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
    start_epoch = int(state.get("epoch", 0)) + 1
    best_acc = float(state.get("best_acc", 0.0))
    best_model_state = state.get("best_model_state")
    history = state.get(
        "history",
        {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []},
    )
    patience_counter = int(state.get("patience_counter", 0))
    print(f"[resume] Loaded checkpoint {ckpt_path}, resume from epoch {start_epoch}")
    return start_epoch, best_acc, best_model_state, history, patience_counter


def train_model(
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    n_classes,
    n_chans,
    window_size,
    ckpt_dir,
):
    # Initialize Model with model-specific builder
    builder = MODELS[model_name]
    model = builder(n_outputs=n_classes, n_chans=n_chans, n_times=window_size)

    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    X_tr_crops, y_tr_crops, _ = create_crops(X_train, y_train, window_size, CROP_STRIDE, is_train=True)
    X_val_crops, y_val_crops, val_trial_idxs = create_crops(X_val, y_val, window_size, CROP_STRIDE, is_train=False)
    if X_test is not None and y_test is not None:
        X_te_crops, y_te_crops, te_trial_idxs = create_crops(X_test, y_test, window_size, CROP_STRIDE, is_train=False)
    else:
        X_te_crops = y_te_crops = te_trial_idxs = None

    train_loader = DataLoader(
        TensorDataset(torch.Tensor(X_tr_crops), torch.LongTensor(y_tr_crops)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.Tensor(X_val_crops), torch.LongTensor(y_val_crops), torch.LongTensor(val_trial_idxs)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = None
    if X_te_crops is not None:
        test_loader = DataLoader(
            TensorDataset(torch.Tensor(X_te_crops), torch.LongTensor(y_te_crops), torch.LongTensor(te_trial_idxs)),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

    # モデルごとの入力形状調整（全モデルで B,C,T を採用）
    def prepare_input(x):
        return x

    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch, best_acc, best_model_state, history, patience_counter = _load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model._best_state = best_model_state

    # Ensure history lists exist
    for k in ("train_loss", "train_acc", "val_loss", "val_acc"):
        history.setdefault(k, [])

    best_state = model.state_dict() if best_acc > 0 else None
    best_epoch = start_epoch - 1

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for X_batch, y_batch in train_loader:
            X_batch = prepare_input(X_batch).to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.dim() > 2:
                outputs = outputs.mean(dim=tuple(range(2, outputs.dim())))
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y_batch.size(0)
            pred = outputs.argmax(dim=1)
            correct_train += (pred == y_batch).sum().item()
            total_train += y_batch.size(0)

        scheduler.step()

        train_loss_mean = train_loss / max(total_train, 1)
        train_acc = correct_train / max(total_train, 1)

        # Val aggregation
        val_preds, val_targets, val_loss = _aggregate_probs_and_loss(model, val_loader, criterion)
        val_acc = float(np.mean(np.equal(val_preds, val_targets))) if val_targets else 0.0

        history["train_loss"].append(train_loss_mean)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        _log_epoch_metrics(epoch, train_loss_mean, train_acc, val_loss, val_acc)

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | train_loss={train_loss_mean:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Save checkpoint every epoch
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
        _save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_acc, history, patience_counter)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            best_epoch = epoch
            model._best_state = best_state
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    # Load best state for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    test_preds = test_targets = None
    if test_loader is not None:
        test_preds, test_targets, _ = _aggregate_probs_and_loss(model, test_loader, criterion)
    return best_acc, history, (test_preds, test_targets), (val_preds, val_targets), best_epoch


def _configure_mlflow() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "EEGConformer")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[info] MLflow tracking URI set to {tracking_uri}, experiment '{experiment_name}'")


def main():
    args = _parse_args()
    # allow CLI overrides for quick verification runs
    global SUBJECT_IDS, N_FOLDS, MAX_EPOCHS, RESULTS_ROOT, BATCH_SIZE
    if args.subjects:
        parsed = []
        for token in args.subjects.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                a, b = token.split("-")
                parsed.extend(list(range(int(a), int(b) + 1)))
            else:
                parsed.append(int(token))
        SUBJECT_IDS = parsed
    if args.n_folds:
        N_FOLDS = args.n_folds
    if args.max_epochs:
        MAX_EPOCHS = args.max_epochs
    if args.results_root:
        RESULTS_ROOT = args.results_root
    if args.dry_run:
        # minimal run: single subject, 2 folds, 1 epoch
        if isinstance(SUBJECT_IDS, range):
            SUBJECT_IDS = [SUBJECT_IDS[0]]
        elif isinstance(SUBJECT_IDS, list) and len(SUBJECT_IDS) > 0:
            SUBJECT_IDS = [SUBJECT_IDS[0]]
        else:
            SUBJECT_IDS = [1]
        N_FOLDS = 2
        MAX_EPOCHS = 1
        BATCH_SIZE = max(4, BATCH_SIZE // 4)
        # avoid loading/resuming from checkpoints during dry-run
        SKIP_CHECKPOINTS = True

    _configure_mlflow()
    results_dir = ensure_results_dir()
    results = []

    for subject_id in SUBJECT_IDS:
        print(f"--- Processing Subject {subject_id} ---")
        X, y, X_test, y_test = load_subject_data(subject_id)
        if X is None:
            continue

        print(f"Data Loaded: {X.shape}, {y.shape}")
        n_chans = X.shape[1]
        n_classes = len(np.unique(y))

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold+1}/{N_FOLDS}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            for model_name in MODELS:
                print(f"Training {model_name}...")
                model_run_dir = os.path.join(results_dir, f"S{subject_id:02d}", f"fold{fold+1:02d}", model_name)
                test_acc_val = None
                acc = 0.0
                history = {}
                test_preds_targets = (None, None)
                val_preds_targets = (None, None)
                best_epoch = None

                run_name = f"S{subject_id:02d}_fold{fold+1}_{model_name}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params({
                        "subject_id": subject_id,
                        "model_name": model_name,
                        "fold": fold + 1,
                        "batch_size": BATCH_SIZE,
                        "window_size": CROP_WINDOW,
                        "n_splits": N_FOLDS,
                        "learning_rate": LR,
                        "data_root": DATA_ROOT,
                        "results_dir": results_dir,
                    })
                    try:
                        acc, history, test_preds_targets, val_preds_targets, best_epoch = train_model(
                            model_name,
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                            n_classes,
                            n_chans,
                            window_size=CROP_WINDOW,
                            ckpt_dir=os.path.join(model_run_dir, "checkpoints"),
                        )
                    except Exception as e:
                        print(f"[warn] {model_name} crashed: {e}; skipping and continuing.")
                        acc = 0.0
                        history = {}
                        test_preds_targets = (None, None)
                        val_preds_targets = (None, None)
                        best_epoch = None
                        mlflow.set_tag("training.status", "failed")
                        mlflow.log_param("training.error", str(e))
                    else:
                        mlflow.set_tag("training.status", "completed")
                    preds_te, targets_te = test_preds_targets
                    if preds_te is not None and targets_te is not None and len(preds_te) > 0:
                        test_acc_val = float(np.mean(np.equal(preds_te, targets_te)))
                    if mlflow.active_run():
                        mlflow.log_metric("best_val_acc", acc)
                        if best_epoch is not None:
                            mlflow.log_metric("best_epoch", best_epoch)
                        if test_acc_val is not None:
                            mlflow.log_metric("test_accuracy", test_acc_val)
                        mlflow.log_metric("epochs_trained", len(history.get("train_loss", [])))

                # Save history CSV/JSON
                os.makedirs(model_run_dir, exist_ok=True)
                epochs = list(range(1, len(history.get("train_loss", [])) + 1))
                hist_df = pd.DataFrame({
                    "epoch": epochs,
                    "train_loss": history.get("train_loss", []),
                    "train_acc": history.get("train_acc", []),
                    "val_loss": history.get("val_loss", []),
                    "val_acc": history.get("val_acc", []),
                    "subject": f"S{subject_id:02d}",
                    "model": model_name,
                    "fold": fold + 1,
                    "best_epoch": best_epoch,
                })
                hist_json_path = os.path.join(model_run_dir, "history.json")
                hist_csv_path = os.path.join(model_run_dir, "history.csv")
                hist_df.to_json(hist_json_path, orient="records", indent=2)
                hist_df.to_csv(hist_csv_path, index=False)

                # Plot learning curves
                try:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
                    ax.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(os.path.join(model_run_dir, "learning_curves_loss.png"), dpi=120)
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(hist_df["epoch"], hist_df["train_acc"], label="train_acc")
                    ax.plot(hist_df["epoch"], hist_df["val_acc"], label="val_acc")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Accuracy")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(os.path.join(model_run_dir, "learning_curves_acc.png"), dpi=120)
                    plt.close(fig)
                except Exception as plot_err:
                    print(f"[warn] Failed to plot curves: {plot_err}")

                # Confusion matrix (test preferred, else val) using best model
                try:
                    preds, targets = test_preds_targets
                    if preds is None or targets is None or len(preds) == 0:
                        preds, targets = val_preds_targets
                    if preds is not None and targets is not None and len(preds) > 0:
                        cm = confusion_matrix(targets, preds, labels=list(range(n_classes)))
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax,
                                    xticklabels=list(range(n_classes)), yticklabels=list(range(n_classes)))
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("True")
                        ax.set_title(f"Confusion Matrix {model_name} fold {fold+1}")
                        fig.tight_layout()
                        fig.savefig(os.path.join(model_run_dir, "confusion_matrix.png"), dpi=120)
                        plt.close(fig)
                except Exception as cm_err:
                    print(f"[warn] Failed to save confusion matrix: {cm_err}")

                if test_acc_val is not None:
                    print(f"[S{subject_id:02d}][fold {fold+1}][{model_name}] test_acc={test_acc_val:.4f}")
                else:
                    print(f"[S{subject_id:02d}][fold {fold+1}][{model_name}] test_acc=NA (no test split)")

                results.append({
                    "Subject": subject_id,
                    "Fold": fold + 1,
                    "Model": model_name,
                    "Accuracy": acc
                })

    df = pd.DataFrame(results)
    output_path = os.path.join(results_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(output_path, index=False)
    print(f"Experiment Completed. Saved to {output_path}.")


if __name__ == "__main__":
    main()
