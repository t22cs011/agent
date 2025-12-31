import sys
import os
import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from braindecode.models import EEGConformer, ShallowFBCSPNet, Deep4Net, EEGNetv4
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore", message=".*Tensorflow.*")

# --- Configuration ---
DATA_ROOT = "/mnt/bci_source/data/preprocessed_mixed_256hz_v2"
SUBJECT_IDS = range(1, 16)  # S01 to S15
N_FOLDS = 10
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
LR = 0.000625  # 一般的なBraindecodeのデフォルト
WEIGHT_DECAY = 0

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
        input_window_samples=n_times,
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
        input_window_samples=n_times,
        sfreq=256,
        drop_prob=0.5,
    ),
    "EEGNet": lambda n_outputs, n_chans, n_times: EEGNetv4(
        n_outputs=n_outputs,
        n_chans=n_chans,
        input_window_samples=n_times,
        sfreq=256,
        F1=8,
        D=2,
        F2=16,
        drop_prob=0.25,
    ),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Limit thread count to reduce instability on noisy hardware
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)


def ensure_results_dir(preferred_dir="/tmp/results", fallback_dir="/tmp"):
    """Choose a writable directory for results, preferring /tmp/results."""
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
    """Load subject data with flexible key names and concatenate train/val/test if present."""
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

    xs = []
    ys = []

    for split_name, x_keys, y_keys in split_key_options:
        x_key, x_arr = pick_first(x_keys)
        y_key, y_arr = pick_first(y_keys)

        if x_key and y_key:
            x_shape = getattr(x_arr, "shape", "n/a")
            y_shape = getattr(y_arr, "shape", "n/a")
            print(f"    [debug][S{subject_id:02d}] {split_name}: {x_key} shape={x_shape}, {y_key} shape={y_shape}")
            xs.append(x_arr)
            ys.append(y_arr)
        elif x_key and not y_key:
            print(f"    [warn][S{subject_id:02d}] Found {x_key} but missing label key among {y_keys}")
        elif y_key and not x_key:
            print(f"    [warn][S{subject_id:02d}] Found {y_key} but missing feature key among {x_keys}")
        else:
            print(f"    [debug][S{subject_id:02d}] No data found for split '{split_name}'")

    if not xs or not ys:
        print(f"[warn][S{subject_id:02d}] No feature/label arrays collected; skipping subject")
        return None, None

    try:
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
    except Exception as e:
        print(f"[error][S{subject_id:02d}] Failed to concatenate arrays: {e}")
        return None, None

    if X.ndim != 3:
        print(f"[warn][S{subject_id:02d}] X has unexpected ndim={X.ndim}; expected 3")
        return None, None
    if y.ndim != 1:
        print(f"[warn][S{subject_id:02d}] y has unexpected ndim={y.ndim}; expected 1")
        return None, None
    if X.shape[0] != y.shape[0]:
        print(f"[warn][S{subject_id:02d}] Mismatch between samples and labels: X={X.shape}, y={y.shape}")
        return None, None

    print(f"[debug][S{subject_id:02d}] Combined X shape: {X.shape}, y shape: {y.shape}")

    if X.shape[2] < N_TIME_POINTS:
        pad_len = N_TIME_POINTS - X.shape[2]
        print(f"[debug][S{subject_id:02d}] Padding time axis by {pad_len}")
        X = np.pad(X, ((0, 0), (0, 0), (0, pad_len)), "constant")

    X = X[:, :, :N_TIME_POINTS]

    return X, y


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


def train_model(model_name, X_train, y_train, X_val, y_val, n_classes, n_chans, window_size):
    # Initialize Model with model-specific builder
    builder = MODELS[model_name]
    model = builder(n_outputs=n_classes, n_chans=n_chans, n_times=window_size)

    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    X_tr_crops, y_tr_crops, _ = create_crops(X_train, y_train, window_size, CROP_STRIDE, is_train=True)
    X_val_crops, y_val_crops, val_trial_idxs = create_crops(X_val, y_val, window_size, CROP_STRIDE, is_train=False)

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

    # モデルごとの入力形状調整（全モデルで B,C,T を採用）
    def prepare_input(x):
        return x

    # 事前に1バッチで形状を検証し、失敗するモデルはスキップ
    try:
        sample_X, _ = next(iter(train_loader))
        print(f"[dry-run] {model_name} sample batch shape: {sample_X.shape}")
        sample_X = prepare_input(sample_X).to(DEVICE)
        _ = model(sample_X)
    except Exception as e:
        print(f"[warn] {model_name} dry-run forward failed (shape mismatch?): {e}; skipping this model.")
        return 0.0

    best_acc = 0
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0
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
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_probs = {}
        val_targets = {}

        with torch.no_grad():
            for X_b, y_b, t_idx_b in val_loader:
                X_b = prepare_input(X_b).to(DEVICE)
                outputs = model(X_b)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if outputs.dim() > 2:
                    outputs = outputs.mean(dim=tuple(range(2, outputs.dim())))
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                y_b = y_b.numpy()
                t_idx_b = t_idx_b.numpy()

                for i in range(len(y_b)):
                    tid = t_idx_b[i]
                    if tid not in val_probs:
                        val_probs[tid] = []
                        val_targets[tid] = y_b[i]
                    val_probs[tid].append(probs[i])

        correct = 0
        total = len(val_probs)
        for tid in val_probs:
            mean_prob = np.mean(val_probs[tid], axis=0)
            pred = np.argmax(mean_prob)
            if pred == val_targets[tid]:
                correct += 1

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    return best_acc


def main():
    results_dir = ensure_results_dir()
    results = []

    for subject_id in SUBJECT_IDS:
        print(f"--- Processing Subject {subject_id} ---")
        X, y = load_subject_data(subject_id)
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
                try:
                    acc = train_model(
                        model_name,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        n_classes,
                        n_chans,
                        window_size=CROP_WINDOW,
                    )
                except Exception as e:
                    print(f"[warn] {model_name} crashed: {e}; skipping and continuing.")
                    acc = 0.0

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
