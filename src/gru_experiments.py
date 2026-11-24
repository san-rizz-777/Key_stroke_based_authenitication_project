"""
rnn_experiments_fixed.py — fully GPU-resident RNN ablation suite for keystroke dynamics.

Crash-hardened version:
  * faulthandler + file logging at the very top (find out WHERE/WHEN the kernel dies)
  * cudnn.benchmark disabled (re-tuning on changing batch shapes can segfault)
  * GPU memory freed after every run (del + gc + empty_cache)
  * partial results saved after every run, and finished runs are skipped on restart
"""

# ============================================================
# 0. CRASH DIAGNOSTICS — keep this block at the very top
# ============================================================
import os
import sys
import faulthandler
import logging

# Set to True ONLY when debugging: makes CUDA errors point at the exact failing line
# (much slower). Must be set before torch is imported.
DEBUG_SYNC = False
if DEBUG_SYNC:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Prints a Python traceback into this file even on a hard crash (segfault, abort).
# Kept open for the lifetime of the process on purpose.
_fault_file = open("faulthandler.log", "w")
faulthandler.enable(file=_fault_file, all_threads=True)

# Log to both the notebook output and run.log (so you can read it after a crash).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler("run.log", mode="a"),
              logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.info

# ============================================================
# IMPORTS
# ============================================================
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, PowerTransformer
import warnings
warnings.filterwarnings("ignore")

log(f"torch={torch.__version__} | cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available — enable the GPU accelerator in Kaggle settings.")

DEVICE = torch.device("cuda")
log(f"GPU={torch.cuda.get_device_name(0)} | capability={torch.cuda.get_device_capability(0)}")

# FIX #1: benchmark=True re-tunes kernels for every new tensor shape. Our last batch of
# each epoch and the predict() batches have different shapes, which can crash the kernel.
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")       # allow TF32 matmuls on Ampere+ GPUs

# ============================================================
# CONFIG
# ============================================================
SEEDS        = [0, 1, 2]     # each experiment is repeated per seed to get mean ± std
BATCH_SIZE   = 1024
EPOCHS       = 100           # upper bound; early stopping usually ends sooner
PATIENCE     = 10            # epochs without val improvement before stopping
LR           = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE  = 128
DROPOUT      = 0.3
NUM_CLASSES  = 126           # number of subjects (users) to identify

PARTIAL_CSV  = "rnn_ablation_results_partial.csv"
FINAL_CSV    = "rnn_ablation_results.csv"


def gpu_mem():
    """Short string with current GPU memory use, for the logs."""
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return f"alloc={alloc:.0f}MB reserved={reserved:.0f}MB"


# ============================================================
# DATA LOADING (CPU, runs once)
# ============================================================
def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)

    # Drop metadata columns that are not keystroke timing features
    drop_cols = ["session", "time", "errors", "psr", "totaltime"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    feature_cols = [c for c in df.columns if c not in ("subject", "sessionIndex")]
    X = df[feature_cols].values.astype(np.float32, copy=True)
    y = df["subject"].values
    sessions = df["sessionIndex"].values

    # IMPORTANT: fit the scalers on sessions 1-5 only, then apply to later sessions.
    # Fitting on all data would leak test-set statistics into training.
    train_mask = sessions <= 5

    # Yeo-Johnson makes the skewed timing distributions more Gaussian-like
    pt = PowerTransformer(method="yeo-johnson")
    X[train_mask]  = pt.fit_transform(X[train_mask]).astype(np.float32)
    X[~train_mask] = pt.transform(X[~train_mask]).astype(np.float32)

    # Then standardize to zero mean / unit variance per feature
    scaler = StandardScaler()
    X[train_mask]  = scaler.fit_transform(X[train_mask]).astype(np.float32)
    X[~train_mask] = scaler.transform(X[~train_mask]).astype(np.float32)

    # Map arbitrary subject IDs (e.g. "s002") to contiguous class indices 0..125
    unique_subjects = np.unique(y)
    label_map = {s: i for i, s in enumerate(unique_subjects)}
    y_encoded = np.array([label_map[s] for s in y], dtype=np.int64)
    return X.astype(np.float32, copy=False), y_encoded, sessions, feature_cols


# ============================================================
# GPU AUGMENTATION
# All functions take a (B, 31) CUDA tensor and return a new tensor.
# ============================================================
def augment_global(x, scale=0.10, jitter=0.002):
    """One random multiplicative factor per sample (shared by all features) + small Gaussian noise.
    Simulates a user typing slightly faster/slower overall."""
    factor = torch.empty(x.shape[0], 1, device=x.device).uniform_(1 - scale, 1 + scale)
    return x * factor + torch.randn_like(x) * jitter


def augment_per_feature(x, scale=0.10, jitter=0.002):
    """Independent random factor for EVERY feature. The H/DD/UD column groups cover all
    31 columns, so one factor per column is equivalent to the original per-group logic."""
    factor = torch.empty_like(x).uniform_(1 - scale, 1 + scale)
    return x * factor + torch.randn_like(x) * jitter


def static_augment(X, y, n_copies=2, scale=0.10):
    """Pre-generate augmented copies once and concatenate them with the original data.
    Unlike online augmentation, the same noisy copies are reused every epoch."""
    Xs, ys = [X], [y]
    for _ in range(n_copies):
        Xs.append(augment_global(X, scale=scale))
        ys.append(y)
    return torch.cat(Xs), torch.cat(ys)


def to_sequence(x):
    """(B, 31) -> (B, 11, 3): one timestep per key, with channels [hold, down-down, up-down].
    Hold has 11 values, DD and UD have 10 (they're between consecutive keys), so those two
    are edge-padded by repeating their last value to reach length 11. All done on the GPU."""
    h  = x[:, 0:11]
    dd = torch.cat([x[:, 11:21], x[:, 20:21]], dim=1)   # repeat last DD value
    ud = torch.cat([x[:, 21:31], x[:, 30:31]], dim=1)   # repeat last UD value
    return torch.stack([h, dd, ud], dim=2)


# ============================================================
# GPU BATCHING (replaces Dataset + DataLoader)
# ============================================================
def iterate_batches(X, y, batch_size, shuffle):
    """Yield (X, y) mini-batches by indexing GPU tensors directly.
    A fresh random permutation each epoch when shuffle=True; sequential order otherwise."""
    n = X.shape[0]
    if shuffle:
        perm = torch.randperm(n, device=X.device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            yield X[idx], y[idx]
    else:
        for i in range(0, n, batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]


# ============================================================
# MODEL
# ============================================================
class AttentionPool(nn.Module):
    """Learned weighted average over timesteps (instead of a plain mean)."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)           # scalar score per timestep

    def forward(self, x):                               # x: (B, T, H)
        scores  = self.attn(x).squeeze(-1)              # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # normalize over time
        return (x * weights).sum(dim=1)                 # (B, H)


class RNNModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2,
                 num_classes=126, dropout=0.3, pool_type="mean"):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True,
                          # inter-layer dropout is invalid (and warns) for a single layer
                          dropout=dropout if num_layers > 1 else 0.0)
        self.pool_type = pool_type
        if pool_type == "attention":
            self.attn_pool = AttentionPool(hidden_size * 2)   # *2 because bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.rnn(x)                            # (B, T, 2H)
        pooled = self.attn_pool(out) if self.pool_type == "attention" else out.mean(dim=1)
        return self.fc(self.dropout(pooled))            # logits (B, num_classes)


# ============================================================
# GPU METRICS (no sklearn, no per-batch CPU sync)
# ============================================================
@torch.no_grad()
def predict(model, X, batch_size=BATCH_SIZE):
    """Return predicted class indices for all of X.
    Uses the same batch size as training so cuDNN sees fewer distinct shapes."""
    model.eval()                                        # disables dropout
    return torch.cat([model(to_sequence(X[i:i + batch_size])).argmax(1)
                      for i in range(0, X.shape[0], batch_size)])


@torch.no_grad()
def accuracy_gpu(model, X, y):
    # .item() is the only GPU->CPU sync here
    return (predict(model, X) == y).float().mean().item()


@torch.no_grad()
def acc_f1_gpu(model, X, y, num_classes=NUM_CLASSES):
    """Accuracy + macro-F1 computed with bincount, matching sklearn's f1_score(average='macro')."""
    p = predict(model, X)
    acc = (p == y).float().mean().item()

    tp       = torch.bincount(y[p == y], minlength=num_classes).float()  # true positives per class
    pred_cnt = torch.bincount(p, minlength=num_classes).float()          # tp + fp
    true_cnt = torch.bincount(y, minlength=num_classes).float()          # tp + fn
    denom = pred_cnt + true_cnt                                          # = 2tp + fp + fn
    present = denom > 0   # sklearn averages over labels appearing in y_true OR y_pred
    f1 = (2 * tp[present] / denom[present]).mean().item()                # F1 = 2tp / (2tp+fp+fn)
    return acc, f1


# ============================================================
# TRAINING
# ============================================================
def train_step(model, optimizer, criterion, xb, yb, aug_fn):
    """One optimization step on a single GPU batch."""
    if aug_fn is not None:
        xb = aug_fn(xb)                                 # online augmentation, fresh each step
    optimizer.zero_grad(set_to_none=True)               # set_to_none is slightly faster than zeroing
    loss = criterion(model(to_sequence(xb)), yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # stabilizes RNN training
    optimizer.step()


def train_model(model, X_tr, y_tr, X_val, y_val, aug_fn,
                epochs=EPOCHS, patience=PATIENCE):
    """Train with early stopping on validation accuracy; restore the best weights at the end."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Halve LR if val accuracy hasn't improved for 5 epochs (mode="max" because higher is better)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc, best_state, best_epoch, no_improve = 0.0, None, 0, 0

    for epoch in range(epochs):
        model.train()                                   # enables dropout
        for xb, yb in iterate_batches(X_tr, y_tr, BATCH_SIZE, shuffle=True):
            train_step(model, optimizer, criterion, xb, yb, aug_fn)

        val_acc = accuracy_gpu(model, X_val, y_val)     # one sync per epoch, not per batch
        scheduler.step(val_acc)

        # Progress log: if the kernel dies, the last line shows the epoch it reached
        if epoch % 5 == 0:
            log(f"    epoch {epoch:3d} | val_acc={val_acc:.4f} | {gpu_mem()}")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, no_improve = val_acc, epoch, 0
            # Snapshot best weights; clone() is required, otherwise it aliases live parameters.
            # Stays on the GPU: no CPU round trip.
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break                                   # early stop

    model.load_state_dict(best_state)
    del best_state, optimizer, scheduler                # release GPU memory held by the snapshot
    return model, best_val_acc, best_epoch


# ============================================================
# EXPERIMENT RUNNER
# ============================================================
def run_experiment(X_all, y_all, sessions, config, seed):
    # Seed all RNGs. Augmentation uses the torch RNG, so results differ from the old
    # NumPy-based version (statistically comparable, not bit-identical).
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Session index as a GPU tensor so boolean masks index the GPU data directly
    sess = torch.from_numpy(sessions).to(DEVICE)

    # Temporal split: train on sessions 1-4, validate on 5, test on later sessions 6+
    X_tr,  y_tr  = X_all[sess <= 4],  y_all[sess <= 4]
    X_val, y_val = X_all[sess == 5],  y_all[sess == 5]
    X_te,  y_te  = X_all[sess >= 6],  y_all[sess >= 6]

    # Static augmentation enlarges the dataset once up-front
    if config["aug_type"] == "static":
        X_tr, y_tr = static_augment(X_tr, y_tr, 2, config["scale"])

    # Online augmentation is applied per batch inside train_step
    if config["aug_type"] == "online":
        aug_fn = lambda x: augment_global(x, config["scale"], config["jitter"])
    elif config["aug_type"] == "per_feature":
        aug_fn = lambda x: augment_per_feature(x, config["scale"], config["jitter"])
    else:
        aug_fn = None                                   # "none" or "static" (already applied)

    def make_model():
        return RNNModel(num_layers=config["num_layers"], pool_type=config["pool_type"],
                        hidden_size=HIDDEN_SIZE, dropout=DROPOUT,
                        num_classes=NUM_CLASSES).to(DEVICE)

    # Phase 1: train on sessions 1-4 with early stopping on session 5 to find best_epoch
    log("  phase 1: train on S1-4, validate on S5")
    model = make_model()
    model, best_val_acc, best_epoch = train_model(model, X_tr, y_tr, X_val, y_val, aug_fn)
    del model                                           # phase-1 model is no longer needed

    # Phase 2: retrain a fresh model on sessions 1-5 for best_epoch+1 epochs (no validation
    # set left, so we reuse the epoch count found in phase 1). Uses more data for the final model.
    log(f"  phase 2: retrain on S1-5 for {best_epoch + 1} epochs")
    X_full, y_full = X_all[sess <= 5], y_all[sess <= 5]
    if config["aug_type"] == "static":
        X_full, y_full = static_augment(X_full, y_full, 2, config["scale"])

    model_full = make_model()
    optimizer = torch.optim.AdamW(model_full.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    for _ in range(best_epoch + 1):
        model_full.train()
        for xb, yb in iterate_batches(X_full, y_full, BATCH_SIZE, shuffle=True):
            train_step(model_full, optimizer, criterion, xb, yb, aug_fn)

    # Final evaluation on all held-out sessions
    test_acc, test_f1 = acc_f1_gpu(model_full, X_te, y_te)

    # Per-session accuracy shows how performance drifts as time passes since training.
    # Key names match the aggregation below (acc_session6/7/8).
    session_accs = {}
    for s in (6, 7, 8):
        m = sess == s
        session_accs[f"acc_session{s}"] = accuracy_gpu(model_full, X_all[m], y_all[m])

    # FIX #2: drop references so the GPU memory can actually be reclaimed
    del model_full, optimizer, X_tr, y_tr, X_val, y_val, X_te, y_te, X_full, y_full, sess

    return {"seed": seed, "val_acc": best_val_acc, "best_epoch": best_epoch,
            "test_acc": test_acc, "test_f1": test_f1, **session_accs}


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    CSV_PATH = "/kaggle/input/datasets/sanskarrgunde/lab-dataset/lab_dataset.csv"
    log("loading data")
    X, y, sessions, feature_cols = load_and_preprocess(CSV_PATH)
    log(f"data loaded: X={X.shape}, classes={len(np.unique(y))}")

    # Move the full dataset to the GPU once; everything downstream stays there
    X_gpu = torch.from_numpy(X).to(DEVICE)
    y_gpu = torch.from_numpy(y).to(DEVICE)

    # Ablation grid: layers (1/2/4), augmentation strength (0/5/10/15/20%),
    # augmentation style (online / per-feature / static), and pooling (mean / attention)
    experiments = [
        {"name": "GRU_1L_mean_aug10", "num_layers": 1, "pool_type": "mean",      "aug_type": "online",      "scale": 0.10, "jitter": 0.002},
        {"name": "GRU_2L_mean_aug10", "num_layers": 2, "pool_type": "mean",      "aug_type": "online",      "scale": 0.10, "jitter": 0.002},
        {"name": "GRU_4L_mean_aug10", "num_layers": 4, "pool_type": "mean",      "aug_type": "online",      "scale": 0.10, "jitter": 0.002},
        {"name": "GRU_2L_mean_aug00", "num_layers": 2, "pool_type": "mean",      "aug_type": "none",        "scale": 0.00, "jitter": 0.000},
        {"name": "GRU_2L_mean_aug05", "num_layers": 2, "pool_type": "mean",      "aug_type": "online",      "scale": 0.05, "jitter": 0.002},
        {"name": "GRU_2L_mean_aug15", "num_layers": 2, "pool_type": "mean",      "aug_type": "online",      "scale": 0.15, "jitter": 0.002},
        {"name": "GRU_2L_mean_aug20", "num_layers": 2, "pool_type": "mean",      "aug_type": "online",      "scale": 0.20, "jitter": 0.002},
        {"name": "GRU_2L_mean_perfeat10", "num_layers": 2, "pool_type": "mean",  "aug_type": "per_feature", "scale": 0.10, "jitter": 0.002},
        {"name": "GRU_2L_attn_aug10",     "num_layers": 2, "pool_type": "attention","aug_type": "online",    "scale": 0.10, "jitter": 0.002},
        {"name": "GRU_2L_mean_static10",  "num_layers": 2, "pool_type": "mean",   "aug_type": "static",      "scale": 0.10, "jitter": 0.002},
    ]

    # FIX #3 (resume): if a previous run crashed, reload its finished results and skip them
    all_results = []
    done = set()
    if os.path.exists(PARTIAL_CSV):
        prev = pd.read_csv(PARTIAL_CSV)
        all_results = prev.to_dict("records")
        done = {(r["name"], int(r["seed"])) for r in all_results}
        log(f"resuming: {len(done)} runs already finished")

    for exp in experiments:
        for seed in SEEDS:
            if (exp["name"], seed) in done:
                continue

            log(f">>> {exp['name']} | seed={seed} | {gpu_mem()}")
            t0 = time.time()
            res = run_experiment(X_gpu, y_gpu, sessions, exp, seed)
            res["name"] = exp["name"]
            res["minutes"] = (time.time() - t0) / 60
            all_results.append(res)

            # Free cached GPU memory between runs
            gc.collect()
            torch.cuda.empty_cache()

            # Save after EVERY run so a crash never loses finished work
            pd.DataFrame(all_results).to_csv(PARTIAL_CSV, index=False)
            log(f"    test_acc={res['test_acc']:.4f}  val_acc={res['val_acc']:.4f}  "
                f"time={res['minutes']:.2f}m | {gpu_mem()}")

    # Save raw per-seed results, then print mean ± std per experiment
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(FINAL_CSV, index=False)
    log(f"Saved -> {FINAL_CSV}")

    agg = df_results.groupby("name").agg({
        "test_acc":      ["mean", "std"],
        "test_f1":       ["mean", "std"],
        "acc_session6":  ["mean", "std"],
        "acc_session7":  ["mean", "std"],
        "acc_session8":  ["mean", "std"],
        "best_epoch":    "mean",
        "minutes":       "mean",
    }).round(4)
    print("\n=== AGGREGATED (mean ± std over seeds) ===")
    print(agg.to_string())