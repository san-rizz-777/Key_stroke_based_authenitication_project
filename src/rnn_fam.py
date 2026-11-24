"""
LSTM and GRU for keystroke identification (PyTorch).

Each typing attempt becomes a sequence of 11 keystrokes. Every step has 4 values:
  [hold time, down-down gap, up-down gap, "a gap exists" flag]
(the last key, Enter, has no following key, so its two gaps are 0 and the flag is 0).

Same protocol as the classical models, so the comparison is fair:
  Phase 1: train on sessions 1-4, early-stop on session 5 -> pick the number of epochs
  Phase 2: retrain on sessions 1-5 for that many epochs, test ONCE on sessions 6-8
Yeo-Johnson is fitted on the training rows only. Augmentation (optional) rescales every
timing of an attempt by a random factor each epoch, to mimic day-to-day speed drift.

Colab (use a GPU runtime: Runtime > Change runtime type):
    %run rnn.py /content/your_file.csv
Output: results/rnn_results.csv
"""
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer

# ------------------------- settings you can change ---------------------------
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else "data.csv"
KINDS = ["LSTM", "GRU"]
AUG_OPTIONS = [False, True]       # run each model without and with augmentation
SCALE_RANGE = 0.10                # augmentation: multiply timings by a factor in [0.90, 1.10]
JITTER = 0.002                    # augmentation: 2 ms of Gaussian noise
HIDDEN, LAYERS, DROPOUT, BIDIR = 128, 2, 0.3, True
EPOCHS, PATIENCE, BATCH, LR, WD = 60, 8, 256, 2e-3, 1e-4
SEEDS = [0]                       # use [0, 1, 2] for mean +/- std (3x the time)
KNN_BENCHMARK = 0.8079            # your tuned kNN test accuracy, for comparison
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs("results", exist_ok=True)
df = pd.read_csv(CSV_PATH)
H = [c for c in df.columns if c.startswith("H.")]
DD = [c for c in df.columns if c.startswith("DD.")]
UD = [c for c in df.columns if c.startswith("UD.")]
assert (len(H), len(DD), len(UD)) == (11, 10, 10), "expected 11 hold, 10 DD and 10 UD columns"
feats = H + DD + UD                      # grouped order, which to_sequences relies on
le = LabelEncoder().fit(df["subject"])
n_classes = len(le.classes_)

sess = df["sessionIndex"].values
X_all = df[feats].values
y_all = le.transform(df["subject"])


def to_sequences(Xt):
    """(n, 31) in H, DD, UD order -> (n, 11, 4) float32."""
    seq = np.zeros((len(Xt), 11, 4), dtype=np.float32)
    seq[:, :, 0] = Xt[:, :11]            # hold time of each of the 11 keys
    seq[:, :10, 1] = Xt[:, 11:21]        # down-down gap to the next key
    seq[:, :10, 2] = Xt[:, 21:31]        # up-down gap to the next key
    seq[:, :10, 3] = 1.0                 # flag: this step has a gap
    return seq


def augment(X, rng):
    """One random speed factor per attempt, applied to all 31 timings, plus jitter."""
    s = rng.uniform(1 - SCALE_RANGE, 1 + SCALE_RANGE, size=(len(X), 1))
    return X * s + rng.normal(0, JITTER, X.shape)


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        rnn_cls = nn.LSTM if kind == "LSTM" else nn.GRU
        self.rnn = rnn_cls(4, HIDDEN, num_layers=LAYERS, batch_first=True,
                           dropout=DROPOUT if LAYERS > 1 else 0.0, bidirectional=BIDIR)
        out_dim = HIDDEN * (2 if BIDIR else 1)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(out_dim, n_classes))

    def forward(self, x):
        out, _ = self.rnn(x)                 # (batch, 11, out_dim)
        return self.head(out.mean(dim=1))    # average over the 11 keystrokes


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, X_seq):
    model.eval()
    out = []
    for i in range(0, len(X_seq), 2048):
        out.append(model(X_seq[i:i + 2048]).argmax(1).cpu())
    return torch.cat(out).numpy()


def fit(kind, aug, seed, X_tr, y_tr, X_ev=None, y_ev=None, n_epochs=EPOCHS):
    """Train one model. With an eval set: early-stop and report the best epoch."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    pt = PowerTransformer(method="yeo-johnson").fit(X_tr)     # fitted on the real training rows only
    model = Net(kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    y_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_ev_t = None if X_ev is None else torch.tensor(to_sequences(pt.transform(X_ev)), device=device)
    best_acc, best_epoch, bad = -1.0, n_epochs, 0

    for epoch in range(1, n_epochs + 1):
        Xe = augment(X_tr, rng) if aug else X_tr
        X_t = torch.tensor(to_sequences(pt.transform(Xe)), device=device)
        model.train()
        perm = torch.randperm(len(X_t), device=device)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(X_t[b]), y_t[b]).backward()
            opt.step()
        if X_ev_t is not None:
            acc = accuracy_score(y_ev, predict(model, X_ev_t))
            if acc > best_acc:
                best_acc, best_epoch, bad = acc, epoch, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    return pt, model, best_acc, best_epoch


rows = []
for kind in KINDS:
    for aug in AUG_OPTIONS:
        for seed in SEEDS:
            t0 = time.time()
            # Phase 1: sessions 1-4 -> validate on 5 to choose the number of epochs
            m1, m2 = sess <= 4, sess == 5
            _, _, val_acc, best_epoch = fit(kind, aug, seed, X_all[m1], y_all[m1], X_all[m2], y_all[m2])
            # Phase 2: sessions 1-5 for that many epochs, then test once on 6-8
            m_tr, m_te = sess <= 5, sess > 5
            pt, model, _, _ = fit(kind, aug, seed, X_all[m_tr], y_all[m_tr], n_epochs=best_epoch)
            pred = predict(model, torch.tensor(to_sequences(pt.transform(X_all[m_te])), device=device))
            y_te, s_te = y_all[m_te], sess[m_te]
            row = {
                "model": kind, "augmentation": aug, "seed": seed, "best_epoch": best_epoch,
                "val_accuracy": round(val_acc, 4),
                "test_accuracy": round(accuracy_score(y_te, pred), 4),
                "test_macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
                "minutes": round((time.time() - t0) / 60, 1),
            }
            for s in sorted(np.unique(s_te)):
                row[f"acc_session{s}"] = round(accuracy_score(y_te[s_te == s], pred[s_te == s]), 4)
            rows.append(row)
            print(row)

res = pd.DataFrame(rows)
res.to_csv("results/rnn_results.csv", index=False)
cols = ["test_accuracy", "test_macro_f1"] + [c for c in res.columns if c.startswith("acc_session")]
print("\n", res.groupby(["model", "augmentation"])[cols].mean().round(4).to_string())
print(f"\nTuned kNN benchmark: {KNN_BENCHMARK}")