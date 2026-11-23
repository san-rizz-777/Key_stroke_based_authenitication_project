"""
Does simulating typing-speed drift in the TRAINING data help on later sessions?

Idea: test accuracy falls from session 6 to 8, and typing speed seems to shift
between days. Here every training attempt gets extra copies where all timings are
multiplied by a random factor (e.g. 0.90 to 1.10) plus a little jitter. Only the
training sessions (1-5) are augmented; sessions 6-8 are untouched.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from pandas.core.common import random_state
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else "data/lab_dataset.csv"
SCALE_RANGES = [0.0, 0.05, 0.10, 0.20]   # 0.0 = no augmentation (the baseline)
COPIES = 2                                # extra augmented copies per training row
JITTER = 0.002                            # seconds (2 ms) of Gaussian noise
USE_SVM = True                           # SVM on 3x the rows is slow (many minutes per run)
SEED = 42

# Best settings found by tune.py
KNN_conf = dict(n_neighbors=7, weights="distance", p=1)
SVM_conf = dict(C=1.49, gamma=0.0397)
RandomForest_conf = dict(n_estimators=400, min_samples_leaf=1,max_features="sqrt",max_depth=20)
XGBoost_conf = dict(tree_method="hist", colsample_bytree=0.9233589392465844,learning_rate=0.06049716507542575,max_depth=3,min_child_weight=3,n_estimators=300,reg_lambda=3.8718305057994216,subsample=0.8439986631130484)
LGBM_conf = dict(subsample_freq=1, colsample_bytree=0.6232334448672797,learning_rate=0.22044355215811345,min_child_samples=40,n_estimators=200, num_leaves=15,reg_lambda=8.706020878304859, subsample=0.9329770563201687, verbose=-1)

os.makedirs("results", exist_ok=True)
df = pd.read_csv(CSV_PATH)
feats = [c for c in df.columns if c.startswith(("H.", "DD.", "UD."))]
train = df[df.sessionIndex <= 5]     # sessions 1-5 are in train
test = df[df.sessionIndex > 5]             # sesssions 6,7,8 are in test
X_tr, y_tr = train[feats].values, train["subject"].values
X_te, y_te = test[feats].values, test["subject"].values
sess_te = test["sessionIndex"].values
rng = np.random.default_rng(SEED)


def augment(X, y, scale_range, copies):
    """Return the original rows plus `copies` randomly sped-up / slowed-down versions.
    One random factor per attempt multiplies all 31 timings, so DD = H + UD still holds."""

    # Baseline
    if scale_range == 0.0:
        return X, y

    Xs, ys = [X], [y]
    for _ in range(copies):
        s = rng.uniform(1 - scale_range, 1 + scale_range, size=(len(X), 1))
        Xs.append(X * s + rng.normal(0, JITTER, X.shape))
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)

# Returns a pipeline with already tuned conf
def make(algo):
    if algo == "kNN":
        clf = KNeighborsClassifier(n_jobs=-1, **KNN_conf)
    elif algo == "SVM":
        SVC(kernel="rbf", **SVM_conf)
    elif algo == "RF":
        clf = RandomForestClassifier(n_jobs=-1, random_state=SEED, **RandomForest_conf)
    elif algo == "XGB":
        clf = XGBClassifier(n_jobs=-1, random_state=SEED, **XGBoost_conf)
    else:
        clf = LGBMClassifier(n_jobs=-1,random_state=SEED,  **LGBM_conf)

    if algo in ["kNN", "SVM"]:
        return Pipeline([("scale", PowerTransformer(method="yeo-johnson")), ("model", clf)])
    else:
        return Pipeline([("scale", "passthrough"), ("model", clf)])


rows = []
for name in ["kNN"] + (["SVM"] if USE_SVM else []):
    for r in SCALE_RANGES:
        t0 = time.time()
        Xa, ya = augment(X_tr, y_tr, r, COPIES)
        # fit on the (augmented) training rows only
        pred = make(name).fit(Xa, ya).predict(X_te)
        row = {
            "model": name,
            "scale_range": f"+/-{r:.2f}" if r else "none",
            "train_rows": len(Xa),
            "test_accuracy": round(accuracy_score(y_te, pred), 4),
            "test_macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
            "time_taken": round((time.time() - t0) / 60, 1),
        }
        for s in sorted(np.unique(sess_te)):
            m = sess_te == s
            row[f"acc_session{s}"] = round(accuracy_score(y_te[m], pred[m]), 4)
        rows.append(row)
        print(row)

res = pd.DataFrame(rows)
res.to_csv("results/augment_results.csv", index=False)
print("\n", res.to_string(index=False))