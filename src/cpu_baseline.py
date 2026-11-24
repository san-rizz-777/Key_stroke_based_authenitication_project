"""
cpu_baselines.py — classical baselines for keystroke dynamics, same protocol as the GRU suite.

Protocol (identical to rnn_experiments_fixed.py so results are directly comparable):
  1. Train on sessions 1-4, validate on session 5   -> val_acc
  2. Refit a fresh model on sessions 1-5            -> final model
  3. Test on sessions 6+ (overall + per-session S6/S7/S8)

Preprocessing is also identical: Yeo-Johnson + StandardScaler, fit on sessions 1-5 only.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", force=True,
                    handlers=[logging.FileHandler("baselines.log", mode="a"),
                              logging.StreamHandler()])
log = logging.info

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "data/lab_dataset.csv"
SEEDS    = [0, 1, 2]       # only used by randomized models (RF / XGBoost / LightGBM)
N_JOBS   = -1              # use all CPU cores

# ------------------------------------------------------------
# Tuned hyperparameters (from best_params.json; the "model__" pipeline prefix is removed).
# ------------------------------------------------------------
BEST_PARAMS = {
    # p=1 -> Manhattan distance
    "kNN": {"n_neighbors": 7, "weights": "distance", "p": 1},

    # kernel was not tuned, so sklearn's default (rbf) applies; gamma only matters for rbf
    "SVM": {"C": 1.4936568554617629, "gamma": 0.0396760507705299, "kernel": "rbf"},

    "Random Forest": {"n_estimators": 400, "max_depth": 20, "min_samples_leaf": 1,
                      "max_features": "sqrt"},

    "XGBoost": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.06049716507542575,
                "min_child_weight": 3, "subsample": 0.8439986631130484,
                "colsample_bytree": 0.9233589392465844, "reg_lambda": 3.8718305057994216},

    # NOTE: subsample has no effect in LightGBM unless subsample_freq > 0. It was not set
    # during tuning, so it is left unset here to reproduce the tuned results.
    "LightGBM": {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.22044355215811345,
                 "min_child_samples": 40, "subsample": 0.9329770563201687,
                 "colsample_bytree": 0.6232334448672797, "reg_lambda": 8.706020878304859},
}

# Models without randomness give the same result for every seed, so run them once
DETERMINISTIC = {"kNN", "SVM"}


# ============================================================
# DATA (same as the GRU script)
# ============================================================
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    drop_cols = ["session", "time", "errors", "psr", "totaltime"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    feature_cols = [c for c in df.columns if c not in ("subject", "sessionIndex")]
    X = df[feature_cols].values.astype(np.float32, copy=True)
    y = df["subject"].values
    sessions = df["sessionIndex"].values

    # Fit scalers on sessions 1-5 only (no test leakage)
    train_mask = sessions <= 5
    pt = PowerTransformer(method="yeo-johnson")
    X[train_mask]  = pt.fit_transform(X[train_mask]).astype(np.float32)
    X[~train_mask] = pt.transform(X[~train_mask]).astype(np.float32)
    scaler = StandardScaler()
    X[train_mask]  = scaler.fit_transform(X[train_mask]).astype(np.float32)
    X[~train_mask] = scaler.transform(X[~train_mask]).astype(np.float32)

    label_map = {s: i for i, s in enumerate(np.unique(y))}
    y_enc = np.array([label_map[s] for s in y], dtype=np.int64)
    return X, y_enc, sessions


# ============================================================
# MODEL FACTORY
# ============================================================
def make_model(name, seed):
    p = dict(BEST_PARAMS[name])
    if name == "kNN":
        return KNeighborsClassifier(n_jobs=N_JOBS, **p)
    if name == "SVM":
        return SVC(**p)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=seed, n_jobs=N_JOBS, **p)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        # tree_method="hist" is the fast CPU method; labels must be 0..K-1 (they are)
        return XGBClassifier(tree_method="hist", device="cpu", n_jobs=N_JOBS,
                             random_state=seed, verbosity=0, **p)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=N_JOBS, verbose=-1, **p)
    raise ValueError(name)


# ============================================================
# ONE RUN
# ============================================================
def run_one(name, X, y, sessions, seed):
    t0 = time.time()

    tr, va, te = sessions <= 4, sessions == 5, sessions >= 6

    # Step 1: train on S1-4, validate on S5
    m = make_model(name, seed)
    m.fit(X[tr], y[tr])
    val_acc = accuracy_score(y[va], m.predict(X[va]))

    # Step 2: refit a fresh model on S1-5 (same as the GRU "phase 2")
    m = make_model(name, seed)
    full = sessions <= 5
    m.fit(X[full], y[full])

    # Step 3: evaluate on held-out sessions
    pred = m.predict(X[te])
    res = {
        "name": name, "seed": seed, "val_acc": val_acc,
        "test_acc": accuracy_score(y[te], pred),
        "test_f1":  f1_score(y[te], pred, average="macro"),
    }
    for s in (6, 7, 8):
        idx = sessions == s
        res[f"acc_session{s}"] = accuracy_score(y[idx], m.predict(X[idx]))
    res["minutes"] = (time.time() - t0) / 60
    return res


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    X, y, sessions = load_and_preprocess(CSV_PATH)
    log(f"data: X={X.shape}, classes={len(np.unique(y))}")

    results = []
    for name in BEST_PARAMS:
        seeds = [0] if name in DETERMINISTIC else SEEDS
        for seed in seeds:
            log(f">>> {name} | seed={seed}")
            r = run_one(name, X, y, sessions, seed)
            results.append(r)
            log(f"    val={r['val_acc']:.4f} test={r['test_acc']:.4f} "
                f"f1={r['test_f1']:.4f} S6/7/8="
                f"{r['acc_session6']:.3f}/{r['acc_session7']:.3f}/{r['acc_session8']:.3f} "
                f"time={r['minutes']:.2f}m")
            # save after every run so nothing is lost on a crash
            pd.DataFrame(results).to_csv("baseline_results_partial.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv("results/cpu_baseline_results.csv", index=False)

    agg = df.groupby("name").agg({
        "val_acc": "mean",
        "test_acc": ["mean", "std"], "test_f1": ["mean", "std"],
        "acc_session6": "mean", "acc_session7": "mean", "acc_session8": "mean",
        "minutes": "mean",
    }).round(4)
    print("\n=== BASELINES (mean ± std over seeds; kNN/SVM are deterministic) ===")
    print(agg.sort_values(("test_acc", "mean"), ascending=False).to_string())