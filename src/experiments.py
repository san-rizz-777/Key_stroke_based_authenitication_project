"""
Experiment grid for keystroke identification.

Compares feature sets x preprocessing x models, all with the same split:
train on sessions 1-5, test on sessions 6-8. Reports accuracy per test session,
which doubles as a drift check (does accuracy fall in later sessions?).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else "data/lab_dataset.csv"
TRAIN_FRAC = 1.0                       # e.g. 0.5 to train on half the rows for a fast first pass
MODELS = ["kNN", "SVM", "Random Forest", "XGBoost", "LightGBM"]
GBM_ROUNDS = 300                       # boosting rounds; with 126 classes each round builds 126 trees, so lower this for a quick pass
TREE_MODELS = {"Random Forest", "XGBoost", "LightGBM"}   # scale-invariant, so Yeo-Johnson is skipped for them
SEED = 42

if XGBClassifier is None and "XGBoost" in MODELS:
    print("xgboost not installed (pip install xgboost), skipping it"); MODELS.remove("XGBoost")
if LGBMClassifier is None and "LightGBM" in MODELS:
    print("lightgbm not installed (pip install lightgbm), skipping it"); MODELS.remove("LightGBM")

os.makedirs("results", exist_ok=True)
df = pd.read_csv(CSV_PATH)
H = [c for c in df.columns if c.startswith("H.")]
DD = [c for c in df.columns if c.startswith("DD.")]
UD = [c for c in df.columns if c.startswith("UD.")]

FEATURE_SETS = {"all31": H + DD + UD, "H+UD": H + UD, "H only": H, "UD only": UD}
PREPROCESSING = ["standard", "yeo-johnson", "speed-norm"]


def build_X(d, cols, speed_norm):
    """Speed-norm divides every gap (DD, UD) by that attempt's own mean DD.
    It uses only the row itself, so nothing leaks from the test sessions."""
    X = d[cols].copy()
    if speed_norm:
        tempo = d[DD].mean(axis=1).clip(lower=1e-3)
        gap_cols = [c for c in cols if c.startswith(("DD.", "UD."))]
        if gap_cols:
            X[gap_cols] = X[gap_cols].div(tempo, axis=0)
    return X.values


def make_model(name, prep):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=GBM_ROUNDS, learning_rate=0.1, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=GBM_ROUNDS, learning_rate=0.1, num_leaves=31, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1)
    scaler = PowerTransformer(method="yeo-johnson") if prep == "yeo-johnson" else StandardScaler()
    if name == "kNN":
        return make_pipeline(scaler, KNeighborsClassifier(n_neighbors=5, weights="distance", p=1, n_jobs=-1))
    return make_pipeline(scaler, SVC(kernel="rbf", C=10, gamma="scale"))


train = df[df.sessionIndex <= 5]
if TRAIN_FRAC < 1.0:
    train = train.sample(frac=TRAIN_FRAC, random_state=SEED)
test = df[df.sessionIndex > 5]
le = LabelEncoder().fit(df["subject"])            # boosting libraries need integer class labels
y_tr, y_te = le.transform(train["subject"]), le.transform(test["subject"])
test_sessions = sorted(test.sessionIndex.unique())
print(f"Train rows: {len(train)}, test rows: {len(test)}, users: {df.subject.nunique()}")

rows = []
for fs_name, cols in FEATURE_SETS.items():
    for prep in PREPROCESSING:
        X_tr = build_X(train, cols, prep == "speed-norm")
        X_te = build_X(test, cols, prep == "speed-norm")
        for name in MODELS:
            if name in TREE_MODELS and prep == "yeo-johnson":
                continue                     # trees ignore monotonic transforms, so skip
            t0 = time.time()
            model = make_model(name, prep).fit(X_tr, y_tr)
            pred = model.predict(X_te)
            row = {
                "feature_set": fs_name, "preprocessing": prep, "model": name,
                "accuracy": round(accuracy_score(y_te, pred), 4),
                "macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
                "minutes": round((time.time() - t0) / 60, 2),
            }
            for s in test_sessions:
                m = (test.sessionIndex.values == s)
                row[f"acc_session{s}"] = round(accuracy_score(y_te[m], pred[m]), 4)
            rows.append(row)
            print(row)

res = pd.DataFrame(rows)
res.to_csv("results/experiments.csv", index=False)

print("\n=== Accuracy: rows = feature set + preprocessing, columns = model ===")
print(res.pivot_table(index=["feature_set", "preprocessing"], columns="model", values="accuracy").round(3).to_string())
best = res.loc[res.accuracy.idxmax()]
print("\nBest overall:", best[["feature_set", "preprocessing", "model", "accuracy"]].to_dict())
print("Saved results/experiments.csv")