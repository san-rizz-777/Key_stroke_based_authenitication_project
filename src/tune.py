"""
Hyperparameter tuning only: kNN, SVM, Random Forest, XGBoost, LightGBM.

Fixed from your experiments (not searched here):
  * features: all 31 timing columns
  * kNN and SVM: Yeo-Johnson transform (it beat plain scaling by 3-5 points)
  * tree models: no scaling needed

Protocol (no leakage):
  * Settings are chosen using sessions 1-5 only, with forward-chaining validation:
        train on sessions 1-3 -> validate on 4
        train on sessions 1-4 -> validate on 5
  * Sessions 6-8 are used ONCE per model, at the end, with the best settings.
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# custom configuration
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else "data/lab_dataset.csv"
MODELS = ["kNN", "SVM", "Random Forest", "XGBoost", "LightGBM"]
QUICK = False                     # True = small, fast first pass
TUNE_FRAC = 0.5                   # fraction of sessions 1-5 rows used during the search
N_ITER = {"kNN": 15, "SVM": 12, "Random Forest": 10, "XGBoost": 10, "LightGBM": 10}              # can be changed
SEED = 42
# for fast first check tune only on kNN
if QUICK:
    TUNE_FRAC = 0.25
    N_ITER = {k: 4 for k in N_ITER}

os.makedirs("results", exist_ok=True)
df = pd.read_csv(CSV_PATH)
feats = [c for c in df.columns if c.startswith(("H.", "DD.", "UD."))]


def build(name):
    """Return (pipeline, search space) for one model."""
    if name == "kNN":
        scale, model = PowerTransformer(method="yeo-johnson"), KNeighborsClassifier(n_jobs=-1)
        space = {
            "model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],                       # 1 = Manhattan, 2 = Euclidean
        }
    elif name == "SVM":
        scale, model = PowerTransformer(method="yeo-johnson"), SVC(kernel="rbf")
        space = {"model__C": loguniform(1, 1000), "model__gamma": loguniform(1e-4, 1e-1)}
    elif name == "Random Forest":
        scale, model = "passthrough", RandomForestClassifier(n_jobs=-1, random_state=SEED)
        space = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 20, 40],
            "model__max_features": ["sqrt", 0.3, 0.5],
            "model__min_samples_leaf": [1, 2, 4],
        }
    elif name == "XGBoost":
        scale, model = "passthrough", XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED)
        space = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": loguniform(0.03, 0.3),
            "model__max_depth": [3, 4, 6, 8],
            "model__subsample": uniform(0.6, 0.4),            # 0.6 to 1.0
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": [1, 3, 5],
            "model__reg_lambda": loguniform(0.5, 10),
        }
    else:  # LightGBM
        scale = "passthrough"
        model = LGBMClassifier(subsample_freq=1, n_jobs=-1, random_state=SEED, verbose=-1)
        space = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": loguniform(0.03, 0.3),
            "model__num_leaves": [15, 31, 63],
            "model__min_child_samples": [10, 20, 40],
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_lambda": loguniform(0.1, 10),
        }
    return Pipeline([("scale", scale), ("model", model)]), space


def clean(v):
    """Make a parameter value JSON-friendly."""
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


#  data and splits
le = LabelEncoder().fit(df["subject"])                 # boosting libraries need integer labels
train = df[df.sessionIndex <= 5].reset_index(drop=True)
test = df[df.sessionIndex > 5].reset_index(drop=True)
X_tr, y_tr = train[feats].values, le.transform(train["subject"])
X_te, y_te = test[feats].values, le.transform(test["subject"])
sess_te = test["sessionIndex"].values

# Split is based on the session
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(train), size=int(len(train) * TUNE_FRAC), replace=False))
X_s, y_s, s_s = X_tr[idx], y_tr[idx], train["sessionIndex"].values[idx]
folds = [(np.where(s_s < v)[0], np.where(s_s == v)[0]) for v in (4, 5)]   # train on the past, validate on a later session
print(f"Users: {len(le.classes_)} | train rows: {len(train)} | search rows: {len(idx)} | test rows: {len(test)}")

# tuning
rows, best_params = [], {}
for name in MODELS:
    t0 = time.time()
    pipe, space = build(name)
    search = RandomizedSearchCV(
        pipe, space, n_iter=N_ITER[name], cv=folds, scoring="accuracy",
        random_state=SEED, n_jobs=1, refit=False, error_score="raise",
    )
    search.fit(X_s, y_s)
    best = search.best_params_
    best_params[name] = {k: clean(v) for k, v in best.items()}

    # refit with the best settings on ALL of sessions 1-5, then test once on sessions 6-8
    final = clone(pipe).set_params(**best).fit(X_tr, y_tr)
    pred = final.predict(X_te)
    row = {
        "model": name,
        "val_accuracy": round(float(search.best_score_), 4),     # sessions 4-5, used for choosing
        "test_accuracy": round(accuracy_score(y_te, pred), 4),   # sessions 6-8, used once
        "test_macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    for s in sorted(np.unique(sess_te)):
        m = sess_te == s
        row[f"acc_session{s}"] = round(accuracy_score(y_te[m], pred[m]), 4)
    rows.append(row)
    print(row, "\n   best:", best_params[name], "\n")


# Output
res = pd.DataFrame(rows)
res.to_csv("results/tuned_results.csv", index=False)
with open("results/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print(res.to_string(index=False))
print("\nSaved results/tuned_results.csv and results/best_params.json")