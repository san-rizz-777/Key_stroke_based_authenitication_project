"""
Keystroke dynamics: identification baselines (CMU benchmark)

Dataset: DSL-StrongPasswordData.csv
  columns: subject, sessionIndex, rep, then 31 timing features (H.*, DD.*, UD.*)
  51 subjects x 8 sessions x 50 reps = 20,400 rows
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

path = sys.argv[1] if len(sys.argv) > 1 else "data/lab_dataset.csv"
df = pd.read_csv(path)

# ---- 1. Quick look ---------------------------------------------------------
print("Shape:", df.shape)
print("Users:", df["subject"].nunique())
print("Sessions:", sorted(df["sessionIndex"].unique()))
print("Missing values:", int(df.isna().sum().sum()))

feature_cols = [c for c in df.columns if c.startswith(("H.", "DD.", "UD."))]
print("Number of timing features:", len(feature_cols))

# ---- 2. Session-based split (NOT random) -----------------------------------
# Train on sessions 1-5, test on sessions 6-8. Later sessions are typed on
# different days, so this mimics real use: enroll once, authenticate later.
train = df[df["sessionIndex"] <= 5]
test = df[df["sessionIndex"] > 5]

X_train, y_train = train[feature_cols].values, train["subject"].values
X_test, y_test = test[feature_cols].values, test["subject"].values
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---- 3. Baselines -----------------------------------------------------------
# Scaler lives inside the pipeline so it is fit on training data only (no leakage).
models = {
    "kNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
    "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    results.append({"model": name, "accuracy": acc, "macro_f1": f1})
    print(f"{name:15s} accuracy={acc:.4f}  macro-F1={f1:.4f}")

# ---- 4. Save results table for the README ----------------------------------
pd.DataFrame(results).to_csv("results/baseline_results.csv", index=False)
print("Saved baseline_results.csv")