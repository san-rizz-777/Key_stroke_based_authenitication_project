import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import time
import warnings

warnings.filterwarnings('ignore')

# CONFIG
CSV_PATH = 'data/lab_dataset.csv'
FEATURE_COLS = [c for c in pd.read_csv(CSV_PATH, nrows=0).columns
                if c not in ['subject', 'sessionIndex', 'session', 'time', 'errors', 'psr', 'totaltime']]


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    drop_cols = ['session', 'time', 'errors', 'psr', 'totaltime']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = df[FEATURE_COLS].values
    y = df['subject'].values
    sessions = df['sessionIndex'].values

    unique_subjects = np.unique(y)
    label_map = {s: i for i, s in enumerate(unique_subjects)}
    y_encoded = np.array([label_map[s] for s in y])

    return X, y_encoded, sessions


def apply_speed_norm(X, sessions):
    """Divide all gaps by the attempt's mean DD."""
    # Assuming DD columns are indices 11-20
    DD_COLS = list(range(11, 21))
    # Calculate mean DD per attempt
    mean_dd = X[:, DD_COLS].mean(axis=1, keepdims=True)
    # Avoid division by zero
    mean_dd[mean_dd == 0] = 1e-6
    # Divide all features by mean_dd (or just gaps? The handoff says "dividing gaps by an attempt's mean DD")
    # Let's divide all features by mean_dd
    X_norm = X / mean_dd
    return X_norm


def preprocess(X, sessions, use_yeo=True, use_speed_norm=False):
    train_mask = sessions <= 5
    X_processed = X.copy()

    if use_speed_norm:
        X_processed = apply_speed_norm(X_processed, sessions)

    if use_yeo:
        pt = PowerTransformer(method='yeo-johnson')
        X_processed[train_mask] = pt.fit_transform(X_processed[train_mask])
        X_processed[~train_mask] = pt.transform(X_processed[~train_mask])

    scaler = StandardScaler()
    X_processed[train_mask] = scaler.fit_transform(X_processed[train_mask])
    X_processed[~train_mask] = scaler.transform(X_processed[~train_mask])

    return X_processed


def run_classical_experiments():
    X, y, sessions = load_data(CSV_PATH)

    train_mask = sessions <= 5
    test_mask = sessions >= 6

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    sessions_train = sessions[train_mask]
    sessions_test = sessions[test_mask]

    # Define preprocessing variants
    preprocess_configs = [
        {'name': 'standard', 'use_yeo': False, 'use_speed_norm': False},
        {'name': 'yeo', 'use_yeo': True, 'use_speed_norm': False},
        {'name': 'speed', 'use_yeo': False, 'use_speed_norm': True},
        {'name': 'speed+yeo', 'use_yeo': True, 'use_speed_norm': True},
    ]

    # Define models
    models = {
        'kNN': KNeighborsClassifier(n_neighbors=7, weights='distance', p=1),
        'SVM': SVC(C=1.4936, gamma=0.0396, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1),
        'LDA': LinearDiscriminantAnalysis(),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'GaussianNB': GaussianNB(),
    }

    results = []

    for pcfg in preprocess_configs:
        print(f"\n=== Preprocessing: {pcfg['name']} ===")
        # Apply preprocessing to the full dataset
        X_processed = preprocess(X, sessions, use_yeo=pcfg['use_yeo'], use_speed_norm=pcfg['use_speed_norm'])
        X_train_p = X_processed[train_mask]
        X_test_p = X_processed[test_mask]

        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            start = time.time()
            model.fit(X_train_p, y_train)
            train_time = time.time() - start

            y_pred = model.predict(X_test_p)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            # Per-session accuracy
            session_accs = {}
            for s in [6, 7, 8]:
                idx = sessions_test == s
                if idx.sum() > 0:
                    session_accs[f'S{s}'] = accuracy_score(y_test[idx], y_pred[idx])
                else:
                    session_accs[f'S{s}'] = np.nan

            results.append({
                'preprocessing': pcfg['name'],
                'model': model_name,
                'test_acc': acc,
                'test_f1': f1,
                'train_time_min': train_time / 60,
                **session_accs
            })
            print(f"    Acc: {acc:.4f} | F1: {f1:.4f} | Time: {train_time:.1f}s")

    df_results = pd.DataFrame(results)
    df_results.to_csv('classical_ablation_results.csv', index=False)
    print("\nClassical experiments complete. Saved to classical_ablation_results.csv")
    print(df_results.to_string())


if __name__ == '__main__':
    run_classical_experiments()