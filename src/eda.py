"""
EDA for the keystroke dataset.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("results", exist_ok=True)
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else "data/lab_dataset.csv"
df = pd.read_csv(CSV_PATH)

H = [c for c in df.columns if c.startswith("H.")]
DD = [c for c in df.columns if c.startswith("DD.")]
UD = [c for c in df.columns if c.startswith("UD.")]
feats = H + DD + UD
print(f"Rows: {len(df)}, users: {df['subject'].nunique()}, features: {len(feats)} (H={len(H)}, DD={len(DD)}, UD={len(UD)})")

# 1. Class balance
counts = df.groupby("subject").size()
print("Rows per user: min", counts.min(), "max", counts.max())

# 2. Basic sanity: missing values, ranges, negatives
print("Missing values in features:", int(df[feats].isna().sum().sum()))
print("\nFeature summary (seconds):")
print(df[feats].describe().T[["mean", "std", "min", "max"]].round(4).to_string())
print(f"\nShare of negative UD values: {(df[UD] < 0).values.mean():.3%} (negative = next key pressed before release)")

# 3. Redundancy check: DD should equal H(previous key) + UD
diffs = []
# H, DD and UD columns are in key order, so DD[i] and UD[i] belong to key H[i]
for h, d, u in zip(H[:-1], DD, UD):
    diffs.append(np.abs(df[d] - (df[h] + df[u])).max())
print("Max |DD - (H + UD)| across rows:", round(max(diffs), 6), "(near 0 means DD is redundant)")

# 4. Outliers: rows with very large values
big = (df[feats] > 5).any(axis=1).sum()
print("Rows with any timing > 5 s:", int(big))

# 5. Hold-time histograms for 4 users
users = sorted(df["subject"].unique())[:4]
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for u in users:
    axes[0].hist(df.loc[df.subject == u, H[1]], bins=40, alpha=0.5, label=u)
    axes[1].hist(df.loc[df.subject == u, DD[0]], bins=40, alpha=0.5, label=u)
    axes[2].hist(df.loc[df.subject == u, UD[0]], bins=40, alpha=0.5, label=u)
axes[0].set_title(f"Hold time: {H[1]}"); axes[1].set_title(f"Down-down time: {DD[0]}"); axes[2].set_title(f"Up-down time: {UD[0]}")
for ax in axes:
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Number of typing attempts")
axes[0].legend(); plt.tight_layout(); plt.savefig("results/eda_hist.png", dpi=120); plt.show()

# 6. Correlation heatmap
plt.figure(figsize=(9, 8))
plt.imshow(df[feats].corr(), cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation (-1 to 1)")
plt.xticks(range(len(feats)), feats, rotation=90, fontsize=6)
plt.yticks(range(len(feats)), feats, fontsize=6)
plt.xlabel("Feature"); plt.ylabel("Feature")
plt.title("Feature correlation (features listed in order: H, DD, UD)")
plt.tight_layout(); plt.savefig("results/eda_corr.png", dpi=120); plt.show()

# 7. How separable are users? PCA on a few users
sub = df[df.subject.isin(sorted(df.subject.unique())[:8])]
pca = PCA(n_components=2)
Z = pca.fit_transform(StandardScaler().fit_transform(sub[feats]))
var = pca.explained_variance_ratio_ * 100
plt.figure(figsize=(6, 5))
for u in sub.subject.unique():
    m = (sub.subject == u).values
    plt.scatter(Z[m, 0], Z[m, 1], s=5, label=u)
plt.legend(markerscale=3, title="User")
plt.xlabel(f"PC1 ({var[0]:.1f}% of variance)")
plt.ylabel(f"PC2 ({var[1]:.1f}% of variance)")
plt.title("PCA of 8 users (each dot = one typing attempt)")
plt.tight_layout(); plt.savefig("results/eda_pca.png", dpi=120); plt.show()

# 8. Session drift: train on session 1 only, test on each later session
s1 = df[df.sessionIndex == 1]
clf = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(s1[feats], s1["subject"])
print("\nDrift check (kNN trained on session 1 only):")
for s in sorted(df.sessionIndex.unique())[1:]:
    t = df[df.sessionIndex == s]
    print(f"  session {s}: accuracy {clf.score(t[feats], t['subject']):.3f}")

print("\nSaved plots in results/ (eda_hist, eda_corr, eda_pca).")