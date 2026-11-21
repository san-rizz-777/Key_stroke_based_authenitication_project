"""
Per-person feature correlation heatmaps.
Each heatmap uses ONE person's attempts only, so it shows how that person's
own timings move together (not differences between people).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else "data/lab_dataset.csv"
N_USERS = 4          # how many users to sample
SEED = 7             # change to get a different sample
USERS = None         # or set a list, e.g. ["s002", "s003", "s004", "s005"]

os.makedirs("results", exist_ok=True)
df = pd.read_csv(CSV_PATH)
H = [c for c in df.columns if c.startswith("H.")]
DD = [c for c in df.columns if c.startswith("DD.")]
UD = [c for c in df.columns if c.startswith("UD.")]
feats = H + DD + UD

if USERS is None:
    rng = np.random.default_rng(SEED)
    USERS = sorted(rng.choice(df["subject"].unique(), size=N_USERS, replace=False))
print("Users:", USERS)


def block_means(corr):
    """Average correlation inside the main blocks (diagonal excluded for H-H)."""
    h, d, u = len(H), len(DD), len(UD)
    hh = corr.iloc[:h, :h].values[~np.eye(h, dtype=bool)].mean()
    hu = corr.iloc[:h, h + d:].values.mean()
    # DD[i] and UD[i] describe the same key pair, so compare them directly
    dd_ud_pairs = np.mean([corr.loc[a, b] for a, b in zip(DD, UD)])
    return hh, hu, dd_ud_pairs


# ---- 1. One heatmap per sampled user ---------------------------------------
cols = 2
rows = int(np.ceil(len(USERS) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows), squeeze=False)
print("\nuser   mean H-H   mean H-UD   mean DD-UD (matching pairs)")
for ax, u in zip(axes.ravel(), USERS):
    corr = df.loc[df.subject == u, feats].corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(f"{u}  (n={int((df.subject == u).sum())} attempts)")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=5)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    hh, hu, du = block_means(corr)
    print(f"{u}   {hh:8.2f}   {hu:9.2f}   {du:10.2f}")
for ax in axes.ravel()[len(USERS):]:
    ax.axis("off")
fig.colorbar(im, ax=axes, label="Correlation (-1 to 1)", shrink=0.8)
plt.savefig("results/person_corr.png", dpi=120, bbox_inches="tight")
plt.show()

# ---- 2. All users pooled, but each user's own mean removed -----------------
# Removes "this person is slower/faster overall", leaving only attempt-to-attempt wobble.
within = df[feats] - df.groupby("subject")[feats].transform("mean")
corr_w = within.corr()
plt.figure(figsize=(8, 7))
plt.imshow(corr_w, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation (-1 to 1)")
plt.xticks(range(len(feats)), feats, rotation=90, fontsize=6)
plt.yticks(range(len(feats)), feats, fontsize=6)
plt.xlabel("Feature")
plt.ylabel("Feature")
plt.title("Within-person correlation (each user's mean removed, all users pooled)")
plt.tight_layout()
plt.savefig("results/within_person_corr.png", dpi=120)
plt.show()
hh, hu, du = block_means(corr_w)
print(f"\nWithin-person, all users: H-H {hh:.2f}, H-UD {hu:.2f}, DD-UD pairs {du:.2f}")