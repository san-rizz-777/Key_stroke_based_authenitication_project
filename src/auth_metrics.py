"""
auth_metrics.py — verification (authentication) metrics from any closed-set classifier.

Your models answer "which of the 126 users typed this?". Authentication asks a different
question: "is this person who they claim to be?". We convert one into the other like this:

    For every test sample (true user u) and every claimed identity c:
        score(sample, c) = the model's score for class c
        c == u  -> GENUINE attempt   (user claims to be themselves)
        c != u  -> IMPOSTOR attempt  (user claims to be somebody else)

    Accept the claim if score >= threshold.
        FAR = fraction of impostor attempts accepted   (security risk)
        FRR = fraction of genuine attempts rejected    (annoyance)
        EER = the point where FAR == FRR               (threshold-free summary, lower is better)

`P` below is an (N, C) score matrix: softmax probabilities for the GRU, predict_proba for
sklearn / XGBoost / LightGBM, or decision_function scores for SVM. Only the ranking of
scores matters for ROC/EER, so uncalibrated scores are fine.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc


# ------------------------------------------------------------
# Trials
# ------------------------------------------------------------
def make_trials(P, y, max_impostor_trials=None, seed=0):
    """Build genuine and impostor score arrays from an (N, C) score matrix."""
    n = P.shape[0]
    idx = np.arange(n)
    gen = P[idx, y]                                  # score of the true class
    mask = np.ones(P.shape, dtype=bool)
    mask[idx, y] = False
    imp = P[mask]                                    # N*(C-1) scores of wrong classes
    if max_impostor_trials and imp.size > max_impostor_trials:
        rng = np.random.default_rng(seed)            # subsample to keep ROC fast on big sets
        imp = rng.choice(imp, max_impostor_trials, replace=False)
    return gen, imp


# ------------------------------------------------------------
# Core curve + point metrics
# ------------------------------------------------------------
def det_points(gen, imp):
    """FAR, FRR and thresholds for every operating point (accept if score >= threshold)."""
    labels = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
    scores = np.r_[gen, imp]
    far, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    return far, 1.0 - tpr, thr


def eer(gen, imp):
    """Equal error rate and the threshold where it occurs."""
    far, frr, thr = det_points(gen, imp)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thr[i])


def frr_at_far(gen, imp, far_target):
    """Lowest FRR achievable while keeping FAR <= far_target (e.g. 0.01 = 1%)."""
    far, frr, _ = det_points(gen, imp)
    ok = far <= far_target                           # far is non-decreasing along the curve
    return float(frr[ok][-1])


def rates_at_threshold(gen, imp, thr):
    """(FAR, FRR) at a fixed threshold."""
    return float((imp >= thr).mean()), float((gen < thr).mean())


def topk_accuracy(P, y, ks=(1, 3, 5)):
    """Identification top-k accuracy (is the true user among the k highest-scoring users?)."""
    order = np.argsort(-P, axis=1)[:, :max(ks)]
    return {f"top{k}_acc": float((order[:, :k] == y[:, None]).any(axis=1).mean()) for k in ks}


# ------------------------------------------------------------
# Per-user view: each user's "detector" is column u of the score matrix
# ------------------------------------------------------------
def per_user_eer(P, y, min_genuine=5):
    vals = []
    for u in np.unique(y):
        g = P[y == u, u]                             # user u typing, claiming to be u
        im = P[y != u, u]                            # everybody else claiming to be u
        if len(g) >= min_genuine and len(im) > 0:
            vals.append(eer(g, im)[0])
    return np.array(vals)


# ------------------------------------------------------------
# Reports
# ------------------------------------------------------------
def auth_report(P, y, sessions=None, session_ids=(6, 7, 8),
                max_impostor_trials=3_000_000, seed=0):
    """Threshold-free authentication metrics on a test set (use the FINAL model's scores)."""
    gen, imp = make_trials(P, y, max_impostor_trials, seed)
    far, frr, _ = det_points(gen, imp)
    i = int(np.argmin(np.abs(far - frr)))

    out = {
        "auth_eer": float((far[i] + frr[i]) / 2),
        "auth_auc": float(auc(far, 1.0 - frr)),
        "auth_frr_at_far1pct":  float(frr[far <= 0.01][-1]),
        "auth_frr_at_far01pct": float(frr[far <= 0.001][-1]),
        **{f"auth_{k}": v for k, v in topk_accuracy(P, y, ks=(3, 5)).items()},
    }

    # EER per session shows how verification quality drifts as time passes
    if sessions is not None:
        for s in session_ids:
            m = sessions == s
            if m.any():
                g, im = make_trials(P[m], y[m], max_impostor_trials, seed)
                out[f"auth_eer_session{s}"] = eer(g, im)[0]

    # Spread across users: a good average can hide a few users the system fails on
    ue = per_user_eer(P, y)
    if ue.size:
        out["auth_user_eer_mean"] = float(ue.mean())
        out["auth_user_eer_median"] = float(np.median(ue))
        out["auth_user_eer_worst"] = float(ue.max())
    return out


def operating_point(P_cal, y_cal, P_test, y_test, max_impostor_trials=3_000_000, seed=0):
    """Deployment-realistic numbers: pick the threshold on a calibration set (session 5),
    then measure FAR/FRR on later sessions with that FIXED threshold.
    Use the same model for P_cal and P_test so the score scale is consistent."""
    g, i = make_trials(P_cal, y_cal, max_impostor_trials, seed)
    cal_eer, thr = eer(g, i)
    gt, it = make_trials(P_test, y_test, max_impostor_trials, seed)
    far, frr = rates_at_threshold(gt, it, thr)
    return {"op_cal_eer": cal_eer, "op_threshold": thr,
            "op_far": far, "op_frr": frr, "op_hter": (far + frr) / 2}


# ------------------------------------------------------------
# Plot: DET curves (FAR vs FRR on a normal-deviate scale; straighter and more readable than ROC)
# ------------------------------------------------------------
def plot_det(trials, path="det_curves.png", n_points=2000):
    """trials: {model_name: (gen, imp)} from make_trials()."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (g, im) in trials.items():
        far, frr, _ = det_points(g, im)
        sel = np.unique(np.linspace(0, len(far) - 1, n_points).astype(int))   # thin out millions of points
        far, frr = far[sel], frr[sel]
        ok = (far > 0) & (far < 1) & (frr > 0) & (frr < 1)
        ax.plot(norm.ppf(far[ok]), norm.ppf(frr[ok]), label=name)

    ticks = [0.001, 0.01, 0.05, 0.2, 0.5]
    ax.set_xticks(norm.ppf(ticks)); ax.set_xticklabels([f"{t:.1%}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks)); ax.set_yticklabels([f"{t:.1%}" for t in ticks])
    ax.plot(norm.ppf([0.001, 0.5]), norm.ppf([0.001, 0.5]), "k:", lw=0.8)       # EER diagonal
    ax.set_xlabel("False accept rate (FAR)"); ax.set_ylabel("False reject rate (FRR)")
    ax.set_title("DET curves (closer to the origin is better)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path