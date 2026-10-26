import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def fold_metrics(labels, probs, threshold=0.5):
    y = np.asarray(labels)
    p = np.asarray(probs)
    pred = (p >= threshold).astype(int)
    return dict(accuracy=(pred == y).mean(),
                sensitivity=(pred[y == 1] == 1).mean(),
                specificity=(pred[y == 0] == 0).mean(),
                mcc=matthews_corrcoef(y, pred),
                roc_auc=roc_auc_score(y, p))


def per_fold(preds, threshold=0.5):
    """Recompute the fold table from a predictions file."""
    rows = [dict(fold=k, **fold_metrics(g.label, g.probability, threshold))
            for k, g in preds.groupby("fold")]
    return pd.DataFrame(rows)


# Intervals are 95% percentile intervals over 10,000 resamples of folds
# A fold is a participant under P1, P4 and P4c but not under P2 and P3, so
# P2 or P3 against P1 uses two_sample_diff. Each call starts its own
# generator from the seed

def bootstrap_ci(values, n_boot=10_000, seed=42):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return v.mean(), *np.percentile(means, [2.5, 97.5])


def paired_diff(a, b, n_boot=10_000, seed=42):
    """a minus b, the same resampled participants in both arms"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])


def two_sample_diff(a, b, n_boot=10_000, seed=42):
    """a minus b when the folds are not matched units (P2 against P1)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ra = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
    rb = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
    return a.mean() - b.mean(), *np.percentile(ra - rb, [2.5, 97.5])


def fmt(t, scale=1, digits=3):
    m, lo, hi = (scale * x for x in t)
    return f"{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def siblings_in_training(preds, per_recording=5):
    """For a frame-level split: how many frames of each test frame's own burst stayed in training"""
    rec = preds.sample_id.str.extract(r"^(\d+)_[a-z]_(\d+)_f\d+$")
    same_fold = preds.groupby([preds.fold, rec[0], rec[1]]).sample_id.transform("size")
    return per_recording - same_fold


def best_threshold_accuracy(labels, probs):
    """Highest accuracy over all thresholds. Chosen on test data, so an upper bound"""
    y, p = np.asarray(labels), np.asarray(probs)
    cand = np.unique(np.r_[p, 0, 1])
    return max(((p >= t).astype(int) == y).mean() for t in cand)
