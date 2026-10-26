import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
FEATURES = [f"p{p:02d}" for p in PERCENTILES] + ["std", "mean", "iqr"]


def frame_features(frame, body_threshold=0):
    """Statistics of the face pixels. Patras background is 0, PUCV has no zero background."""
    body = frame.ravel().astype(np.float64)
    body = body[body > body_threshold]
    feats = {f"p{p:02d}": np.percentile(body, p) for p in PERCENTILES}
    feats.update(std=body.std(), mean=body.mean(),
                 iqr=np.percentile(body, 75) - np.percentile(body, 25))
    return feats


def _fit_predict(train, test, cols):
    scaler = StandardScaler().fit(train[cols])
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(train[cols]), train.label)
    return model.predict_proba(scaler.transform(test[cols]))[:, 1]


def loso_scores(df, cols):
    out = np.full(len(df), np.nan)
    for s in df.fold.unique():
        held = (df.fold == s).to_numpy()
        out[held] = _fit_predict(df[~held], df[held], cols)
    return out


def nested_baseline(df, threshold=0.5):
    """df has one row per frame: fold, sample_id, label and the FEATURES columns"""
    probs = np.full(len(df), np.nan)
    chosen = {}
    for s in sorted(df.fold.unique()):
        held = (df.fold == s).to_numpy()
        train = df[~held]
        scores = {c: roc_auc_score(train.label, loso_scores(train, [c])) for c in FEATURES}
        best = max(scores, key=scores.get)
        chosen[s] = best
        probs[held] = _fit_predict(train, df[held], [best])

    out = df[["fold", "sample_id", "label"]].copy()
    out["probability"] = probs
    out["prediction"] = (probs >= threshold).astype(int)
    out["feature"] = out.fold.map(chosen)
    return out
