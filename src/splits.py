"""The partition rules:

Every function returns a list of (fold, test indices) into the frame list
Training is always the rest of the frames, so only the choice of test set
differs between rules

P1  by participant   (fold = subject id)
P2  by frame         (fold = 1..n)
P3  by recording     (fold = 1..n)
P4 and P4c use the P1 folds and add frames in run.py
"""

import random
import pandas as pd


def by_participant(frames):
    subjects = sorted({f[2] for f in frames})
    return [(s, [i for i, f in enumerate(frames) if f[2] == s]) for s in subjects]


def by_frame(frames, n_folds, per_class, seed):
    """Balanced folds of per_class sober and per_class intoxicated frames, drawn from the pool"""
    rng = random.Random(seed)
    idx = {0: [], 1: []}
    for i, f in enumerate(frames):
        idx[f[1]].append(i)
    rng.shuffle(idx[0])
    rng.shuffle(idx[1])

    folds = []
    for k in range(n_folds):
        test = idx[0][k * per_class:(k + 1) * per_class] + idx[1][k * per_class:(k + 1) * per_class]
        if len(test) < 2 * per_class:
            break
        folds.append((k + 1, sorted(test)))
    return folds


def by_recording(frames, n_folds, seed):
    """One sober recording of one person and one intoxicated recording of another"""
    rng = random.Random(seed)
    recs = {}
    for i, f in enumerate(frames):
        recs.setdefault((f[2], f[4], f[1]), []).append(i)

    sober = [k for k in recs if k[2] == 0]
    drunk = [k for k in recs if k[2] == 1]
    rng.shuffle(sober)
    rng.shuffle(drunk)

    n = min(n_folds, len(sober), len(drunk))
    for k in range(n):
        if sober[k][0] == drunk[k][0]:
            # same person on both sides would make this a P1 fold, so swap with the next pair, 
            # or at the end of the list with an earlier pair that gives no clash
            if k + 1 < len(drunk):
                j = k + 1
            else:
                j = next(i for i in range(k) if drunk[i][0] != sober[k][0] and drunk[k][0] != sober[i][0])
            drunk[k], drunk[j] = drunk[j], drunk[k]
    return [(k + 1, sorted(recs[sober[k]] + recs[drunk[k]])) for k in range(n)]


def make_folds(rule, frames, seed, per_class):
    n = len({f[2] for f in frames})
    if rule in ("P1", "P4", "P4c"):
        return by_participant(frames)
    if rule == "P2":
        return by_frame(frames, n, per_class, seed)
    if rule == "P3":
        return by_recording(frames, n, seed)
    raise ValueError(rule)


def folds_from_manifest(frames, manifest):
    pos = {f[3]: i for i, f in enumerate(frames)}
    return [(int(fold), sorted(pos[s] for s in g.sample_id)) for fold, g in manifest.groupby("fold")]


def to_manifest(frames, folds):
    rows = [(frames[i][3], fold) for fold, test in folds for i in test]
    return pd.DataFrame(rows, columns=["sample_id", "fold"])


def check(rule, train, val, test, added=()):
    """Assert the property each rule is meant to have."""
    test_ids = {f[3] for f in test}
    assert not test_ids & {f[3] for f in train + val + list(added)}, "test frame in training"

    test_subjects = {f[2] for f in test}
    pool_subjects = {f[2] for f in train + val}
    test_recs = {(f[2], f[4]) for f in test}
    pool_recs = {(f[2], f[4]) for f in train + val}

    if rule in ("P1", "P4", "P4c"):
        assert len(test_subjects) == 1 and not test_subjects & pool_subjects
    if rule == "P3":
        assert not test_recs & pool_recs, "sibling of a test frame in training"
    if rule == "P4":
        assert {f[2] for f in added} == test_subjects
    if rule == "P4c":
        assert not {f[2] for f in added} & test_subjects
    if added:
        assert not {(f[2], f[4]) for f in added} & test_recs, "added frames share a recording with the test set"
