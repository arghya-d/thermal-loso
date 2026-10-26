import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from src import data, splits
from src.model import get_model
from src.stats import fold_metrics
from src.train import fit, make_loaders, predict


def load_config(path, **changes):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.update(changes)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(cfg, rule, name, manifest=None, out_dir="outputs/runs", pretrained=True):
    """Train and test one partition rule. Writes <name>_preds.csv and <name>_folds.csv
    manifest: a DataFrame (sample_id, fold). If given, the same frames and test
    folds are used as in that file, otherwise frames are drawn with the seed
    """
    device = get_device()
    seed = cfg["seed"]
    set_seed(seed)

    if manifest is not None:
        frames = data.frames_from_ids(cfg, manifest.sample_id)
        folds = splits.folds_from_manifest(frames, manifest)
    else:
        frames = data.sample_frames(cfg)
        folds = splits.make_folds(rule, frames, seed, cfg["frames_per_recording"])

    # P4 and P4c add frames of one person from the intermediate acquisitions,
    # labelled intoxicated
    pool = []
    if rule in ("P4", "P4c"):
        pool = data.sample_frames(cfg, acqs=tuple(cfg["identity_acqs"]), n=cfg["identity_frames"])

    subjects = sorted({f[2] for f in frames})
    whole_cohort = rule in ("P2", "P3") or cfg.get("global_norm", False)
    if whole_cohort:
        mean, std = data.norm_stats(cfg, subjects)

    preds, rows = [], []
    for fold, test_idx in tqdm(folds, desc=name):
        held = set(test_idx)
        test = [frames[i] for i in test_idx]
        train = [f for i, f in enumerate(frames) if i not in held]

        # 10% of the training pool for early stopping. Seeded with the participant
        # under P1 and the fold index under P2/P3, as in the original runs
        val_seed = seed + fold if rule in ("P1", "P4", "P4c") else seed + fold - 1
        random.Random(val_seed).shuffle(train)
        n_val = max(1, int(len(train) * cfg["val_fraction"]))
        val, train = train[:n_val], train[n_val:]

        added = []
        if rule == "P4":
            added = [f for f in pool if f[2] == fold]
        elif rule == "P4c":
            others = sorted({f[2] for f in pool} & (set(subjects) - {fold}))
            donor = random.Random(seed + fold).choice(others)
            added = [f for f in pool if f[2] == donor]
        splits.check(rule, train, val, test, added)
        train = train + added

        if not whole_cohort:
            mean, std = data.norm_stats(cfg, [s for s in subjects if s != fold])

        train_loader, val_loader, test_loader = make_loaders(cfg, train, val, test, mean, std, device)
        model = get_model(cfg["model"], cfg["dropout"], pretrained).to(device)
        model, best_epoch, best_loss = fit(model, train_loader, val_loader, cfg, device)

        probs, ids = predict(model, test_loader, device)
        labels = [f[1] for f in test]
        preds += [dict(fold=fold, sample_id=i, label=y, probability=p,
                       prediction=int(p >= cfg["threshold"])) for i, y, p in zip(ids, labels, probs)]
        rows.append(dict(fold=fold, **fold_metrics(labels, probs, cfg["threshold"]),
                         best_epoch=best_epoch, best_val_loss=best_loss))

    preds, folds_df = pd.DataFrame(preds), pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    preds.to_csv(os.path.join(out_dir, f"{name}_preds.csv"), index=False)
    folds_df.to_csv(os.path.join(out_dir, f"{name}_folds.csv"), index=False)
    print(f"{name}: accuracy {100 * folds_df.accuracy.mean():.1f}  MCC {folds_df.mcc.mean():.3f}  "
          f"AUC {folds_df.roc_auc.mean():.3f}")
    return preds, folds_df
