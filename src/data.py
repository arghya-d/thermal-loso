import glob
import os
import random
import re
from functools import lru_cache

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

SITES = {"face": "f", "eye": "e", "ear": "r", "hand": "h"}


# reading the two datasets

def list_recordings(cfg):
    """{(subject, acquisition): file(s)} for the configured dataset and site"""
    if cfg["dataset"] == "pucv":
        return _pucv_recordings(cfg["data_root"])
    return _patras_recordings(cfg["data_root"], SITES[cfg["site"]])


def _patras_recordings(root, site):
    # files: like 01_<name>_1_e_<sex>_<age>_<weight>.tif, one 50-frame burst each
    recs = {}
    for path in sorted(glob.glob(os.path.join(root, "*", f"*_{site}_*.tif"))):
        m = re.match(r"(\d+)_.*?_(\d+)_([ferh])", os.path.basename(path))
        if m and m.group(3) == site:
            recs[(int(m.group(1)), int(m.group(2)))] = path
    return recs


def _pucv_recordings(root):
    #one folder per condition, files Sujeto_<subject>_<condition>_<frame>_.jpg
    recs = {}
    for path in glob.glob(os.path.join(root, "*", "Sujeto_*.jpg")):
        m = re.search(r"Sujeto_(\d+)_(\d+)_(\d+)_\.jpg$", path)
        if m:
            subject, cond, frame = map(int, m.groups())
            recs.setdefault((subject, cond), []).append((frame, path))
    return {k: tuple(p for _, p in sorted(v)) for k, v in sorted(recs.items())}


@lru_cache(maxsize=None)
def read_recording(files):
    """All frames of one recording as a float32 array (n, h, w)"""
    if isinstance(files, str):
        stack = tifffile.imread(files).astype(np.float32)
        return stack[None] if stack.ndim == 2 else stack
    return np.stack([np.asarray(Image.open(f).convert("L"), dtype=np.float32) for f in files])


def make_id(cfg, subject, acq, idx):
    if cfg["dataset"] == "pucv":
        return f"{subject}_f_{acq}_f{idx:02d}"
    return f"{subject:02d}_{SITES[cfg['site']]}_{acq}_f{idx:02d}"


def parse_id(sample_id):
    subject, _, acq, idx = re.match(r"^(\d+)_([a-z])_(\d+)_f(\d+)$", sample_id).groups()
    return int(subject), int(acq), int(idx)


# choosing frames
# A frame is a tuple (image, label, subject, sample_id, acquisition).

def sample_frames(cfg, acqs=None, n=None):
    """n random frames from each recording of the given acquisitions
    Uses the global random state, so call set_seed first.
    """
    acqs = acqs or (cfg["sober"], cfg["intoxicated"])
    n = n or cfg["frames_per_recording"]
    excluded = set(cfg.get("excluded", []))

    frames = []
    for (subject, acq), files in list_recordings(cfg).items():
        if subject in excluded or acq not in acqs:
            continue
        stack = read_recording(files)
        label = 0 if acq == cfg["sober"] else 1
        for i in random.sample(range(len(stack)), min(n, len(stack))):
            frames.append((stack[i], label, subject, make_id(cfg, subject, acq, i), acq))
    return frames


def frames_from_ids(cfg, sample_ids):
    """The exact frames listed in a manifest, in the same order"""
    recs = list_recordings(cfg)
    frames = []
    for sid in sample_ids:
        subject, acq, i = parse_id(sid)
        stack = read_recording(recs[(subject, acq)])
        label = 0 if acq == cfg["sober"] else 1
        frames.append((stack[i], label, subject, sid, acq))
    return frames


# Normalisation

@lru_cache(maxsize=None)
def _sums(files):
    x = read_recording(files).astype(np.float64)
    return x.size, x.sum(), (x ** 2).sum()


def norm_stats(cfg, subjects):
    """Mean and sd over every frame of these subjects' sober and intoxicated recordings."""
    subjects = set(subjects)
    n = s = ss = 0.0
    for (subject, acq), files in list_recordings(cfg).items():
        if subject in subjects and acq in (cfg["sober"], cfg["intoxicated"]):
            a, b, c = _sums(files)
            n, s, ss = n + a, s + b, ss + c
    mean = s / n
    return mean, float(np.sqrt((ss - n * mean ** 2) / (n - 1)))


# transforms and dataset

def _to_pil(x):
    return to_pil_image(x.astype(np.float32), mode="F")   # keeps raw values, no 8-bit rounding


def _to_tensor(x):
    x = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(np.stack([x, x, x]))           # grey, copied to 3 channels


def get_transforms(size, mean, std, train):
    ops = [transforms.Lambda(_to_pil), transforms.Resize(tuple(size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    ops += [transforms.Lambda(_to_tensor), transforms.Normalize([mean] * 3, [std] * 3)]
    return transforms.Compose(ops)


class FrameDataset(Dataset):
    def __init__(self, frames, transform):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        image, label, _, sample_id, _ = self.frames[i]
        return self.transform(image), torch.tensor(float(label)), sample_id
