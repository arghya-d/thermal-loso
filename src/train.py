import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import FrameDataset, get_transforms


def make_loaders(cfg, train, val, test, mean, std, device):
    size, bs = cfg["image_size"], cfg["batch_size"]
    kw = dict(batch_size=bs, num_workers=cfg.get("num_workers", 0), pin_memory=device.type == "cuda")
    # a last batch of one frame breaks BatchNorm, so drop it only in that case
    train_loader = DataLoader(FrameDataset(train, get_transforms(size, mean, std, True)),
                              shuffle=True, drop_last=len(train) % bs == 1, **kw)
    val_loader = DataLoader(FrameDataset(val, get_transforms(size, mean, std, False)), **kw)
    test_loader = DataLoader(FrameDataset(test, get_transforms(size, mean, std, False)), **kw)
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def eval_loss(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        total += loss_fn(model(x).squeeze(1), y).item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ids = [], []
    for x, _, sample_id in loader:
        probs += torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().tolist()
        ids += list(sample_id)
    return probs, ids


def fit(model, train_loader, val_loader, cfg, device):
    """Adam, learning rate cut on plateau, early stopping on the validation loss. Returns the best weights"""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=cfg.get("lr_factor", 0.1),
                                                           patience=cfg.get("lr_patience", 5))
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss, best_epoch, waited = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg["epochs"] + 1):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_loss(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_loss
