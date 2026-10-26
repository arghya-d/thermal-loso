import torch.nn as nn
from torchvision import models


def get_model(name="resnet18", dropout=0.5, pretrained=True):
    """ImageNet backbone with the classifier replaced by dropout and one output (a logit)"""
    weights = "DEFAULT" if pretrained else None

    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 1))
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(model.classifier[1].in_features, 1))
    else:
        raise ValueError(f"unknown model {name}")
    return model
