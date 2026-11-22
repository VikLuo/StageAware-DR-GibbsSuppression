"""Torchvision ConvNeXt-Small architecture with checkpoint-compatible heads.

This models the architecture reconstructed from the available first-fold weights;
the original training code and all training/checkpoint metadata are unavailable.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import torch
from torch import nn
from torchvision.models import convnext_small


def build_convnext_small(num_classes: int, head_style: str = "legacy", dropout: float = 0.3) -> nn.Module:
    """Build 2-class gate or 5-class ICDR grading network.

    legacy generates classifier.2.1.{weight,bias}, compatible with the
    key layout of the historical Fold 1 weights. The presence and
    probability of Dropout cannot be recovered from a state_dict alone.
    """
    if num_classes not in (2, 5):
        raise ValueError("num_classes must be 2 or 5")
    model = convnext_small(weights=None)
    features = model.classifier[-1].in_features
    if head_style == "legacy":
        model.classifier[-1] = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(features, num_classes))
    elif head_style == "linear":
        model.classifier[-1] = nn.Linear(features, num_classes)
    else:
        raise ValueError("head_style must be 'legacy' or 'linear'")
    return model


def load_checkpoint(model: nn.Module, filepath: str | Path, device: torch.device) -> nn.Module:
    """Load only trusted model weights and require matching architecture."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        weights = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Torch <2.0 does not support weights_only; refuse unsafe deserialization.
        raise RuntimeError("Use a current PyTorch version supporting torch.load(weights_only=True)")
    if isinstance(weights, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in weights and isinstance(weights[key], dict):
                weights = weights[key]
                break
    if not isinstance(weights, (dict, OrderedDict)):
        raise ValueError("Checkpoint does not contain a valid state_dict")
    if weights and all(k.startswith("module.") for k in weights):
        weights = OrderedDict((k.removeprefix("module."), v) for k, v in weights.items())
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    return model
