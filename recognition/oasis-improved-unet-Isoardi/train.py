from __future__ import annotations
import os, time, json
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

# model + losses + metrics
from modules import ImprovedUNet, CombinedLoss, dice_coefficient

# dataloaders (uses YOUR dataset.py as-is)
from dataset import get_datasets_and_data_loaders


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_sample_shapes(loader, device: torch.device):
    """Tiny smoke check so crashes are obvious."""
    imgs, msks = next(iter(loader))
    imgs, msks = imgs.to(device), msks.to(device)
    assert imgs.ndim == 4 and msks.ndim == 4, f"Expected [B,1,H,W], got {imgs.shape} & {msks.shape}"
    assert imgs.size(1) == 1 and msks.size(1) == 1, f"Expected 1 channel, got {imgs.size(1)} & {msks.size(1)}"
    return imgs.shape