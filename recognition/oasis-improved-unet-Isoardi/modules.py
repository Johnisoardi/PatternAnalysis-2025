# modules.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# -----------------------------
# Blocks (components)
# -----------------------------
class DoubleConv(nn.Module):
    """(Conv-BN-ReLU) x 2"""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool then DoubleConv"""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(c_in, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upscale with ConvTranspose then concat skip, then DoubleConv"""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 2, stride=2)
        # After concat: channels = c_out (from up) + skip_channels (also c_out if symmetric)
        self.conv = DoubleConv(c_in, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle odd input dims
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -----------------------------
# Model
# -----------------------------
class ImprovedUNet(nn.Module):
    """
    Compact U-Net for 2D segmentation.
    Args:
        in_channels: e.g., 1 for grayscale
        num_classes: 1 for binary (logit), >1 for multiclass (logits per class)
        base_channels: width multiplier
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_channels: int = 64):
        super().__init__()
        b = base_channels
        self.inc   = DoubleConv(in_channels, b)
        self.down1 = Down(b, b*2)
        self.down2 = Down(b*2, b*4)
        self.down3 = Down(b*4, b*8)
        self.bot   = DoubleConv(b*8, b*16)
        self.up3   = Up(b*16, b*8)
        self.up2   = Up(b*8,  b*4)
        self.up1   = Up(b*4,  b*2)
        self.up0   = Up(b*2,  b)
        self.outc  = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c0 = self.inc(x)           # b
        c1 = self.down1(c0)        # 2b
        c2 = self.down2(c1)        # 4b
        c3 = self.down3(c2)        # 8b
        b  = self.bot(c3)          # 16b
        u3 = self.up3(b,  c3)      # 8b
        u2 = self.up2(u3, c2)      # 4b
        u1 = self.up1(u2, c1)      # 2b
        u0 = self.up0(u1, c0)      # b
        return self.outc(u0)       # logits [B, C, H, W]


# -----------------------------
# Losses
# -----------------------------
class SoftDiceLoss(nn.Module):
    """Soft Dice loss. Binary or multiclass via num_classes."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 1) -> torch.Tensor:
        if num_classes == 1:
            probs = torch.sigmoid(logits)          # [B,1,H,W]
            targets = targets.float()
            num = 2.0 * (probs * targets).sum(dim=(1,2,3))
            den = (probs + targets).sum(dim=(1,2,3)) + self.eps
            return 1.0 - (num / den).mean()
        else:
            probs = F.softmax(logits, dim=1)       # [B,C,H,W]
            if targets.ndim == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0,3,1,2).float()
            num = 2.0 * (probs * one_hot).sum(dim=(0,2,3))   # [C]
            den = (probs + one_hot).sum(dim=(0,2,3)) + self.eps
            return 1.0 - (num / den).mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of BCE/CE and SoftDice.
    - If num_classes==1: BCEWithLogits + Dice
    - Else: CrossEntropy + Dice
    """
    def __init__(self, dice_weight: float = 0.7, ce_weight: float = 0.3, ignore_index: int = -100):
        super().__init__()
        self.dw, self.cw = dice_weight, ce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.ce  = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 1) -> torch.Tensor:
        if num_classes == 1:
            ce = self.bce(logits, targets.float())
        else:
            if targets.ndim == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            ce = self.ce(logits, targets.long())
        dl = self.dice(logits, targets, num_classes=num_classes)
        return self.cw*ce + self.dw*dl


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 1,
                     threshold: float = 0.5, eps: float = 1e-6) -> List[float]:
    """
    Returns per-class Dice (hard Dice; batch-averaged).
      - binary: [dice]
      - multiclass: [dice_c for each class c]
    """
    if num_classes == 1:
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        targets = targets.float()
        inter = (preds * targets).sum(dim=(1,2,3))
        den   = (preds + targets).sum(dim=(1,2,3)) + eps
        return [ (2.0*inter/den).mean().item() ]
    else:
        pred_cls = torch.argmax(logits, dim=1)  # [B,H,W]
        if targets.ndim == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        out: List[float] = []
        for c in range(num_classes):
            pc = (pred_cls == c).float()
            tc = (targets  == c).float()
            inter = (pc * tc).sum(dim=(1,2))
            den   = (pc + tc).sum(dim=(1,2)) + eps
            out.append((2.0*inter/den).mean().item())
        return out


@torch.no_grad()
def iou_coefficient(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 1,
                    threshold: float = 0.5, eps: float = 1e-6) -> List[float]:
    """Per-class IoU (hard; batch-averaged)."""
    if num_classes == 1:
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        targets = targets.float()
        inter = (preds * targets).sum(dim=(1,2,3))
        union = (preds + targets - preds*targets).sum(dim=(1,2,3)) + eps
        return [ (inter/union).mean().item() ]
    else:
        pred_cls = torch.argmax(logits, dim=1)
        if targets.ndim == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        out: List[float] = []
        for c in range(num_classes):
            pc = (pred_cls == c).float()
            tc = (targets  == c).float()
            inter = (pc * tc).sum(dim=(1,2))
            union = (pc + tc - pc*tc).sum(dim=(1,2)) + eps
            out.append((inter/union).mean().item())
        return out


# -----------------------------
# Utilities
# -----------------------------
def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "ImprovedUNet",
    "DoubleConv",
    "Down",
    "Up",
    "SoftDiceLoss",
    "CombinedLoss",
    "dice_coefficient",
    "iou_coefficient",
    "count_parameters",
]
