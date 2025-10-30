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

class Trainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, test_loader,
                 device: torch.device, save_dir: str = "./checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.device = device
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {k: [] for k in ["train_loss", "val_loss", "train_dice", "val_dice", "lr"]}
        self.best_val_dice = 0.0
        self.best_epoch = 0

    def _run_epoch(self, loader, criterion, optimizer=None, epoch=0, phase="Train"):
        training = optimizer is not None
        self.model.train(training)
        loss_sum, dice_sum, n = 0.0, 0.0, 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [{phase}]")
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device,  non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = criterion(logits, masks, num_classes=1)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    logits = self.model(images)
                    loss = criterion(logits, masks, num_classes=1)

            # batch metrics
            dice_list = dice_coefficient(logits, masks, num_classes=1)  # [dice]
            dice      = float(dice_list[0])
            bs = images.size(0)
            loss_sum += loss.item() * bs
            dice_sum += dice * bs
            n += bs

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{(dice_sum/max(n,1)):.4f}"})

        return loss_sum / max(n,1), dice_sum / max(n,1)

    def train(self, epochs=50, lr=3e-4, weight_decay=1e-5, patience=10, min_delta=1e-4):
        criterion = CombinedLoss(dice_weight=0.7, ce_weight=0.3)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        epochs_no_improve = 0
        t0 = time.time()
        for ep in range(1, epochs+1):
            tr_loss, tr_dice = self._run_epoch(self.train_loader, criterion, optimizer=optimizer, epoch=ep, phase="Train")
            va_loss, va_dice = self._run_epoch(self.val_loader,   criterion, optimizer=None,     epoch=ep, phase="Validation")
            scheduler.step()

            # log
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_dice"].append(tr_dice)
            self.history["val_dice"].append(va_dice)
            self.history["lr"].append(optimizer.param_groups[0]["lr"])

            print(f"\nEpoch {ep}/{epochs} | "
                  f"train: loss {tr_loss:.4f}, dice {tr_dice:.4f} | "
                  f"val: loss {va_loss:.4f}, dice {va_dice:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.6f}")

            # early stopping + best save
            if va_dice > self.best_val_dice + min_delta:
                self.best_val_dice = va_dice
                self.best_epoch = ep
                epochs_no_improve = 0
                self._save_ckpt(ep, optimizer, best=True)
                print(f"  ↳ Saved best (val_dice={va_dice:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  ↳ No improvement for {epochs_no_improve} epoch(s)")

            if ep % 10 == 0:
                self._save_ckpt(ep, optimizer, best=False)

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping @ epoch {ep}. Best val Dice {self.best_val_dice:.4f} @ epoch {self.best_epoch}.")
                break

        elapsed = (time.time() - t0)/60.0
        print(f"\nTraining done in {elapsed:.2f} min. Best val Dice {self.best_val_dice:.4f} @ epoch {self.best_epoch}.")
        self._plot_curves()

        # test with best weights
        self._load_best()
        test_loss, test_dice = self._run_epoch(self.test_loader, criterion, optimizer=None, epoch=0, phase="Test")
        print(f"\nTEST: loss={test_loss:.4f} | dice={test_dice:.4f}")

        with open(self.save_dir/"test_results.json","w") as f:
            json.dump({"test_loss": test_loss, "test_dice": test_dice}, f, indent=2)

    def _save_ckpt(self, epoch: int, optimizer, best: bool):
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_dice": self.best_val_dice,
            "history": self.history
        }
        path = self.save_dir / ("best_model.pth" if best else f"checkpoint_epoch_{epoch}.pth")
        torch.save(payload, path)

    def _load_best(self):
        p = self.save_dir / "best_model.pth"
        if p.exists():
            ck = torch.load(p, map_location=self.device)
            self.model.load_state_dict(ck["model_state"])
            print(f"Loaded best model from epoch {ck['epoch']}")
        else:
            print("No best_model.pth found; using current weights.")

    def _plot_curves(self):
        try:
            epochs = range(1, len(self.history["train_loss"])+1)
            fig, ax = plt.subplots(1, 3, figsize=(16,5))
            ax[0].plot(epochs, self.history["train_loss"], label="train")
            ax[0].plot(epochs, self.history["val_loss"],   label="val")
            ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(True, alpha=0.3)

            ax[1].plot(epochs, self.history["train_dice"], label="train")
            ax[1].plot(epochs, self.history["val_dice"],   label="val")
            ax[1].axhline(0.9, ls="--", c="r", label="target 0.9")
            ax[1].set_title("Dice"); ax[1].legend(); ax[1].grid(True, alpha=0.3)

            ax[2].plot(epochs, self.history["lr"])
            ax[2].set_yscale("log"); ax[2].set_title("Learning rate"); ax[2].grid(True, alpha=0.3)

            fig.tight_layout()
            out = self.save_dir/"training_curves.png"
            fig.savefig(out, dpi=200)
            print(f"Saved training curves to {out}")
            plt.close(fig)
        except Exception as e:
            print(f"Plotting skipped: {e}")