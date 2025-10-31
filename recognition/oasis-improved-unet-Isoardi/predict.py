#!/usr/bin/env python3
from __future__ import annotations
import os, argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# local imports
from modules import ImprovedUNet
from dataset import get_datasets_and_data_loaders

# -----------------------------
# Utilities
# -----------------------------
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def hard_dice(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    preds/targets: [B,1,H,W] float in {0,1}
    """
    inter = (preds * targets).sum(dim=(1,2,3))
    den   = (preds + targets).sum(dim=(1,2,3)) + eps
    return float((2.0 * inter / den).mean().item())

@torch.no_grad()
def hard_iou(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds*targets).sum(dim=(1,2,3)) + eps
    return float((inter / union).mean().item())

def load_checkpoint_into(model: torch.nn.Module, ckpt_path: Path, dev: torch.device) -> Optional[int]:
    """
    Supports checkpoints saved by train.py (_save_ckpt).
    Returns epoch number if present.
    """
    ck = torch.load(ckpt_path, map_location=dev)
    if "model_state" in ck:
        model.load_state_dict(ck["model_state"])
        return int(ck.get("epoch", -1))
    # allow raw state_dict too
    model.load_state_dict(ck)
    return -1

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _list_pngs(p: Path) -> List[str]:
    return sorted([f for f in os.listdir(p) if f.lower().endswith(".png")])

# -----------------------------
# Raw images dataset (optional)
# -----------------------------
class GrayFolderDataset(Dataset):
    """
    For --images <dir> (no masks). Just loads grayscale PNGs and returns tensor [1,H,W].
    """
    def __init__(self, img_dir: str):
        self.img_dir = Path(img_dir)
        self.names = _list_pngs(self.img_dir)
        if not self.names:
            raise FileNotFoundError(f"No PNGs found in {self.img_dir}")
    def __len__(self): return len(self.names)
    def __getitem__(self, idx: int):
        name = self.names[idx]
        img = Image.open(self.img_dir / name).convert("L")
        # mimic ToTensor: HxW uint8 -> float [0,1], C=1
        arr = np.asarray(img, dtype=np.float32) / 255.0
        ten = torch.from_numpy(arr)[None, ...]  # [1,H,W]
        return ten, name  # no mask

# -----------------------------
# Inference helpers
# -----------------------------
@torch.no_grad()
def infer_batch(model, images: torch.Tensor, thr: float) -> torch.Tensor:
    """
    images: [B,1,H,W] in [0,1]; returns binary preds [B,1,H,W] in {0,1}
    """
    logits = model(images)
    probs  = torch.sigmoid(logits)
    return (probs > thr).float()

def save_mask(mask_tensor: torch.Tensor, out_path: Path) -> None:
    """
    mask_tensor: [1,H,W] float {0,1}
    Saves as 0/255 PNG.
    """
    m = (mask_tensor.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(m, mode="L").save(out_path)

def save_overlay(gray_img: Image.Image, mask: Image.Image, out_path: Path, alpha: float = 0.35) -> None:
    """
    gray_img: PIL "L"
    mask: PIL "L" with 0/255 — will be colorized red and alpha-blended
    """
    base = gray_img.convert("RGB")
    col  = ImageOps.colorize(mask, black=(0,0,0), white=(255,0,0))  # red
    blended = Image.blend(base, col, alpha=alpha)
    blended.save(out_path)

def make_gallery(rows: List[List[Image.Image]], out_path: Path) -> None:
    """
    rows: list of [img1, img2, ...] with same size; concatenates into a grid.
    """
    if not rows or not rows[0]:
        return
    w, h = rows[0][0].size
    cols = max(len(r) for r in rows)
    grid = Image.new("RGB", (cols*w, len(rows)*h), (255,255,255))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            grid.paste(im.convert("RGB"), (c*w, r*h))
    grid.save(out_path)

# -----------------------------
# Main flows
# -----------------------------
def run_on_split(args):
    """
    Uses your dataset.py loaders (paired imgs/masks) so we can compute metrics.
    """
    dev = device()
    print(f"Device: {dev}")

    # build loaders from your helper
    tr_ds, tr_dl, te_ds, te_dl, va_ds, va_dl = get_datasets_and_data_loaders(batch_size=args.batch_size)
    split_map = {"train": (tr_ds, tr_dl), "val": (va_ds, va_dl), "test": (te_ds, te_dl)}
    if args.split not in split_map:
        raise ValueError("--split must be one of: train, val, test")
    ds, dl = split_map[args.split]
    print(f"Loaded split '{args.split}' with {len(ds)} samples.")

    # model + weights
    model = ImprovedUNet(in_channels=1, num_classes=1, base_channels=args.base_channels).to(dev)
    ckpt = Path(args.checkpoint)
    if ckpt.is_dir():
        ckpt = ckpt / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}")
    ep = load_checkpoint_into(model, ckpt, dev)
    model.eval()
    if ep >= 0:
        print(f"Loaded weights from epoch {ep} ({ckpt})")
    else:
        print(f"Loaded weights from {ckpt}")

    # output dirs
    out_dir = Path(args.out_dir) / args.split
    png_dir = out_dir / "masks"
    ovl_dir = out_dir / "overlays"
    ensure_dir(png_dir)
    if args.save_overlays:
        ensure_dir(ovl_dir)

    # iterate
    all_dice, all_iou = [], []
    gallery_rows: List[List[Image.Image]] = []
    saved = 0

    for images, masks in DataLoader(ds, batch_size=args.batch_size):
        images = images.to(dev, non_blocking=True)  # [B,1,H,W]
        masks  = masks.to(dev, non_blocking=True)   # [B,1,H,W] (0/1)
        preds  = infer_batch(model, images, args.threshold)  # [B,1,H,W]

        # metrics
        all_dice.append(hard_dice(preds, masks))
        all_iou.append(hard_iou(preds, masks))

        # save files (use dataset ordering)
        B = images.size(0)
        for i in range(B):
            # figure out the filename: dataset stores sorted names in ds.image_names
            idx = saved + i
            name = ds.image_names[idx] if hasattr(ds, "image_names") else f"sample_{idx:05d}.png"
            out_mask = png_dir / name
            save_mask(preds[i], out_mask)

            if args.save_overlays:
                # reconstruct PIL from tensor for overlay
                img_np = (images[i].squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                msk_np = (preds[i].squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                img_pil = Image.fromarray(img_np, mode="L")
                msk_pil = Image.fromarray(msk_np, mode="L")
                out_ovl = ovl_dir / name
                save_overlay(img_pil, msk_pil, out_ovl, alpha=args.overlay_alpha)

                # (optional) build a tiny gallery (first N rows)
                if len(gallery_rows) < args.gallery_rows:
                    gallery_rows.append([
                        img_pil.convert("RGB"),
                        msk_pil.convert("RGB"),
                        Image.open(out_ovl).convert("RGB"),
                    ])

        saved += B

    dice_mean = float(np.mean(all_dice)) if all_dice else 0.0
    iou_mean  = float(np.mean(all_iou))  if all_iou  else 0.0
    print(f"[{args.split}] Dice={dice_mean:.4f} | IoU={iou_mean:.4f}")
    if gallery_rows:
        gal_path = out_dir / "prediction_gallery.png"
        make_gallery(gallery_rows, gal_path)
        print(f"Saved gallery → {gal_path}")
    print(f"Saved {saved} mask PNG(s) to {png_dir}")
    if args.save_overlays:
        print(f"Saved overlays to {ovl_dir}")

def run_on_folder(args):
    """
    Inference on a raw folder of PNGs (no masks).
    """
    dev = device()
    print(f"Device: {dev}")

    ds = GrayFolderDataset(args.images)
    dl = DataLoader(ds, batch_size=args.batch_size)

    model = ImprovedUNet(in_channels=1, num_classes=1, base_channels=args.base_channels).to(dev)
    ckpt = Path(args.checkpoint)
    if ckpt.is_dir():
        ckpt = ckpt / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}")
    ep = load_checkpoint_into(model, ckpt, dev)
    model.eval()
    if ep >= 0:
        print(f"Loaded weights from epoch {ep} ({ckpt})")
    else:
        print(f"Loaded weights from {ckpt}")

    out_dir = Path(args.out_dir) / "folder_infer"
    png_dir = out_dir / "masks"
    ovl_dir = out_dir / "overlays"
    ensure_dir(png_dir)
    if args.save_overlays:
        ensure_dir(ovl_dir)

    gallery_rows: List[List[Image.Image]] = []
    saved = 0

    for images, names in dl:
        images = images.to(dev, non_blocking=True)
        preds  = infer_batch(model, images, args.threshold)  # [B,1,H,W]
        B = images.size(0)
        for i in range(B):
            name = names[i]
            out_mask = png_dir / name
            save_mask(preds[i], out_mask)

            if args.save_overlays:
                img_np = (images[i].squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                msk_np = (preds[i].squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                img_pil = Image.fromarray(img_np, mode="L")
                msk_pil = Image.fromarray(msk_np, mode="L")
                out_ovl = ovl_dir / name
                save_overlay(img_pil, msk_pil, out_ovl, alpha=args.overlay_alpha)

                if len(gallery_rows) < args.gallery_rows:
                    gallery_rows.append([
                        img_pil.convert("RGB"),
                        msk_pil.convert("RGB"),
                        Image.open(out_ovl).convert("RGB"),
                    ])

        saved += B

    if gallery_rows:
        gal_path = out_dir / "prediction_gallery.png"
        make_gallery(gallery_rows, gal_path)
        print(f"Saved gallery → {gal_path}")
    print(f"Saved {saved} mask PNG(s) to {png_dir}")
    if args.save_overlays:
        print(f"Saved overlays to {ovl_dir}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Predict/visualise segmentation with ImprovedUNet.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--split", choices=["train","val","test"],
                      help="Run on a labeled dataset split using dataset.py (metrics available).")
    mode.add_argument("--images", type=str,
                      help="Run on a folder of raw PNGs (no masks).")

    p.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth",
                   help="Path to .pth OR directory containing best_model.pth")
    p.add_argument("--out-dir", type=str, default="./predictions",
                   help="Output directory root.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=64,
                   help="Must match training (default 64).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Sigmoid threshold for binary mask.")
    p.add_argument("--save-overlays", action="store_true",
                   help="Also save color overlays of mask on the grayscale image.")
    p.add_argument("--overlay-alpha", type=float, default=0.35,
                   help="Overlay opacity (0..1).")
    p.add_argument("--gallery-rows", type=int, default=4,
                   help="How many rows to include in the small gallery image.")
    return p.parse_args()

def main():
    args = parse_args()
    if args.split:
        run_on_split(args)
    else:
        run_on_folder(args)

if __name__ == "__main__":
    main()
