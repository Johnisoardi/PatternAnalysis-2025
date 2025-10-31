# OASIS Brain MRI Segmentation with a Compact U-Net

## Overview - Problem & Algorithm
This project performs **2D brain tissue segmentation** on the OASIS *keras_png_slices* dataset using an enhanced **U-Net** neural network. The task is to predict a **binary mask** (foreground vs background tissue such as grey and white matter) from single-channel MRI slices. U-Net’s encoder/decoder with **skip connections** preserves spatial detail while learning high level context, making it well suited to medical image segmentation.

## How It Works (High Level)
- Each input slice is converted to grayscale, normalized to `[0,1]`, and passed through a **four level encoder** and **symmetric decoder**.
- A final **1×1 convolution** outputs a logit map, which is converted to a probability via **sigmoid** and then **thresholded** to a binary mask.
- **Training loss:** `Combined = 0.3 × BCEWithLogits + 0.7 × Soft Dice`
- **Evaluation metrics:** **Dice** and **IoU**.

## Repo Structure
```text
── dataset.py            # OASIS PNG dataset + DataLoaders (train/val/test)
├── modules.py           # U-Net, blocks, losses, metrics, parameter count
├── train.py             # training loop, early stopping, curve plotting
├── predict.py           # batch inference on splits or raw folders
├── checkpoints/         # saved models and training_curves.png
└── predictions/         # saved masks, overlays, and prediction_gallery.png
```

## Dataset
OASIS “keras_png_slices” PNGs arranged as:
```text
keras_png_slices_train/        keras_png_slices_seg_train/
keras_png_slices_validate/     keras_png_slices_seg_validate/
keras_png_slices_test/         keras_png_slices_seg_test/
```

Set `DATA_DIR` in `dataset.py` to your path.  
**Currently:**
```text
/mnt/c/Users/jwiso/OneDrive/Desktop/COMP3710/keras_png_slices_data/keras_png_slices_data
```

## Pre-processing Steps
- **Images:** load as grayscale (`.convert("L")`) and apply `ToTensor()` to get floats in `[0,1]` with shape `[1, H, W]`.
- **Masks:** load as grayscale, use `PILToTensor()`, then binarize with `(mask > 0).float()` to get values in `{0,1}`.
- **Geometry:** no resizing or cropping as the decoder handles odd spatial sizes via padding.

## Justification of Splits
Followed the dataset’s provided splits:
- **Train:** roughly 9.6k slices  
- **Val:** roughly 1.1k slices  
- **Test:** roughly 0.5k slices  

This preserves an unseen test set for fair evaluation and uses a validation set for early stopping and hyperparameter tuning without leaking test information.

## Environment & Dependencies
| Dependency  | Recommended Version     | Notes                                                |
|-------------|--------------------------|------------------------------------------------------|
| Python      | 3.10+                    | Conda environment already satisfies this. 
                |
| PyTorch     | ≥ 2.1 (CUDA 11.8+)       | Mixed precision (`torch.amp`) accelerates GPU training.   |
| Torchvision | ≥ 0.16                   | Provides transforms and normalisation utilities.     |
| NumPy       | ≥ 1.26                   | Array ops and numerical utilities.                   |
| Pillow      | ≥ 10                     | Image I/O and conversion.                            |
| Matplotlib  | ≥ 3.7                    | Curve and prediction plots.                          |
| tqdm        | ≥ 4.66                   | Progress bars during training/eval.                  |


## Training
From the folder containing `modules.py` and `dataset.py`:
```bash
python train.py --epochs 50 --batch-size 8 --lr 3e-4 --wd 1e-5 --save-dir ./checkpoints
```

### You Will Get
- `checkpoints/best_model.pth` *(best validation Dice)*
- `checkpoints/checkpoint_epoch_XX.pth` *(periodic)*
- `checkpoints/training_curves.png` *(loss, Dice, LR)*

**Justifying these choices**  
Combined loss balances pixel-wise accuracy and region overlap (class imbalance); **AdamW + cosine LR** are strong defaults; **early stopping** prevents overfitting.

## Evaluate on a Labeled Split and Save Masks and Overlays
```bash
python predict.py --split test --checkpoint ./checkpoints --save-overlays --gallery-rows 5
```

### Outputs
- `predictions/test/masks/*.png` *(0/255 binary)*
- `predictions/test/overlays/*.png` *(image + red mask overlay)*
- `predictions/test/prediction_gallery.png`
- Printed mean **Dice** and **IoU** for the split
