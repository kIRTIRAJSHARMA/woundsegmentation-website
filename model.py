"""
model.py
Loads the trained U-Net (EfficientNet-B4 encoder) wound segmentation model
and runs inference on a single image. This mirrors CELL 6 / CELL 39 of the
original notebook exactly, so the weights you trained there will load here
with no changes needed.
"""

import os
import sys
import urllib.request
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Config (must match training config) ──────────────────────────────
IMG_SIZE = 256
ENCODER = "efficientnet-b4"
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "model_weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "best_baseline.pth")
WEIGHTS_URL  = "https://huggingface.co/kirtiraj7/woundseg-model/resolve/main/best_baseline.pth"


def download_weights():
    """Download model weights from Hugging Face if not already present."""
    if os.path.exists(WEIGHTS_PATH):
        return

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"Downloading model weights from {WEIGHTS_URL} ...", flush=True)

    try:
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        print(f"Model weights downloaded successfully ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        # Remove partial file if download failed
        if os.path.exists(WEIGHTS_PATH):
            os.remove(WEIGHTS_PATH)
        print(f"ERROR: Failed to download model weights: {e}", flush=True)
        sys.exit(1)


# ── Preprocessing (identical to val_transform in the notebook) ───────
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ToTensorV2(),
])

_model = None  # loaded once, cached


def load_model():
    """Downloads weights if needed, then loads the model once (singleton)."""
    global _model
    if _model is not None:
        return _model

    download_weights()

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {WEIGHTS_PATH} and download did not succeed."
        )

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,       # we're loading our own trained weights
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
    ).to(DEVICE)

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    _model = model
    print(f" Model loaded on {DEVICE}")
    return _model


def run_inference(image_path, threshold: float = THRESHOLD):
    """
    Runs the trained model on a single image file and returns:
      - overlay_bgr:  original image with the predicted wound region highlighted in red
      - mask_bgr:      binary mask (white = wound) as a 3-channel image for saving
      - heatmap_bgr:   probability heatmap overlaid on the original image
      - dice_area_pct: percentage of the image area predicted as wound (simple stat)
    All returned images are BGR numpy arrays, ready for cv2.imwrite.
    """
    model = load_model()

    raw = cv2.imread(image_path)
    if raw is None:
        raise ValueError(f"Could not read image at {image_path}")
    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    aug = val_transform(image=raw_rgb)
    img_t = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(img_t)
    prob = torch.sigmoid(logit).squeeze().cpu().numpy()
    pred_bin = (prob > threshold).astype(np.uint8)

    h, w = raw_rgb.shape[:2]
    prob_disp = cv2.resize(prob, (w, h))
    mask_disp = cv2.resize(pred_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    # Overlay: red highlight over predicted wound region
    overlay_rgb = raw_rgb.copy()
    overlay_rgb[mask_disp == 1] = (
        overlay_rgb[mask_disp == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
    ).astype(np.uint8)

    # Heatmap: probability map in jet colormap blended onto original
    heat_u8 = (prob_disp * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR
    heat_color_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.addWeighted(raw_rgb, 0.55, heat_color_rgb, 0.45, 0)

    # Binary mask as a viewable 3-channel image
    mask_rgb = cv2.cvtColor(mask_disp * 255, cv2.COLOR_GRAY2RGB)

    wound_area_pct = round(float(mask_disp.mean()) * 100, 2)

    return {
        "overlay": cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
        "mask": cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR),
        "heatmap": cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR),
        "wound_area_pct": wound_area_pct,
    }
