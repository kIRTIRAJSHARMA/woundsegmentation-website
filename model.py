"""
model.py
U-Net (EfficientNet-B4) wound segmentation model.
All heavy imports are deferred inside load_model() and run_inference().
"""

import os
import sys
import gc
import traceback
import urllib.request

os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

IMG_SIZE  = 256
ENCODER   = "efficientnet-b4"
IMG_MEAN  = [0.485, 0.456, 0.406]
IMG_STD   = [0.229, 0.224, 0.225]
THRESHOLD = 0.5
DEVICE    = "cpu"

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR  = os.path.join(BASE_DIR, "model_weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "best_baseline.pth")
WEIGHTS_URL  = "https://huggingface.co/kirtiraj7/woundseg-model/resolve/main/best_baseline.pth"

_model         = None
_val_transform = None


def _rss_mb():
    """Return process RSS memory in MB using only stdlib."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return -1


def _log(msg):
    print(f"{msg}  [RSS {_rss_mb():.0f} MB]", flush=True)


def download_weights():
    if os.path.exists(WEIGHTS_PATH):
        _log("=== WEIGHTS ALREADY PRESENT ===")
        return

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    _log("=== DOWNLOADING WEIGHTS ===")

    try:
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        _log(f"=== WEIGHTS READY ({size_mb:.1f} MB) ===")
    except Exception as e:
        if os.path.exists(WEIGHTS_PATH):
            os.remove(WEIGHTS_PATH)
        print(f"=== WEIGHTS DOWNLOAD FAILED: {e} ===", flush=True)
        traceback.print_exc()
        sys.exit(1)


def load_model():
    global _model, _val_transform

    if _model is not None:
        _log("=== MODEL ALREADY CACHED ===")
        return _model

    _log("=== PREDICT START ===")

    # ── torch ────────────────────────────────────────────────────────
    _log("=== IMPORTING TORCH ===")
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _log("=== TORCH IMPORTED ===")
    except Exception:
        print("=== TORCH IMPORT FAILED ===", flush=True)
        traceback.print_exc()
        raise

    # ── segmentation_models_pytorch ──────────────────────────────────
    _log("=== IMPORTING SMP ===")
    try:
        import segmentation_models_pytorch as smp
        _log("=== SMP IMPORTED ===")
    except Exception:
        print("=== SMP IMPORT FAILED ===", flush=True)
        traceback.print_exc()
        raise

    # ── albumentations ───────────────────────────────────────────────
    _log("=== IMPORTING ALBUMENTATIONS ===")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        _log("=== ALBUMENTATIONS IMPORTED ===")
    except Exception:
        print("=== ALBUMENTATIONS IMPORT FAILED ===", flush=True)
        traceback.print_exc()
        raise

    # ── weights ──────────────────────────────────────────────────────
    download_weights()

    # ── preprocessing pipeline ───────────────────────────────────────
    _log("=== BUILDING TRANSFORM ===")
    try:
        _val_transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ToTensorV2(),
        ])
        _log("=== TRANSFORM READY ===")
    except Exception:
        print("=== TRANSFORM BUILD FAILED ===", flush=True)
        traceback.print_exc()
        raise

    # ── create model ─────────────────────────────────────────────────
    _log("=== CREATING MODEL ===")
    try:
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type="scse",
        ).to(DEVICE)
        _log("=== MODEL CREATED ===")
    except Exception:
        print("=== MODEL CREATION FAILED ===", flush=True)
        traceback.print_exc()
        raise

    # ── load checkpoint ──────────────────────────────────────────────
    _log("=== LOADING STATE DICT ===")
    try:
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        _log("=== STATE DICT LOADED ===")
    except Exception:
        print("=== STATE DICT LOAD FAILED ===", flush=True)
        traceback.print_exc()
        raise

    # ── apply weights ────────────────────────────────────────────────
    _log("=== LOADING STATE INTO MODEL ===")
    try:
        model.load_state_dict(checkpoint)
        del checkpoint
        gc.collect()
        _log("=== STATE DICT APPLIED ===")
    except Exception:
        print("=== STATE DICT APPLY FAILED ===", flush=True)
        traceback.print_exc()
        raise

    # ── eval mode ────────────────────────────────────────────────────
    _log("=== MODEL EVAL ===")
    model.eval()
    _model = model
    _log("=== MODEL READY ===")

    return _model


def run_inference(image_path, threshold=THRESHOLD):
    import cv2
    import numpy as np
    import torch

    cv2.setNumThreads(1)

    model = load_model()

    _log("=== STARTING INFERENCE ===")

    try:
        raw = cv2.imread(image_path)
        if raw is None:
            raise ValueError(f"Could not read image at {image_path}")
        raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        aug   = _val_transform(image=raw_rgb)
        img_t = aug["image"].unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logit = model(img_t)
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()

        del img_t, logit
        gc.collect()

        pred_bin  = (prob > threshold).astype(np.uint8)
        h, w      = raw_rgb.shape[:2]
        prob_disp = cv2.resize(prob, (w, h))
        mask_disp = cv2.resize(pred_bin, (w, h), interpolation=cv2.INTER_NEAREST)

        overlay_rgb = raw_rgb.copy()
        overlay_rgb[mask_disp == 1] = (
            overlay_rgb[mask_disp == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
        ).astype(np.uint8)

        heat_u8        = (prob_disp * 255).astype(np.uint8)
        heat_color     = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_color_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        heatmap_rgb    = cv2.addWeighted(raw_rgb, 0.55, heat_color_rgb, 0.45, 0)
        mask_rgb       = cv2.cvtColor(mask_disp * 255, cv2.COLOR_GRAY2RGB)

        wound_area_pct = round(float(mask_disp.mean()) * 100, 2)

    except Exception:
        print("=== INFERENCE FAILED ===", flush=True)
        traceback.print_exc()
        raise

    _log(f"=== INFERENCE COMPLETE (wound {wound_area_pct}%) ===")

    return {
        "overlay":        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
        "mask":           cv2.cvtColor(mask_rgb,    cv2.COLOR_RGB2BGR),
        "heatmap":        cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR),
        "wound_area_pct": wound_area_pct,
    }
