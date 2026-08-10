# """
# model.py
# Loads the trained U-Net (EfficientNet-B4 encoder) wound segmentation model
# and runs inference on a single image.
# """

# import os
# import sys
# import urllib.request

# # ── Constants (no heavy imports at module level) ──────────────────────
# IMG_SIZE  = 256
# ENCODER   = "efficientnet-b4"
# IMG_MEAN  = [0.485, 0.456, 0.406]
# IMG_STD   = [0.229, 0.224, 0.225]
# THRESHOLD = 0.5
# DEVICE    = "cpu"

# BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
# WEIGHTS_DIR  = os.path.join(BASE_DIR, "model_weights")
# WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "best_baseline.pth")
# WEIGHTS_URL  = "https://huggingface.co/kirtiraj7/woundseg-model/resolve/main/best_baseline.pth"

# _model         = None
# _val_transform = None


# def download_weights():
#     if os.path.exists(WEIGHTS_PATH):
#         return
#     os.makedirs(WEIGHTS_DIR, exist_ok=True)
#     print(f"Downloading model weights from {WEIGHTS_URL} ...", flush=True)
#     try:
#         urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
#         size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
#         print(f"Model weights downloaded ({size_mb:.1f} MB)", flush=True)
#     except Exception as e:
#         if os.path.exists(WEIGHTS_PATH):
#             os.remove(WEIGHTS_PATH)
#         print(f"ERROR: Failed to download model weights: {e}", flush=True)
#         sys.exit(1)


# def load_model():
#     """Download weights if needed, then load model. All heavy imports deferred here."""
#     global _model, _val_transform
#     if _model is not None:
#         return _model

#     # Heavy imports deferred until first request
#     import torch
#     import segmentation_models_pytorch as smp
#     import albumentations as A
#     from albumentations.pytorch import ToTensorV2

#     download_weights()

#     _val_transform = A.Compose([
#         A.Resize(IMG_SIZE, IMG_SIZE),
#         A.Normalize(mean=IMG_MEAN, std=IMG_STD),
#         ToTensorV2(),
#     ])

#     model = smp.Unet(
#         encoder_name=ENCODER,
#         encoder_weights=None,
#         in_channels=3,
#         classes=1,
#         activation=None,
#         decoder_attention_type="scse",
#     ).to(DEVICE)

#     model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
#     model.eval()

#     _model = model
#     print(f"Model loaded on {DEVICE}", flush=True)
#     return _model


# def run_inference(image_path, threshold=THRESHOLD):
#     import cv2
#     import numpy as np
#     import torch

#     model = load_model()

#     raw = cv2.imread(image_path)
#     if raw is None:
#         raise ValueError(f"Could not read image at {image_path}")
#     raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

#     aug   = _val_transform(image=raw_rgb)
#     img_t = aug["image"].unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         logit = model(img_t)
#     prob    = torch.sigmoid(logit).squeeze().cpu().numpy()
#     pred_bin = (prob > threshold).astype(np.uint8)

#     h, w      = raw_rgb.shape[:2]
#     prob_disp = cv2.resize(prob, (w, h))
#     mask_disp = cv2.resize(pred_bin, (w, h), interpolation=cv2.INTER_NEAREST)

#     overlay_rgb = raw_rgb.copy()
#     overlay_rgb[mask_disp == 1] = (
#         overlay_rgb[mask_disp == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
#     ).astype(np.uint8)

#     heat_u8      = (prob_disp * 255).astype(np.uint8)
#     heat_color   = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
#     heat_color_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
#     heatmap_rgb  = cv2.addWeighted(raw_rgb, 0.55, heat_color_rgb, 0.45, 0)

#     mask_rgb = cv2.cvtColor(mask_disp * 255, cv2.COLOR_GRAY2RGB)

#     wound_area_pct = round(float(mask_disp.mean()) * 100, 2)

#     return {
#         "overlay":        cv2.cvtColor(overlay_rgb,  cv2.COLOR_RGB2BGR),
#         "mask":           cv2.cvtColor(mask_rgb,     cv2.COLOR_RGB2BGR),
#         "heatmap":        cv2.cvtColor(heatmap_rgb,  cv2.COLOR_RGB2BGR),
#         "wound_area_pct": wound_area_pct,
#     }



"""
model.py
Loads the trained U-Net (EfficientNet-B4 encoder)
and runs inference on a single image.
"""

import os
import sys
import urllib.request
import gc

# Keep CPU thread usage low on Render's small instance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

IMG_SIZE = 256
ENCODER = "efficientnet-b4"

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

THRESHOLD = 0.5
DEVICE = "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_DIR = os.path.join(BASE_DIR, "model_weights")
WEIGHTS_PATH = os.path.join(
    WEIGHTS_DIR,
    "best_baseline.pth"
)

WEIGHTS_URL = (
    "https://huggingface.co/kirtiraj7/woundseg-model/"
    "resolve/main/best_baseline.pth"
)

_model = None
_val_transform = None


def download_weights():

    if os.path.exists(WEIGHTS_PATH):
        return

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print(
        f"Downloading model weights from {WEIGHTS_URL}...",
        flush=True
    )

    try:
        urllib.request.urlretrieve(
            WEIGHTS_URL,
            WEIGHTS_PATH
        )

        size_mb = (
            os.path.getsize(WEIGHTS_PATH)
            / (1024 * 1024)
        )

        print(
            f"Model weights downloaded: {size_mb:.1f} MB",
            flush=True
        )

    except Exception as e:

        if os.path.exists(WEIGHTS_PATH):
            os.remove(WEIGHTS_PATH)

        print(
            f"ERROR downloading weights: {e}",
            flush=True
        )

        raise


def load_model():

    global _model
    global _val_transform

    if _model is not None:
        return _model

    print("Starting model loading...", flush=True)

    import torch

    # Limit CPU thread usage
    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    download_weights()

    print("Creating preprocessing pipeline...", flush=True)

    _val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=IMG_MEAN,
            std=IMG_STD
        ),
        ToTensorV2(),
    ])

    print("Creating U-Net model...", flush=True)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
    )

    model = model.to(DEVICE)

    print("Loading checkpoint...", flush=True)

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    # Release checkpoint memory
    del checkpoint
    gc.collect()

    model.eval()

    _model = model

    print(
        "Model loaded successfully on CPU",
        flush=True
    )

    return _model


def run_inference(
    image_path,
    threshold=THRESHOLD
):

    import cv2
    import numpy as np
    import torch

    # Reduce OpenCV thread memory usage
    cv2.setNumThreads(1)

    model = load_model()

    print("Reading image...", flush=True)

    raw = cv2.imread(image_path)

    if raw is None:
        raise ValueError(
            f"Could not read image at {image_path}"
        )

    raw_rgb = cv2.cvtColor(
        raw,
        cv2.COLOR_BGR2RGB
    )

    print("Preprocessing image...", flush=True)

    aug = _val_transform(
        image=raw_rgb
    )

    img_t = (
        aug["image"]
        .unsqueeze(0)
        .to(DEVICE)
    )

    print("Running inference...", flush=True)

    with torch.inference_mode():

        logit = model(img_t)

        prob = (
            torch.sigmoid(logit)
            .squeeze()
            .cpu()
            .numpy()
        )

    # Release tensor memory immediately
    del img_t
    del logit

    gc.collect()

    print("Post-processing...", flush=True)

    pred_bin = (
        prob > threshold
    ).astype(np.uint8)

    h, w = raw_rgb.shape[:2]

    prob_disp = cv2.resize(
        prob,
        (w, h)
    )

    mask_disp = cv2.resize(
        pred_bin,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    overlay_rgb = raw_rgb.copy()

    overlay_rgb[
        mask_disp == 1
    ] = (
        overlay_rgb[
            mask_disp == 1
        ] * 0.5
        + np.array([255, 0, 0]) * 0.5
    ).astype(np.uint8)

    heat_u8 = (
        prob_disp * 255
    ).astype(np.uint8)

    heat_color = cv2.applyColorMap(
        heat_u8,
        cv2.COLORMAP_JET
    )

    heat_color_rgb = cv2.cvtColor(
        heat_color,
        cv2.COLOR_BGR2RGB
    )

    heatmap_rgb = cv2.addWeighted(
        raw_rgb,
        0.55,
        heat_color_rgb,
        0.45,
        0
    )

    mask_rgb = cv2.cvtColor(
        mask_disp * 255,
        cv2.COLOR_GRAY2RGB
    )

    wound_area_pct = round(
        float(mask_disp.mean()) * 100,
        2
    )

    print(
        f"Inference complete - wound area "
        f"{wound_area_pct}%",
        flush=True
    )

    return {
        "overlay": cv2.cvtColor(
            overlay_rgb,
            cv2.COLOR_RGB2BGR
        ),

        "mask": cv2.cvtColor(
            mask_rgb,
            cv2.COLOR_RGB2BGR
        ),

        "heatmap": cv2.cvtColor(
            heatmap_rgb,
            cv2.COLOR_RGB2BGR
        ),

        "wound_area_pct":
            wound_area_pct,
    }