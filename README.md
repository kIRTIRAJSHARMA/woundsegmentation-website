# Wound Segmentation — Web App

A Flask website wrapping your trained U-Net (EfficientNet-B4) wound
segmentation model. Upload an image → get the original, overlay, heatmap,
and binary mask, plus a wound-area percentage.

## File structure

```
wound-segmentation-app/
├── app.py                  # Flask server (routes: "/" and "/predict")
├── model.py                 # Loads best_baseline.pth and runs inference
├── requirements.txt
├── model_weights/
│   └── best_baseline.pth    # <-- put your trained checkpoint here
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   ├── uploads/              # user-uploaded images land here
│   └── results/               # generated overlay/mask/heatmap images
└── templates/
    └── index.html
```

## Step 1 — Get your trained weights out of Colab

In your notebook, after training finishes (CELL 38 already does this),
make sure `best_baseline.pth` is saved to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
import os, shutil
os.makedirs('/content/drive/MyDrive/wound_segmentation', exist_ok=True)
shutil.copy('/content/best_baseline.pth',
            '/content/drive/MyDrive/wound_segmentation/best_baseline.pth')
```

Then download that `.pth` file from Google Drive to your computer.

## Step 2 — Open the project in VS Code

1. Unzip the project folder you downloaded from this chat.
2. Open the `wound-segmentation-app` folder in VS Code (`File → Open Folder`).
3. Copy your downloaded `best_baseline.pth` into `model_weights/`.

## Step 3 — Create a virtual environment (VS Code terminal)

Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

VS Code may prompt "Select this interpreter for the workspace" — click **Yes**.

## Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA acceleration, install the matching
CUDA build of torch/torchvision from https://pytorch.org/get-started/locally/
instead of the CPU wheels pip grabs by default — this app runs fine on CPU too,
just slower per image.

## Step 5 — Run the app

```bash
python app.py
```

You should see:
```
✅ Model loaded on cpu   (or cuda)
 * Running on http://127.0.0.1:5000
```

Open **http://127.0.0.1:5000** in your browser. Upload a wound image, click
**Analyze Image**, and you'll see the original, red-highlighted overlay,
probability heatmap, and binary mask side by side.

## How it fits together

- `templates/index.html` + `static/css/style.css` + `static/js/script.js`
  — the page: a drag-and-drop upload box and a results grid. `script.js`
  POSTs the chosen file to `/predict` as `multipart/form-data` and renders
  whatever URLs the server sends back.
- `app.py` — the `/predict` route saves the upload, calls
  `model.run_inference()`, writes the three output images to
  `static/results/`, and returns their URLs as JSON.
- `model.py` — rebuilds the exact `smp.Unet` architecture from your
  notebook (`efficientnet-b4` encoder, `scse` decoder attention),
  loads `best_baseline.pth`, and reproduces the same preprocessing
  (`Resize → Normalize → ToTensorV2`) and post-processing (sigmoid →
  threshold → overlay/heatmap) as your `predict_single_image` function.
  The model is loaded once at startup and cached, so requests after the
  first one are fast.

## Notes / next steps

- **Threshold**: currently fixed at `0.5` in `model.py` (`THRESHOLD`).
  If your notebook's threshold-sweep (CELL 16) found a better value,
  change it there.
- **File size limit**: capped at 10 MB in `app.py`
  (`MAX_CONTENT_LENGTH`) — raise it if needed.
- **Deploying publicly**: `app.run(debug=True)` is for local development
  only. For a real deployment, turn `debug` off and run behind a
  production server, e.g. `gunicorn -w 2 -b 0.0.0.0:5000 app:app`
  (Linux/Mac) or host on Render / Railway / a VPS. GPU inference needs a
  GPU-backed host; CPU is fine for occasional single-image use.
- **Few-shot model (Part 2)**: this app only wires up the baseline U-Net,
  since that's the one meant for direct single-image inference. If you
  want the few-shot support/query flow exposed too, that needs a second
  upload field (support image + support mask) and a second route — say
  the word and I'll add it.
