# 🛰️ AOI Downloader & Ship Detection — Real-ESRGAN ×4 + ConvNeXt

An interactive **Streamlit GUI** for selecting an **Area of Interest (AOI)** on a world map, downloading high-resolution map tiles, optionally **upscaling with Real-ESRGAN ×4**, and performing **ship detection** using a **ConvNeXt** model.  
Designed for convenient visual analysis.

---

## 📑 Table of Contents
1. [Overview](#-overview)
2. [Features](#-features)
3. [Repository Structure](#-repository-structure)
4. [Installation](#-installation)
5. [Usage](#-usage)
6. [Model Details](#-model-details)
7. [ConvNeXt Architecture](#-convnext-architecture)
8. [Model Training](#-model-training)
9. [Real-ESRGAN Integration](#-real-esrgan-integration)
10. [Raw vs. RealESRGAN](#-raw-vs-realesrgan)
11. [Results](#-results)
12. [Output Directories](#-output-directories)

---

## 🚀 Overview

This application lets you:

1. Draw an AOI directly on a map (EOX Sentinel-2, Esri, or OSM).  
2. Download and stitch XYZ tiles into a single image.  
3. Optionally run **Real-ESRGAN ×4** super-resolution.  
4. Run **ConvNeXt-based ship detection** on either the RAW or UPSCALED image.  
5. Display both images side-by-side and download the results.

The UI is compact:  
**Left:** interactive map  
**Right:** progress panel  
**Below:** side-by-side input and detection images (no scrolling needed)

---

## ✨ Features

- 🗺️ Draw AOIs directly on interactive map layers  
- 🧩 Automatic tile download and stitching  
- 🔼 Optional Real-ESRGAN ×4 super-resolution  
- 🚢 ConvNeXt-based ship detection  
- ⚙️ Adjustable stride and probability threshold  

---

## 📁 Repository Structure

```
repo_root/
├── application/
│   ├── gui_app.py                 # Main Streamlit GUI entrypoint
│   ├── ConvNextInference.py       # Ship detection logic (detect_and_draw)
│   ├── providers.py               # Map tile providers (EOX, Esri, OSM)
│   ├── tiling.py                  # AOIRequest, TileStitcher, save_output_image
│   └── RRDBNet.py                 # Real-ESRGAN (RRDBNet) wrapper
│
├── training/
│   ├── convnext_train.py          # Training script
│   ├── convnext_base_config.json  # Training config (dataset path, epochs, etc.)
│   ├── training_curves.png        # Accuracy/Loss curves (moved here)
│   └── pretrained/
│       ├── ConvNext/
│       │   ├── convnext_ships.pt          # Trained ConvNeXt checkpoint
│       │   └── metrics.json               # Training metrics (accuracy/loss)
│       │
│       └── RealESRGAN/
│           └── RealESRGAN_x4plus.pth      # Pretrained Real-ESRGAN weights
│
├── scripts/
│   ├── create_venv.sh             # Linux/macOS venv setup
│   └── create_venv.bat            # Windows venv setup
│
├── downloads/                     # Runtime output (raw/upscaled)
├── results/                       # Detection overlays (raw/upscaled)
└── README.md
```

---

## 🧩 Installation

> Requires **Python 3.12+** and an internet connection for wheel downloads.

**Windows (Cmd):**
```
scripts/create_venv.bat
call venv\Scripts\activate.bat
```

**Linux / macOS (Bash):**
```
chmod +x scripts/create_venv.sh
./scripts/create_venv.sh
source venv/bin/activate
```

By default this installs:
- PyTorch 2.7.0 + CUDA 12.8 (auto-fallback to CPU wheels)
- Streamlit / Folium / Real-ESRGAN / OpenCV / TQDM / Matplotlib

| Variable | Default | Description |
|-----------|----------|-------------|
| `CUDA_WHL_TAG` | `cu128` | PyTorch CUDA wheel tag |
| `PYTORCH_VER` | `2.7.0` | PyTorch version |
| `VENV_DIR` | `venv` | Virtual environment folder |

---

## ▶️ Usage

Run the Streamlit app from the repo root:

```
invoke extract
invoke gui
```

Run `invoke extract` to extract model weights and dataset.

### Workflow
1. Draw a rectangular AOI on the map.  
2. In the sidebar:
   - Choose tile provider and zoom.  
   - Toggle **RealESRGAN** if you want ×4 super-resolution.  
   - Set detection stride and threshold.  
3. Click **Download → (Optional) Upscale → Detect**.  
4. Observe:
   - **Left image:** model input (RAW or UPSCALED)  
   - **Right image:** detection overlay  

### Recommended Zoom Levels ESRI
- **With RealESRGAN enabled:** set **zoom = 15** (recommended).  
- **Without RealESRGAN enabled:** set **zoom = 17** (recommended).

---

## 🧠 Model Details

- **Architecture:** ConvNeXt Base (binary classifier: ship / no-ship)  
- **Checkpoint:** `training/pretrained/ConvNext/convnext_ships.pt`  
- **Inference:** Sliding-window detection with adjustable stride and probability threshold  
- **Output:** Number of detected ships and overlay image saved to `results/`

---

## 🧩 ConvNeXt Architecture

ConvNeXt is a modernized convolutional neural network that re-imagines ResNet through the design lens of Vision Transformers (ViTs).  
Key features:

- **Stage-based hierarchical design** (similar to ResNet-50/101).  
- **Large kernel depthwise convolutions (7×7)** for better spatial capture.  
- **LayerNorm** normalization instead of BatchNorm for stability on GPUs.  
- **Inverted bottlenecks** and higher-dimensional expansions inspired by MobileNet V2.  
- **Simplified training pipeline** using standard data augmentations and cosine learning rate decay.

In this project, the **ConvNeXt-Base** variant (≈89 M parameters, pretrained on ImageNet-1K) is fine-tuned for **binary classification (ship vs no-ship)**.  
The original classifier layer is replaced with a new `Linear(in_features, 2)` head.

<p align="center">
  <img src="training/convnext_architecture.png" alt="ConvNeXt Architecture" width="700">
</p>

---

## 🧬 Model Training

The ConvNeXt ship classifier was trained using the script `training/convnext_train.py`.

**Dataset:**  
The network was trained on the *Ships in Satellite Imagery* dataset (`shipsnet`), located under  
`dataset/ships_in_satellite_imagery/shipsnet/shipsnet`.  
Each image file is labeled directly in its filename prefix (`0__...png` for no-ship, `1__...png` for ship).

**Data Processing and Augmentation**
- Images are resized to **224×224 px** using bicubic interpolation.  
- Augmentations include horizontal flips, color jitter (brightness, contrast, saturation), and random erasing (`p=0.25`).  
- Inputs are normalized using ImageNet mean & std statistics.

**Training Configuration**
- 70 % training, 15 % validation, 15 % testing split.  
- Base learning rate `3e-4`, weight decay `1e-4`, optimizer: **AdamW**.  
- Loss function: **Cross-Entropy with label smoothing 0.1**.  
- Scheduler: **Cosine Annealing LR**.  
- **Warm-up training**: the classifier head is trained for 5 epochs with the backbone frozen, after which all layers are unfrozen and trained jointly.  
- Mixed precision (FP16 AMP) on CUDA for faster and more memory-efficient training.  
- Total 30 epochs, batch size 32, seed 42.

**Results and Outputs**
- The model achieving the highest validation accuracy is saved as:  
  `training/pretrained/ConvNext/convnext_ships.pt`
- Metrics (accuracy/loss per epoch) are stored in:  
  `training/pretrained/ConvNext/metrics.json`
- A plot of training curves (`training/training_curves.png`) shows convergence of loss and accuracy across train/val splits.

<p align="center">
  <img src="training/training_curves.png" alt="Training Curves" width="600">
</p>

---

## 🔼 Real-ESRGAN Integration

- Backbone: **RRDBNet (Residual-in-Residual Dense Blocks)**  
- Used when upscaling is selected (once or twice).  
- Saves to `downloads/upscaled/` and runs detection on the final upscaled image.  
- **Checkpoint:** `training/pretrained/RealESRGAN/RealESRGAN_x4plus.pth`  
- CUDA if available; CPU fallback otherwise.

---

## 🧭 Raw vs. RealESRGAN

The ConvNeXt network used for inference expects an input tensor of **224×224×3 (RGB)**.  
Thus, the visible size of ships inside this crop directly affects the network’s ability to extract meaningful spatial features.

### 📉 Raw (Low-Resolution) Input
- **Advantages:**
  - Requires less satellite bandwidth and download time.
  - Covers larger geographic areas per tile.
  - Efficient for wide-area scanning or coarse monitoring.
- **Disadvantages:**
  - Ships may occupy only a few pixels → low feature quality.
  - Small vessels often go undetected due to lack of texture.
  - Increased confusion with background noise (e.g., waves, docks).

### 🔼 Real-ESRGAN Upscaled Input
- **Advantages:**
  - Artificially enhances fine edges and visual detail.
  - Enables ConvNeXt to detect smaller ships that would be invisible in RAW input.
  - Useful when only coarse-zoom tiles (e.g., z14–z16) are available.
- **Disadvantages:**
  - Introduces **GAN artifacts** that may not correspond to real-world structures.
  - Artifacts can mislead the model, producing false positives.
  - Adds 4× computational cost and slower inference.
  - Does not truly increase resolution — it “hallucinates” plausible detail.

### ⚖️ Practical Trade-Off
- **Real-ESRGAN** is ideal when high-zoom satellite tiles are unavailable or bandwidth-limited.  
- However, **true high-resolution imagery** always yields better physical accuracy and fewer false detections.  
- In essence: *Real-ESRGAN enhances perceptual clarity but trades physical fidelity for visual quality.*

---

## 🖼️ Results — ESRI

### ⚓ Detection on RAW (Port of Shanghai)

**Input Image:**  
![Port of Shanghai - Raw](results/port_of_shangai_raw_esri.png)

**Detections:**  
![Port of Shanghai - Detections](results/port_of_shangai_raw_detections_esri.png)

---

### 🌊 Detection with Real-ESRGAN (Ports: Algeciras, Spain & Victoria Harbour, Hong Kong)

Comparison between RAW, UPSCALED (×4), and DETECTION results.

<div align="center">

| Raw Image | Upscaled ×4 | Detections on Upscaled |
|---|---|---|
| ![Port of Algeciras - Raw](results/port_of_aglericas_spain_raw_esri.png) | ![Port of Algeciras - Upscaled](results/port_of_aglericas_spain_upscaled_esri.png) | ![Port of Algeciras - Detections](results/port_of_aglericas_spain_upscaled_detections_esri.png) |
| ![Victoria Harbour - Raw](results/port_victoria_harbour_hong_kong_raw_esri.png) | ![Victoria Harbour - Upscaled](results/port_victoria_harbour_hong_kong_upscaled_esri.png) | ![Victoria Harbour - Detections](results/port_victoria_harbour_hong_kong_upscaled_detections_esri.png) |

</div>

---

## 🖼️ Results — Sentinel-2

### ⚓ Detection on RAW (Port of Shanghai)

**Input Image:**  
![Port of Shanghai - Raw](results/port_of_shangai_raw_sentinel.png)

**Detections:**  
![Port of Shanghai - Detections](results/port_of_shangai_raw_detections_sentinel.png)

---

### 🌊 Detection with Real-ESRGAN (Ports: Algeciras, Spain & Victoria Harbour, Hong Kong)

<div align="center">

| Raw Image | Upscaled ×4 | Detections on Upscaled |
|---|---|---|
| ![Port of Algeciras - Raw](results/port_of_aglericas_spain_raw_sentinel.png) | ![Port of Algeciras - Upscaled](results/port_of_aglericas_spain_upscaled_sentinel.png) | ![Port of Algeciras - Detections](results/port_of_aglericas_spain_upscaled_detections_sentinel.png) |
| ![Victoria Harbour - Raw](results/port_victoria_harbour_hong_kong_raw_sentinel.png) | ![Victoria Harbour - Upscaled](results/port_victoria_harbour_hong_kong_upscaled_sentinel.png) | ![Victoria Harbour - Detections](results/port_victoria_harbour_hong_kong_upscaled_detections_sentinel.png) |

</div>

---

## 📂 Output Directories

| Path | Contents |
|---|---|
| `downloads/raw/` | Stitched AOIs (kept only if **no upscaling**) |
| `downloads/upscaled/` | ×4 (or ×4×4 final) AOIs when upscaling is selected |
| `results/raw/` | Detection overlays for **RAW** images |
| `results/upscaled/` | Detection overlays for **UPSCALED** images |

--- 
