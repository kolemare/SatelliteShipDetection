#!/usr/bin/env python3
"""
Tile-wise ConvNeXt ship detection overlay (no filesystem I/O).

Returns overlay image (RGB) and number of positive tiles.
Loads weights from convnext_ships.pt in the same folder.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

TILE = 224
STRIDE = 112
PROB_THRESH = 0.50
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def _default_checkpoint_path() -> Path:
    """Weights live in same folder as this file."""
    return Path(__file__).resolve().parent / "convnext_ships.pt"

def _build_model(num_classes: int = 2) -> torch.nn.Module:
    m = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    in_features = m.classifier[2].in_features
    m.classifier[2] = torch.nn.Linear(in_features, num_classes)
    return m

def load_convnext_from_state_dict(ckpt_path: Path | None = None) -> torch.nn.Module:
    ckpt = ckpt_path or _default_checkpoint_path()
    if not ckpt.exists():
        raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt}")
    model = _build_model(2)
    sd = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    return model

def _tile_coords(h: int, w: int, tile: int, stride: int):
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] != h - tile:
        ys.append(max(0, h - tile))
    if xs[-1] != w - tile:
        xs.append(max(0, w - tile))
    return ys, xs

@torch.no_grad()
def _predict_probs(model: torch.nn.Module, batch_t: torch.Tensor) -> torch.Tensor:
    if DEVICE.type == "cuda":
        batch_t = batch_t.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(batch_t)
    else:
        out = model(batch_t.to(DEVICE))
    if out.ndim == 2 and out.size(1) == 2:
        probs = F.softmax(out, dim=1)[:, 1]
    elif out.ndim == 2 and out.size(1) == 1:
        probs = torch.sigmoid(out[:, 0])
    else:
        probs = torch.sigmoid(out.squeeze())
    return probs.float().cpu()

def _draw_positive_tiles(model: torch.nn.Module, img_rgb: np.ndarray,
                         tile: int, stride: int, prob_thresh: float,
                         batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, int]:
    h, w, _ = img_rgb.shape
    ys, xs = _tile_coords(h, w, tile, stride)
    batch_imgs, batch_spans = [], []
    out = img_rgb.copy()
    pos = 0

    def flush():
        nonlocal batch_imgs, batch_spans, out, pos
        if not batch_imgs:
            return
        batch = torch.stack(batch_imgs, 0)
        probs = _predict_probs(model, batch).numpy()
        for (y0, y1, x0, x1), p in zip(batch_spans, probs):
            if p > prob_thresh:
                cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
                pos += 1
        batch_imgs.clear()
        batch_spans.clear()

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            crop = img_rgb[y0:y1, x0:x1, :]
            if (y1 - y0) != tile or (x1 - x0) != tile:
                pad = np.zeros((tile, tile, 3), dtype=crop.dtype)
                pad[: (y1 - y0), : (x1 - x0)] = crop
                crop = pad
            tens = _to_tensor(Image.fromarray(crop))
            batch_imgs.append(tens)
            batch_spans.append((y0, y1, x0, x1))
            if len(batch_imgs) >= batch_size:
                flush()
    flush()
    return out, pos

def detect_and_draw(image_rgb: np.ndarray, ckpt_path: Path | None = None,
                    tile: int = TILE, stride: int = STRIDE,
                    prob_thresh: float = PROB_THRESH) -> tuple[np.ndarray, int]:
    model = load_convnext_from_state_dict(ckpt_path)
    vis, n_pos = _draw_positive_tiles(model, image_rgb, tile, stride, prob_thresh)
    return vis, n_pos
