#!/usr/bin/env python3
"""
Tile-wise ConvNeXt ship detection overlay.

Behavior:
- Slide TILE x TILE window over input with STRIDE.
- For each tile, run classifier; if P(ship) > PROB_THRESH, draw that tile as a box.
- No heatmap or connected components.

Expected structure:
  repo_root/
    ├── application/ConvNextInference.py
    └── pretrained/ConvNext/convnext_ships.pt
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# -------------------------
# Defaults / Hyperparams
# -------------------------
TILE = 224
STRIDE = 112
PROB_THRESH = 0.50        # draw a box if P(ship) > this
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# -------------------------
# Path Helpers
# -------------------------
def _repo_root_from_here() -> Path:
    """repo_root = one level above 'application/'."""
    return Path(__file__).resolve().parents[1]


def _default_checkpoint_path() -> Path:
    """
    Default checkpoint:
      repo_root/pretrained/ConvNext/convnext_ships.pt
    """
    repo_root = _repo_root_from_here()
    return repo_root / "pretrained" / "ConvNext" / "convnext_ships.pt"


# -------------------------
# Model
# -------------------------
def _build_model(num_classes: int = 2) -> torch.nn.Module:
    m = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    in_features = m.classifier[2].in_features
    m.classifier[2] = torch.nn.Linear(in_features, num_classes)
    return m


def load_convnext_from_state_dict(ckpt_path: Path) -> torch.nn.Module:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_path}")
    model = _build_model(2)
    sd = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    return model


# -------------------------
# Sliding Window
# -------------------------
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
    """
    batch_t: (N,3,224,224), normalized
    return: (N,) probability for "ship"
    """
    if DEVICE.type == "cuda":
        batch_t = batch_t.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(batch_t)
    else:
        batch_t = batch_t.to(DEVICE)
        out = model(batch_t)

    # If [N,2] -> softmax, pick class 1 as "ship".
    # If [N,1] -> sigmoid.
    if out.ndim == 2 and out.size(1) == 2:
        probs = F.softmax(out, dim=1)[:, 1]
    elif out.ndim == 2 and out.size(1) == 1:
        probs = torch.sigmoid(out[:, 0])
    else:
        probs = torch.sigmoid(out.squeeze())
    return probs.float().cpu()


def _draw_positive_tiles(
    model: torch.nn.Module,
    img_rgb: np.ndarray,
    tile: int,
    stride: int,
    prob_thresh: float,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, int]:
    """
    Slide a window; draw rectangles where P(ship) > prob_thresh.
    Returns (overlay_image, num_boxes).
    """
    h, w, _ = img_rgb.shape
    ys, xs = _tile_coords(h, w, tile, stride)

    # For speed: batch the tiles; draw after scoring
    batch_imgs: List[torch.Tensor] = []
    batch_spans: List[Tuple[int, int, int, int]] = []

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
                # draw only the real (unpadded) area
                cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(out, f"{p:.2f}", (x0 + 4, max(0, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                pos += 1
        batch_imgs.clear()
        batch_spans.clear()

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            crop = img_rgb[y0:y1, x0:x1, :]

            # If at border, pad to 224×224 for the network
            if (y1 - y0) != tile or (x1 - x0) != tile:
                pad = np.zeros((tile, tile, 3), dtype=crop.dtype)
                pad[: (y1 - y0), : (x1 - x0), :] = crop
                crop = pad

            tens = _to_tensor(Image.fromarray(crop))
            batch_imgs.append(tens)
            # Keep original (un-padded) span for drawing
            batch_spans.append((y0, y1, x0, x1))

            if len(batch_imgs) >= batch_size:
                flush()

    flush()
    return out, pos


def detect_and_draw(
    image_path: Path,
    ckpt_path: Path,
    out_path: Path,
    tile: int = TILE,
    stride: int = STRIDE,
    prob_thresh: float = PROB_THRESH,
) -> int:
    """
    Tile-wise detection; save overlay image; return number of positive tiles.
    """
    model = load_convnext_from_state_dict(ckpt_path)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    vis, n_pos = _draw_positive_tiles(
        model=model,
        img_rgb=img,
        tile=tile,
        stride=stride,
        prob_thresh=prob_thresh,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return n_pos


# -------------------------
# CLI
# -------------------------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image.")
    ap.add_argument("--out", required=True, help="Path to output overlay image.")
    ap.add_argument("--checkpoint", default=None, help="Defaults to ../pretrained/ConvNext/convnext_ships.pt")
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--prob-thresh", type=float, default=PROB_THRESH)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else _default_checkpoint_path()
    if not ckpt.exists():
        print(f"❌ Missing checkpoint: {ckpt}")
        print("Expected default: ../pretrained/ConvNext/convnext_ships.pt")
        raise SystemExit(1)

    n = detect_and_draw(
        image_path=Path(args.image),
        ckpt_path=ckpt,
        out_path=Path(args.out),
        tile=args.tile,
        stride=args.stride,
        prob_thresh=args.prob_thresh,
    )
    print(f"✅ Positive tiles (P> {args.prob_thresh:.2f}): {n} | Device: {DEVICE.type}")


if __name__ == "__main__":
    _cli()
