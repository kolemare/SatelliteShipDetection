#!/usr/bin/env python3
"""
Real-ESRGAN x4 super-sampling wrapper (no file I/O).
Weights are expected next to this file: RealESRGAN_x4plus.pth
"""

from __future__ import annotations
from pathlib import Path
import types, sys
import torch
import numpy as np

# -------------------------------------------------------------------------
# 🔧 Torchvision compatibility shim:
# Newer torchvision removed `torchvision.transforms.functional_tensor`.
# basicsr/realesrgan still import from there, e.g.:
#   from torchvision.transforms.functional_tensor import rgb_to_grayscale
# We inject a module with that name that forwards to the current API.
# -------------------------------------------------------------------------
try:
    import torchvision.transforms.functional as _TF
    # If the old submodule doesn't exist, create it and expose needed symbols.
    import importlib
    try:
        importlib.import_module("torchvision.transforms.functional_tensor")  # type: ignore
    except ModuleNotFoundError:
        _shim = types.ModuleType("torchvision.transforms.functional_tensor")
        # Map the old symbol(s) to the new location
        _shim.rgb_to_grayscale = _TF.rgb_to_grayscale  # type: ignore[attr-defined]
        # Register shim so downstream imports succeed
        sys.modules["torchvision.transforms.functional_tensor"] = _shim
except Exception as _e:
    # If torchvision isn't available at import time, let the normal error surface later
    pass

from realesrgan.utils import RealESRGANer  # type: ignore
from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore


class SuperSampler:
    def __init__(self, device: str | None = None,
                 weights_path: str | Path | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device:
            self.device = device

        weights = Path(weights_path) if weights_path else Path(__file__).resolve().parent / "RealESRGAN_x4plus.pth"
        if not weights.exists():
            raise FileNotFoundError(f"❌ RealESRGAN weights not found: {weights}")

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        self.enhancer = RealESRGANer(
            scale=4,
            model_path=str(weights),
            model=model,
            tile=256,
            tile_pad=10,
            pre_pad=10,
            half=(self.device == "cuda"),
            device=torch.device(self.device),
        )

    def enhance_np(self, img_bgr: np.ndarray) -> np.ndarray:
        """Input BGR np.ndarray, returns enhanced BGR np.ndarray."""
        with torch.inference_mode():
            out, _ = self.enhancer.enhance(img_bgr, outscale=4)
        return out
