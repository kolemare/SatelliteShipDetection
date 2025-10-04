#!/usr/bin/env python3
"""
Real-ESRGAN x4 (RRDBNet) super-sampling.

- Automatically uses GPU if available, otherwise falls back to CPU.
- Uses only *local* pretrained weights (no download).

Programmatic use:
  from application.realESRGAN import SuperSampler
  ss = SuperSampler()  # auto GPU/CPU
  out_path = ss.process_image("../downloads/raw/aoi.png", "../downloads/upscaled")
"""

from __future__ import annotations
from pathlib import Path
import cv2
import torch


# -------------------------------------------------------------------------
# Patch basicsr for torchvision >= 0.15
# -------------------------------------------------------------------------
def _patch_basicsr_import():
    try:
        import importlib.util
        spec = importlib.util.find_spec("basicsr")
        if not spec or not spec.origin:
            return
        p = Path(spec.origin).parent / "data" / "degradations.py"
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            bad = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
            good = "from torchvision.transforms.functional import rgb_to_grayscale"
            if bad in txt and good not in txt:
                print("🔧 Patching basicsr import in degradations.py …")
                p.write_text(txt.replace(bad, good), encoding="utf-8")
    except Exception:
        # Silent best-effort; if patch fails, imports will error as usual.
        pass


_patch_basicsr_import()
# -------------------------------------------------------------------------

from realesrgan.utils import RealESRGANer  # type: ignore
from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore


class SuperSampler:
    """
    Real-ESRGAN x4 wrapper using local pretrained weights only.
    Automatically uses GPU if available.
    """

    def __init__(
        self,
        device: str | None = None,
        weights_path: str | Path = "training/pretrained/RealESRGAN/RealESRGAN_x4plus.pth",
        tile: int = 256,
        tile_pad: int = 10,
        pre_pad: int = 10,
    ):
        # Auto-detect GPU if available
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        self.weights_path = Path(weights_path).resolve()

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"❌ RealESRGAN weights not found at: {self.weights_path}\n"
                "Place RealESRGAN_x4plus.pth there (relative to application/)."
            )

        print(f"🧠 Using device: {self.device.upper()}")

        # RRDBNet config for RealESRGAN_x4plus
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )

        # GPU/CPU inference (half=True only if GPU)
        use_half = self.device == "cuda"
        self.enhancer = RealESRGANer(
            scale=4,
            model_path=str(self.weights_path),
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=use_half,
            device=torch.device(self.device),
        )

    def process_image(self, in_path: str | Path, out_dir: str | Path) -> Path:
        """Upscale a single image ×4 and save into out_dir with `_x4` suffix."""
        in_path = Path(in_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {in_path}")

        with torch.inference_mode():
            output, _ = self.enhancer.enhance(img, outscale=4)

        out_path = out_dir / f"{in_path.stem}_x4{in_path.suffix}"
        cv2.imwrite(str(out_path), output)
        print(f"✅ Saved super-sampled image: {out_path}")
        return out_path


# Standalone CLI use (optional)
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python realESRGAN.py <input_image> <output_dir>")
        sys.exit(1)

    inp = Path(sys.argv[1])
    outd = Path(sys.argv[2])

    ss = SuperSampler()  # auto GPU/CPU
    ss.process_image(inp, outd)
