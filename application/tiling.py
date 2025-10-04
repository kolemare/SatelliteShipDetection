from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import io
import time
from typing import Callable, Optional

import requests
from PIL import Image
import mercantile


@dataclass
class AOIRequest:
    west: float
    south: float
    east: float
    north: float
    zoom: int
    out_name: str
    out_format: str   # "png" | "jpg"
    jpg_quality: int  # 1..95
    provider_key: str
    provider_url: str
    provider_attr: str


class TileStitcher:
    """Download XYZ tiles covering AOI and stitch to a single PIL.Image (RGB)."""

    def __init__(self, tile_size: int = 256, timeout: int = 30):
        self.tile_size = tile_size
        self.timeout = timeout

    def _fetch_tile(self, url: str) -> Image.Image:
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    def stitch(self, aoi: AOIRequest, progress_cb: Optional[Callable[[float], None]] = None) -> Image.Image:
        west, south, east, north = aoi.west, aoi.south, aoi.east, aoi.north
        z = aoi.zoom

        tiles = list(mercantile.tiles(west, south, east, north, [z]))
        if not tiles:
            raise RuntimeError("No tiles for this AOI at the selected zoom.")

        xs = [t.x for t in tiles]
        ys = [t.y for t in tiles]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        grid_w = max_x - min_x + 1
        grid_h = max_y - min_y + 1

        canvas_w = grid_w * self.tile_size
        canvas_h = grid_h * self.tile_size
        canvas = Image.new("RGB", (canvas_w, canvas_h))

        total = len(tiles)
        for i, t in enumerate(tiles, 1):
            # Build URL (support {s} pattern for OSM-like subdomains)
            if "{s}" in aoi.provider_url:
                sub = ["a", "b", "c"][i % 3]
                url = aoi.provider_url.format(z=t.z, x=t.x, y=t.y, s=sub)
            else:
                url = aoi.provider_url.format(z=t.z, x=t.x, y=t.y)

            tile_img = self._fetch_tile(url)

            px = (t.x - min_x) * self.tile_size
            py = (t.y - min_y) * self.tile_size
            canvas.paste(tile_img, (px, py))

            if progress_cb and (i % 3 == 0 or i == total):
                progress_cb(i / total)

        return canvas


def save_output_image(img: Image.Image, aoi: AOIRequest, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = aoi.out_name.strip() or f"aoi_{aoi.provider_key}_z{aoi.zoom}_{int(time.time())}"
    ext = "jpg" if aoi.out_format.lower() == "jpg" else "png"
    out_path = out_dir / f"{name}.{ext}"

    if ext == "png":
        img.save(out_path, format="PNG", optimize=True)
    else:
        q = max(1, min(95, int(aoi.jpg_quality)))
        img.save(out_path, format="JPEG", quality=q, optimize=True)

    return out_path
