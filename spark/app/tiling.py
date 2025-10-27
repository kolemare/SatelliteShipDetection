#!/usr/bin/env python3
"""
Tile fetching & stitching for XYZ providers + AOI container.

Adds AOIRequest.from_wkt(...) so the streaming job can construct AOIs
directly from a POLYGON WKT string without touching the filesystem.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import io
import re
import time
from typing import Callable, Optional, Tuple

import requests
from PIL import Image
import mercantile


# --------------------------- WKT helpers -------------------------------------


def _bounds_from_wkt(wkt: str) -> Tuple[float, float, float, float]:
    """
    Return (west, south, east, north) bounds parsed from a POLYGON(...) WKT.
    Tries shapely if available; otherwise falls back to a lightweight parser.
    """
    # 1) Try shapely if present (most robust)
    try:
        from shapely import wkt as _wkt  # type: ignore
        geom = _wkt.loads(wkt)
        west, south, east, north = geom.bounds
        return float(west), float(south), float(east), float(north)
    except Exception:
        pass

    # 2) Fallback: extract coordinate pairs from a simple POLYGON WKT
    #    Handles: POLYGON((x1 y1,x2 y2,...))
    m = re.search(r"POLYGON\s*\(\(\s*(.*?)\s*\)\)", wkt, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError("Unsupported or invalid WKT (expected POLYGON).")
    coord_blob = m.group(1)
    pts: list[Tuple[float, float]] = []
    for token in coord_blob.split(","):
        token = token.strip()
        parts = token.split()
        if len(parts) < 2:
            continue
        x, y = float(parts[0]), float(parts[1])
        pts.append((x, y))
    if not pts:
        raise ValueError("No coordinates found in POLYGON WKT.")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


# ------------------------------ AOI ------------------------------------------


@dataclass(frozen=True)
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

    @classmethod
    def from_wkt(cls, bbox_wkt: str, zoom: int, provider) -> "AOIRequest":
        """
        Build from POLYGON WKT and a provider object (from providers.ProviderCatalog().get(...)).
        The provider object must expose: key, tiles, attribution.
        """
        west, south, east, north = _bounds_from_wkt(bbox_wkt)
        # provider.key may be an Enum; keep its string value if so
        try:
            pkey = provider.key.value  # Enum[str].value
        except Exception:
            pkey = str(provider.key)
        return cls(
            west=west, south=south, east=east, north=north,
            zoom=int(zoom),
            out_name="",
            out_format="png",
            jpg_quality=90,
            provider_key=str(pkey),
            provider_url=str(provider.tiles),
            provider_attr=str(provider.attribution),
        )


# --------------------------- Stitcher ----------------------------------------


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


# --------------------------- Optional save util -------------------------------


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
