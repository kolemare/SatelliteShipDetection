#!/usr/bin/env python3
"""
Streamlit AOI Downloader (Hi-Res)
- Pan/zoom a Leaflet map, draw a rectangle AOI
- Pick provider (EOX Sentinel-2 cloudless, Esri World Imagery, or custom XYZ)
- Pick zoom (quality), PNG/JPG, JPG quality
- Optional post-process upscale (Lanczos) 1x/2x/4x
- Download tiles for AOI, stitch, upscale (optional), save, and preview

Run:
  python3 -m venv venv && source venv/bin/activate
  pip install streamlit streamlit-folium folium mercantile requests pillow
  streamlit run streamlit_aoi_downloader.py
"""

from __future__ import annotations
import io
import time
from pathlib import Path
from dataclasses import dataclass

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw

import requests
from PIL import Image
import mercantile

# ---------- Config ----------
APP_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = APP_DIR / "downloads"
OUTPUT_DIR.mkdir(exist_ok=True)

# Known providers (XYZ):
PROVIDERS = {
    "EOX Sentinel-2 Cloudless (2019)": {
        "tiles": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2019_3857/default/g/{z}/{y}/{x}.jpg",
        "attr": "&copy; <a href='https://eox.at'>EOX</a> Sentinel-2 Cloudless (2019)",
        "max_zoom": 17,
        "ext": "jpg",
        "tile_size": 256,
    },
    "Esri World Imagery": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        "max_zoom": 19,
        "ext": "jpg",  # actual format can vary; treat as RGB
        "tile_size": 256,
    },
    "OpenStreetMap (for testing)": {
        "tiles": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "&copy; OpenStreetMap contributors",
        "max_zoom": 19,
        "ext": "png",
        "tile_size": 256,
    },
}

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
    post_upscale: int  # 1, 2, 4


# ---------- Helpers ----------
def _bbox_from_draw_result(draw_result: dict) -> tuple[float, float, float, float] | None:
    """
    Extract (west, south, east, north) from streamlit-folium draw result.
    Only rectangle is supported.
    """
    if not draw_result:
        return None
    # New shape just created?
    last = draw_result.get("last_active_drawing")
    if not last:
        # If edited, geometry is in "all_drawings"
        all_drawings = draw_result.get("all_drawings") or []
        if not all_drawings:
            return None
        last = all_drawings[-1]
    geom = last.get("geometry")
    if not geom or geom.get("type") != "Polygon":
        return None
    coords = geom["coordinates"][0]  # rectangle ring: 5 coords (closed)
    # lon, lat pairs. We want min/max.
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)
    return (west, south, east, north)


def fetch_tile(url: str, timeout: int = 30) -> Image.Image:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def stitch_tiles(aoi: AOIRequest, tile_size: int, progress_cb=None) -> Image.Image:
    """
    Download XYZ tiles covering AOI and stitch to a single PIL.Image (RGB).
    """
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

    canvas_w = grid_w * tile_size
    canvas_h = grid_h * tile_size
    canvas = Image.new("RGB", (canvas_w, canvas_h))

    total = len(tiles)
    for i, t in enumerate(tiles, 1):
        # Build URL (support {s} for OSM-ish subdomains)
        if "{s}" in aoi.provider_url:
            # cycle through a,b,c
            sub = ["a", "b", "c"][i % 3]
            url = aoi.provider_url.format(z=t.z, x=t.x, y=t.y, s=sub)
        else:
            url = aoi.provider_url.format(z=t.z, x=t.x, y=t.y)

        tile_img = fetch_tile(url)

        px = (t.x - min_x) * tile_size
        py = (t.y - min_y) * tile_size
        canvas.paste(tile_img, (px, py))

        if progress_cb and (i % 3 == 0 or i == total):
            progress_cb(i / total)

    return canvas


def save_output(img: Image.Image, aoi: AOIRequest) -> Path:
    name = aoi.out_name.strip() or f"aoi_{aoi.provider_key.replace(' ', '_')}_z{aoi.zoom}_{int(time.time())}"
    ext = "jpg" if aoi.out_format.lower() == "jpg" else "png"
    out_path = OUTPUT_DIR / f"{name}.{ext}"

    if ext == "png":
        img.save(out_path, format="PNG", optimize=True)
    else:
        q = max(1, min(95, int(aoi.jpg_quality)))
        img.save(out_path, format="JPEG", quality=q, optimize=True)

    return out_path


# ---------- UI ----------
st.set_page_config(page_title="AOI Downloader (Hi-Res)", layout="wide")
st.title("🛰️ Satellite AOI Downloader (Hi-Res)")

with st.sidebar:
    st.markdown("### Provider & Quality")
    provider_key = st.selectbox("Provider", list(PROVIDERS.keys()), index=1)
    provider = PROVIDERS[provider_key]

    custom_url = ""
    custom_attr = ""
    if provider_key == "Custom XYZ…":
        st.info("Use a template like: https://your.server/tiles/{z}/{x}/{y}.png (supports {s} too)")
        custom_url = st.text_input("Custom XYZ URL", "")
        custom_attr = st.text_input("Attribution (HTML)", "")
        max_zoom = st.slider("Max zoom supported by your server", 1, 22, 19)
    else:
        max_zoom = provider["max_zoom"]

    zoom = st.slider("Zoom (higher = more detail)", 1, max_zoom, min(10, max_zoom))
    post_upscale = st.selectbox("Post-process upscale (Lanczos)", [1, 2, 4], index=1)

    st.markdown("### Output")
    out_fmt = st.radio("Format", ["PNG", "JPG"], index=0)
    jpg_q = st.slider("JPG quality", min_value=1, max_value=95, value=90)
    out_name = st.text_input("Output file name (optional)", "")

    st.markdown("---")
    go_btn = st.button("⬇️ Download AOI")

st.markdown(
    """
**Instructions:**  
- Pan/zoom the map.  
- Use the **rectangle** tool (🟥 icon) to draw your AOI.  
- Click **Download AOI** in the sidebar.  
"""
)

# Build base map to match provider
center = [20, 0]
m = folium.Map(location=center, zoom_start=3, tiles=None, control_scale=True)

if provider_key == "Custom XYZ…":
    tiles_url = custom_url.strip()
    attr = custom_attr.strip()
    if tiles_url:
        folium.TileLayer(tiles=tiles_url, attr=attr or "Custom XYZ", name="Custom XYZ",
                         max_zoom=max_zoom, overlay=False, control=True).add_to(m)
else:
    folium.TileLayer(
        tiles=provider["tiles"],
        attr=provider["attr"],
        name=provider_key,
        max_zoom=provider["max_zoom"],
        overlay=False,
        control=True,
    ).add_to(m)

# Add OSM as optional toggle
folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)

# Draw plugin (rectangle only)
draw = Draw(
    draw_options={
        "polyline": False,
        "polygon": False,
        "circle": False,
        "marker": False,
        "circlemarker": False,
        "rectangle": True,
    },
    edit_options={"edit": True, "remove": True},
)
draw.add_to(m)

# Render map and capture draw events
draw_state = st_folium(m, height=650, width=None, returned_objects=["last_active_drawing", "all_drawings"])

# Handle download
if go_btn:
    bbox = _bbox_from_draw_result(draw_state)
    if not bbox:
        st.error("Please draw a rectangle first.")
    else:
        west, south, east, north = bbox

        # Resolve provider URL/attr/tile_size
        if provider_key == "Custom XYZ…":
            if not custom_url.strip():
                st.error("Custom XYZ URL is empty.")
                st.stop()
            provider_url = custom_url.strip()
            provider_attr = custom_attr.strip()
            tile_size = 256
        else:
            provider_url = provider["tiles"]
            provider_attr = provider["attr"]
            tile_size = provider.get("tile_size", 256)

        aoi = AOIRequest(
            west=west, south=south, east=east, north=north,
            zoom=zoom, out_name=out_name,
            out_format=out_fmt.lower(), jpg_quality=jpg_q,
            provider_key=provider_key, provider_url=provider_url,
            provider_attr=provider_attr, post_upscale=post_upscale,
        )

        st.info(f"Fetching tiles @ z={aoi.zoom} from {aoi.provider_key} …")
        progress_bar = st.progress(0)

        def _cb(frac): progress_bar.progress(min(100, int(frac * 100)))

        try:
            # Stitch tiles
            stitched = stitch_tiles(aoi, tile_size=tile_size, progress_cb=_cb)

            # Optional upscale
            if aoi.post_upscale in (2, 4):
                new_w = stitched.width * aoi.post_upscale
                new_h = stitched.height * aoi.post_upscale
                stitched = stitched.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
                st.info(f"Upscaled to {new_w}×{new_h} (Lanczos).")

            out_path = save_output(stitched, aoi)
            st.success(f"Saved: {out_path}")
            # Show preview (careful on very large upscales)
            st.image(str(out_path), caption=out_path.name, use_column_width=True)
            with open(out_path, "rb") as f:
                st.download_button("Download file", f, file_name=out_path.name)
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            progress_bar.progress(100)
