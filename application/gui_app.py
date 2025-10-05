#!/usr/bin/env python3
"""
Streamlit GUI
- Select an AOI on a map (EOX Sentinel-2, Esri, OSM)
- Download XYZ tiles and stitch them into a raw image (downloads/raw/)
- Automatically run Real-ESRGAN x4 super-resolution (downloads/upscaled/)
- Run ConvNeXt-based ship detection on RAW & UPSCALED and save overlays:
    results/raw/<name>_det.png
    results/upscaled/<name>_det.png
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw

# internal modules (must exist in your repo)
from providers import ProviderCatalog, ProviderKey
from tiling import AOIRequest, TileStitcher, save_output_image
from RRDBNet import SuperSampler

# our inference module
from ConvNextInference import detect_and_draw


# =========================
# App configuration
# =========================
@dataclass
class AppConfig:
    repo_root: Path
    downloads_root: Path
    raw_dir: Path
    upscaled_dir: Path
    results_root: Path
    results_raw: Path
    results_up: Path
    ckpt_path: Path

    @classmethod
    def from_repo(cls) -> "AppConfig":
        repo_root = Path(__file__).resolve().parents[1]  # this file's directory
        downloads_root = repo_root / "downloads"
        raw_dir = downloads_root / "raw"
        upscaled_dir = downloads_root / "upscaled"

        results_root = repo_root / "results"
        results_raw = results_root / "raw"
        results_up = results_root / "upscaled"

        # model checkpoint (from your training script)
        ckpt_path = repo_root / "training" / "pretrained" / "ConvNext" / "convnext_ships.pt"
        print("AAA " + str(ckpt_path))

        raw_dir.mkdir(parents=True, exist_ok=True)
        upscaled_dir.mkdir(parents=True, exist_ok=True)
        results_raw.mkdir(parents=True, exist_ok=True)
        results_up.mkdir(parents=True, exist_ok=True)

        return cls(repo_root, downloads_root, raw_dir, upscaled_dir,
                   results_root, results_raw, results_up, ckpt_path)


# =========================
# Map component
# =========================
class MapView:
    """Folium map + draw plugin + provider base layer."""
    def __init__(self, provider, default_center=(20, 0), default_zoom=3):
        self.provider = provider
        self.default_center = default_center
        self.default_zoom = default_zoom

    def render(self):
        m = folium.Map(location=self.default_center, zoom_start=self.default_zoom, tiles=None, control_scale=True)
        folium.TileLayer(
            tiles=self.provider.tiles,
            attr=self.provider.attribution,
            name=self.provider.name,
            max_zoom=self.provider.max_zoom,
            overlay=False,
            control=True,
        ).add_to(m)
        folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)

        Draw(
            draw_options={
                "polyline": False, "polygon": False, "circle": False,
                "marker": False, "circlemarker": False, "rectangle": True,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)

        return st_folium(m, height=650, width=None, returned_objects=["last_active_drawing", "all_drawings"])


# =========================
# Controller
# =========================
class AppController:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.catalog = ProviderCatalog()

    def _extract_bbox(self, draw_result: dict) -> tuple[float, float, float, float] | None:
        """Return (west, south, east, north) from streamlit-folium draw result."""
        if not draw_result:
            return None
        last = draw_result.get("last_active_drawing")
        if not last:
            all_drawings = draw_result.get("all_drawings") or []
            if not all_drawings:
                return None
            last = all_drawings[-1]
        geom = last.get("geometry")
        if not geom or geom.get("type") != "Polygon":
            return None
        coords = geom["coordinates"][0]
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        west, east = min(lons), max(lons)
        south, north = min(lats), max(lats)
        return (west, south, east, north)

    def sidebar(self):
        st.sidebar.markdown("### Provider & Quality")
        sel = st.sidebar.selectbox(
            "Provider",
            options=[ProviderKey.ESRI, ProviderKey.EOX, ProviderKey.OSM],
            index=0,
            format_func=lambda k: self.catalog.get(k).name,
        )
        provider = self.catalog.get(sel)

        zoom = st.sidebar.slider("Zoom (higher = more detail)", 1, provider.max_zoom, min(15, provider.max_zoom))

        st.sidebar.markdown("### Output")
        out_fmt = st.sidebar.radio("Format", ["PNG", "JPG"], index=0)
        jpg_q = st.sidebar.slider("JPG quality", 1, 95, 90)
        out_name = st.sidebar.text_input("Output file name (optional)", "")

        st.sidebar.markdown("### Detection")
        stride = st.sidebar.select_slider("Sliding-window stride", options=[56, 64, 96, 112, 128, 160, 192], value=112)
        thresh = st.sidebar.slider("Ship threshold", 0.1, 0.95, 0.5, 0.05)

        st.sidebar.markdown("---")
        go_btn = st.sidebar.button("⬇️ Download, Upscale & Detect")

        return provider, zoom, out_fmt, jpg_q, out_name, stride, thresh, go_btn

    def run(self):
        st.set_page_config(page_title="AOI Downloader + Ship Detection", layout="wide")
        st.title("🛰️ AOI Downloader → Real-ESRGAN x4 → ConvNeXt Ship Detection")

        provider, zoom, out_fmt, jpg_q, out_name, stride, thresh, go_btn = self.sidebar()

        st.markdown(
            """
            **Instructions:**  
            1. Draw a rectangle (🟥) on the map.  
            2. Click **Download, Upscale & Detect**.  
            3. Outputs:  
               - Raw image → `downloads/raw/`  
               - Upscaled image → `downloads/upscaled/`  
               - Detections → `results/raw/` and `results/upscaled/`
            """
        )

        map_view = MapView(provider=provider)
        draw_state = map_view.render()

        if go_btn:
            if not self.cfg.ckpt_path.exists():
                st.error(f"Missing checkpoint: {self.cfg.ckpt_path}")
                return

            bbox = self._extract_bbox(draw_state)
            if not bbox:
                st.error("Please draw a rectangle first.")
                return

            west, south, east, north = bbox
            aoi = AOIRequest(
                west=west, south=south, east=east, north=north,
                zoom=zoom,
                out_name=out_name,
                out_format=out_fmt.lower(),
                jpg_quality=jpg_q,
                provider_key=provider.key.value,
                provider_url=provider.tiles,
                provider_attr=provider.attribution
            )

            st.info(f"📡 Fetching tiles @ z={aoi.zoom} from {provider.name} …")
            progress = st.progress(0)

            def _cb(frac):
                progress.progress(min(100, int(frac * 100)))

            try:
                # ---- Download & Stitch ----
                stitcher = TileStitcher(tile_size=provider.tile_size)
                stitched = stitcher.stitch(aoi, progress_cb=_cb)
                raw_path = save_output_image(stitched, aoi, self.cfg.raw_dir)

                st.success(f"Saved RAW image: {raw_path}")
                st.image(str(raw_path), caption="Raw Download", use_column_width=True)

                # ---- Upscale ----
                st.info("🔼 Upscaling with Real-ESRGAN x4 …")
                sr = SuperSampler()
                upscaled_path = sr.process_image(raw_path, self.cfg.upscaled_dir)
                st.success(f"Upscaled image saved: {upscaled_path}")
                st.image(str(upscaled_path), caption="Upscaled (Real-ESRGAN x4)", use_column_width=True)

                # ---- Detection RAW ----
                raw_out = self.cfg.results_raw / f"{Path(raw_path).stem}_det.png"
                st.info("🚢 Detecting ships on RAW …")
                n_raw = detect_and_draw(
                    image_path=Path(raw_path),
                    ckpt_path=self.cfg.ckpt_path,
                    out_path=raw_out,
                    stride=stride,
                    prob_thresh=thresh,
                )
                st.success(f"RAW detections: {n_raw} → {raw_out.name}")
                st.image(str(raw_out), caption=f"Detections on RAW ({n_raw})", use_column_width=True)

                # ---- Detection UPSCALED ----
                up_out = self.cfg.results_up / f"{Path(upscaled_path).stem}_det.png"
                st.info("🚢 Detecting ships on UPSCALED …")
                n_up = detect_and_draw(
                    image_path=Path(upscaled_path),
                    ckpt_path=self.cfg.ckpt_path,
                    out_path=up_out,
                    stride=stride,
                    prob_thresh=thresh,
                )
                st.success(f"UPSCALED detections: {n_up} → {up_out.name}")
                st.image(str(up_out), caption=f"Detections on UPSCALED ({n_up})", use_column_width=True)

                # ---- Downloads ----
                with open(upscaled_path, "rb") as f:
                    st.download_button("Download Upscaled Image", f, file_name=Path(upscaled_path).name, key="dl_up_image")
                with open(raw_out, "rb") as f:
                    st.download_button("Download RAW Detections", f, file_name=raw_out.name, key="dl_raw_det")
                with open(up_out, "rb") as f:
                    st.download_button("Download UPSCALED Detections", f, file_name=up_out.name, key="dl_up_det")

            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                progress.progress(100)


# =========================
# Entry point
# =========================
def main():
    cfg = AppConfig.from_repo()
    AppController(cfg).run()


if __name__ == "__main__":
    main()
