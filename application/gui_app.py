#!/usr/bin/env python3
"""
Streamlit GUI (compact, no-scroll workflow)
- Map on the left (EOX/Esri/OSM + Draw rectangle)
- Progress & logs on the right (no side-scrolling needed)
- Below: two columns -> Left = input (RAW or UPSCALED), Right = detection overlay
- Checkbox "RealESRGAN":
    * CHECKED  -> save only UPSCALED image to downloads/, run detection on UPSCALED
    * UNCHECKED-> save only RAW image to downloads/, run detection on RAW
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw

# internal modules (must exist in your repo)
from providers import ProviderCatalog, ProviderKey
from tiling import AOIRequest, TileStitcher, save_output_image
from RRDBNet import SuperSampler
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
        # Repo root = parent of this file's directory
        repo_root = Path(__file__).resolve().parents[1]
        downloads_root = repo_root / "downloads"
        raw_dir = downloads_root / "raw"
        upscaled_dir = downloads_root / "upscaled"

        results_root = repo_root / "results"
        results_raw = results_root / "raw"
        results_up = results_root / "upscaled"

        ckpt_path = repo_root / "training" / "pretrained" / "ConvNext" / "convnext_ships.pt"

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

        # Keep height modest to avoid page scrolling; width auto-fits
        return st_folium(m, height=560, width=None, returned_objects=["last_active_drawing", "all_drawings"])


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
        out_fmt = st.sidebar.radio("Format", ["PNG", "JPG"], index=0, horizontal=True)
        jpg_q = st.sidebar.slider("JPG quality", 1, 95, 90)
        out_name = st.sidebar.text_input("Output file name (optional)", "")

        st.sidebar.markdown("### Processing")
        real_esrgan = st.sidebar.checkbox("RealESRGAN", value=True,
                                          help="If checked: upscale ×4 and detect on UPSCALED (only upscaled is kept). If unchecked: detect on RAW (only raw is kept).")
        stride = st.sidebar.select_slider("Sliding-window stride", options=[56, 64, 96, 112, 128, 160, 192], value=112)
        thresh = st.sidebar.slider("Ship threshold", 0.1, 0.95, 0.5, 0.05)

        st.sidebar.markdown("---")
        go_btn = st.sidebar.button("⬇️ Download → (Optional) Upscale → 🚢 Detect", use_container_width=True)

        return provider, zoom, out_fmt, jpg_q, out_name, real_esrgan, stride, thresh, go_btn

    def run(self):
        st.set_page_config(page_title="AOI Downloader + (RealESRGAN) + ConvNeXt Detection", layout="wide")
        st.title("🛰️ AOI → (Real-ESRGAN x4) → 🚢 ConvNeXt Ship Detection")

        provider, zoom, out_fmt, jpg_q, out_name, real_esrgan, stride, thresh, go_btn = self.sidebar()

        # Compact instructions (no tall blocks)
        st.caption("Draw a rectangle on the map, then click the action button in the sidebar. Below the map: "
                   "Left = model input (RAW or UPSCALED), Right = detection overlay.")

        # --- TOP ROW: Map (left) + Progress panel (right) ---
        top_left, top_right = st.columns([3, 2], gap="large")

        with top_left:
            map_view = MapView(provider=provider)
            draw_state = map_view.render()

        with top_right:
            st.subheader("Progress")
            status_area = st.empty()
            progress_bar = st.progress(0)

        # --- ACTION ---
        input_image_path = None
        result_image_path = None

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

            def _tick(pct: int, msg: str):
                progress_bar.progress(min(100, pct))
                status_area.info(msg)

            try:
                # ---- Download & Stitch ----
                _tick(3, f"📡 Fetching tiles @ z={aoi.zoom} from {provider.name} …")
                stitcher = TileStitcher(tile_size=provider.tile_size)
                stitched = stitcher.stitch(aoi, progress_cb=lambda f: progress_bar.progress(min(100, int(3 + f * 40))))

                _tick(45, "🧵 Writing RAW image …")
                raw_path = save_output_image(stitched, aoi, self.cfg.raw_dir)

                # ---- Optional Upscale ----
                if real_esrgan:
                    _tick(55, "🔼 Upscaling with Real-ESRGAN x4 …")
                    sr = SuperSampler()
                    upscaled_path = sr.process_image(raw_path, self.cfg.upscaled_dir)
                    input_image_path = Path(upscaled_path)

                    # Delete RAW so only UPSCALED remains in downloads/
                    try:
                        os.remove(raw_path)
                    except Exception:
                        pass

                    _tick(70, "🚢 Running detection on UPSCALED …")
                    out_det = self.cfg.results_up / f"{Path(upscaled_path).stem}_det.png"
                else:
                    input_image_path = Path(raw_path)
                    _tick(60, "🚢 Running detection on RAW …")
                    out_det = self.cfg.results_raw / f"{Path(raw_path).stem}_det.png"

                # ---- Detection ----
                n_found = detect_and_draw(
                    image_path=input_image_path,
                    ckpt_path=self.cfg.ckpt_path,
                    out_path=out_det,
                    stride=stride,
                    prob_thresh=thresh,
                )
                result_image_path = out_det
                _tick(95, f"✅ Detection complete: {n_found} ships")

                _tick(100, "✔️ Done")

            except Exception as e:
                status_area.error(f"❌ Error: {e}")
                progress_bar.progress(100)

        # --- BOTTOM ROW: Input (left) | Result (right) ---
        bottom_left, bottom_right = st.columns(2, gap="large")
        with bottom_left:
            st.markdown("#### Model Input")
            if input_image_path and Path(input_image_path).exists():
                st.image(str(input_image_path), use_column_width=True)
                with open(input_image_path, "rb") as f:
                    st.download_button(
                        "Download Input Image",
                        f,
                        file_name=Path(input_image_path).name,
                        key="dl_input_img",
                        use_container_width=True
                    )
            else:
                st.info("Input image will appear here.")

        with bottom_right:
            st.markdown("#### Detection Result")
            if result_image_path and Path(result_image_path).exists():
                st.image(str(result_image_path), use_column_width=True)
                with open(result_image_path, "rb") as f:
                    st.download_button(
                        "Download Detection Overlay",
                        f,
                        file_name=Path(result_image_path).name,
                        key="dl_det_img",
                        use_container_width=True
                    )
            else:
                st.info("Detection overlay will appear here.")


# =========================
# Entry point
# =========================
def main():
    cfg = AppConfig.from_repo()
    AppController(cfg).run()


if __name__ == "__main__":
    main()
