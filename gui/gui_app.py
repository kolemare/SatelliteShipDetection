#!/usr/bin/env python3
"""
Streamlit GUI (organized into classes)
- Pan/zoom a Leaflet map and draw a rectangle AOI
- Choose one of 3 providers: EOX Sentinel-2 Cloudless, Esri World Imagery, OSM
- Choose zoom (quality), output format and name
- Download XYZ tiles for the AOI and stitch into a single image (no upscaling)

Run via:
  python gui.py
(or)
  streamlit run gui/gui_app.py
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw

from providers import ProviderCatalog, ProviderKey
from tiling import AOIRequest, TileStitcher, save_output_image


# ------------ App Composition ------------
@dataclass
class AppConfig:
    repo_root: Path
    downloads_dir: Path

    @classmethod
    def from_repo(cls) -> "AppConfig":
        repo_root = Path(__file__).resolve().parents[1]
        downloads = repo_root / "downloads"
        downloads.mkdir(exist_ok=True)
        return cls(repo_root=repo_root, downloads_dir=downloads)


class MapView:
    """Encapsulates Folium map + draw plugin + provider base layer."""
    def __init__(self, provider, default_center=(20, 0), default_zoom=3):
        self.provider = provider
        self.default_center = default_center
        self.default_zoom = default_zoom

    def render(self):
        m = folium.Map(location=self.default_center, zoom_start=self.default_zoom, tiles=None, control_scale=True)
        # Provider base layer
        folium.TileLayer(
            tiles=self.provider.tiles,
            attr=self.provider.attribution,
            name=self.provider.name,
            max_zoom=self.provider.max_zoom,
            overlay=False,
            control=True,
        ).add_to(m)
        # Optional alternate base
        folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)

        # Draw (rectangle only)
        Draw(
            draw_options={
                "polyline": False, "polygon": False, "circle": False,
                "marker": False, "circlemarker": False, "rectangle": True,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)

        return st_folium(m, height=650, width=None, returned_objects=["last_active_drawing", "all_drawings"])


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

        zoom = st.sidebar.slider("Zoom (higher = more detail)", 1, provider.max_zoom, min(12, provider.max_zoom))

        st.sidebar.markdown("### Output")
        out_fmt = st.sidebar.radio("Format", ["PNG", "JPG"], index=0)
        jpg_q = st.sidebar.slider("JPG quality", min_value=1, max_value=95, value=90)
        out_name = st.sidebar.text_input("Output file name (optional)", "")

        st.sidebar.markdown("---")
        go_btn = st.sidebar.button("⬇️ Download AOI")

        return provider, zoom, out_fmt, jpg_q, out_name, go_btn

    def run(self):
        st.set_page_config(page_title="AOI Downloader", layout="wide")
        st.title("🛰️ Satellite AOI Downloader")

        provider, zoom, out_fmt, jpg_q, out_name, go_btn = self.sidebar()

        st.markdown(
            """
            **Instructions:**  
            - Pan/zoom the map.  
            - Use the **rectangle** tool (🟥 icon) to draw your AOI.  
            - Click **Download AOI** in the sidebar.  
            """
        )

        # Map view for current provider
        map_view = MapView(provider=provider)
        draw_state = map_view.render()

        if go_btn:
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

            st.info(f"Fetching tiles @ z={aoi.zoom} from {provider.name} …")
            progress = st.progress(0)

            def _cb(frac):
                progress.progress(min(100, int(frac * 100)))

            try:
                stitcher = TileStitcher(tile_size=provider.tile_size)
                stitched = stitcher.stitch(aoi, progress_cb=_cb)
                out_path = save_output_image(stitched, aoi, self.cfg.downloads_dir)
                st.success(f"Saved: {out_path}")
                st.image(str(out_path), caption=out_path.name, use_column_width=True)
                with open(out_path, "rb") as f:
                    st.download_button("Download file", f, file_name=out_path.name)
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                progress.progress(100)


def main():
    cfg = AppConfig.from_repo()
    AppController(cfg).run()


if __name__ == "__main__":
    main()
