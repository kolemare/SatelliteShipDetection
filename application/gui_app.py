#!/usr/bin/env python3
"""
Streamlit GUI — AOI Downloader + (RealESRGAN x4) + ConvNeXt Detection
---------------------------------------------------------------------
Providers: Esri, OpenStreetMap, Sentinel-2 Cloudless (EOX), Sentinel Hub (auto)
- Sentinel Hub URL is built from a local TOML file relative to THIS script:
    <this_script_dir>/.streamlit/secrets.toml
- Environment variables SENTINEL_HUB_* (if present) override the TOML values.

Example ./.streamlit/secrets.toml:
[SENTINEL_HUB]
INSTANCE_ID = "your_instance_id"
LAYER = "1_TRUE_COLOR"                 # from <ows:Identifier> in your GetCapabilities
STYLE = "default"
FORMAT = "image/jpeg"
TILEMATRIXSET = "PopularWebMercator256"
PREVIEW = 0
SHOWLOGO = false
# Optional time window (leave empty while testing)
TIME_FROM = ""
TIME_TO = ""
# Prefer least-cloudy scenes; cap cloud cover (optional)
PRIORITY = "leastCC"                   # or "mostRecent"
MAXCC = 20
MAX_ZOOM = 18
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import os

# TOML reader: Python 3.11+ has tomllib; otherwise fallback to tomli
try:
    import tomllib  # type: ignore[attr-defined]
    _toml_load = lambda b: tomllib.load(b)
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore
    _toml_load = lambda b: tomllib.load(b)

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw

# internal modules (must exist in your repo)
from providers import ProviderCatalog, ProviderKey, Provider
from tiling import AOIRequest, TileStitcher, save_output_image
from RRDBNet import SuperSampler
from ConvNextInference import detect_and_draw


# ======================================================
# Local secrets loader (RELATIVE to this script)
# ======================================================
def _load_local_secrets_near_app() -> dict:
    """
    Load SENTINEL_HUB section from:
        <this_script_dir>/.streamlit/secrets.toml
    Returns {} if the file/section is missing.
    """
    app_dir = Path(__file__).resolve().parent
    local_path = app_dir / ".streamlit" / "secrets.toml"
    if not local_path.exists():
        return {}
    try:
        with local_path.open("rb") as f:
            data = _toml_load(f)
        section = data.get("SENTINEL_HUB", {})
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


# ======================================================
# Utility: build Sentinel Hub tile template
# ======================================================
def build_sentinel_hub_template(
    instance_id: str,
    layer: str = "1_TRUE_COLOR",
    style: str = "default",
    fmt: str = "image/jpeg",
    tms: str = "PopularWebMercator256",
    time_from: str | None = None,
    time_to: str | None = None,
    preview: int | None = 0,
    showlogo: bool | None = False,
    maxcc: int | None = None,           # NEW
    priority: str | None = None,        # NEW (leastCC or mostRecent)
) -> str:
    """
    Assemble a WMTS KVP URL template that Leaflet/Folium can use with {z}/{x}/{y}.
    """
    base = f"https://services.sentinel-hub.com/ogc/wmts/{instance_id}"
    params = [
        "request=GetTile",
        f"layer={layer}",
        f"style={style}",
        f"tilematrixset={tms}",
        "Service=WMTS",
        "Version=1.0.0",
        f"Format={fmt}",
        "TileMatrix={z}",
        "TileCol={x}",
        "TileRow={y}",
    ]
    if time_from and time_to:
        params.append(f"TIME={time_from}/{time_to}")
    elif time_from:
        params.append(f"TIME={time_from}")

    if maxcc is not None:
        params.append(f"maxcc={int(maxcc)}")
    if priority:
        params.append(f"priority={priority}")  # leastCC / mostRecent

    if preview is not None:
        params.append(f"PREVIEW={int(preview)}")
    if showlogo is not None:
        params.append(f"showlogo={'true' if showlogo else 'false'}")
    return base + "?" + "&".join(params)


# ======================================================
# Sentinel Hub configuration (local TOML + env override)
# ======================================================
def get_sh_config_from_local_or_env() -> dict:
    """
    Order of precedence:
      1) Local TOML: <this_script_dir>/.streamlit/secrets.toml
      2) Environment variables: SENTINEL_HUB_*
    """
    s = _load_local_secrets_near_app()  # purely relative; no Streamlit secrets

    cfg = {
        "INSTANCE_ID": s.get("INSTANCE_ID", ""),
        "LAYER": s.get("LAYER", "1_TRUE_COLOR"),
        "STYLE": s.get("STYLE", "default"),
        "FORMAT": s.get("FORMAT", "image/jpeg"),
        "TILEMATRIXSET": s.get("TILEMATRIXSET", "PopularWebMercator256"),
        "PREVIEW": int(s.get("PREVIEW", 0)) if str(s.get("PREVIEW", "0")).isdigit() else 0,
        "SHOWLOGO": bool(s.get("SHOWLOGO", False)),
        "TIME_FROM": s.get("TIME_FROM", ""),
        "TIME_TO": s.get("TIME_TO", ""),
        "MAX_ZOOM": int(s.get("MAX_ZOOM", 18)) if str(s.get("MAX_ZOOM", "18")).isdigit() else 18,
        # NEW:
        "MAXCC": int(s.get("MAXCC", 0)) if str(s.get("MAXCC", "")).strip() != "" else None,
        "PRIORITY": s.get("PRIORITY", None),
    }

    # Environment overrides
    env = os.environ
    cfg["INSTANCE_ID"]   = env.get("SENTINEL_HUB_INSTANCE_ID", cfg["INSTANCE_ID"])
    cfg["LAYER"]         = env.get("SENTINEL_HUB_LAYER", cfg["LAYER"])
    cfg["STYLE"]         = env.get("SENTINEL_HUB_STYLE", cfg["STYLE"])
    cfg["FORMAT"]        = env.get("SENTINEL_HUB_FORMAT", cfg["FORMAT"])
    cfg["TILEMATRIXSET"] = env.get("SENTINEL_HUB_TILEMATRIXSET", cfg["TILEMATRIXSET"])
    cfg["PREVIEW"]       = int(env.get("SENTINEL_HUB_PREVIEW", cfg["PREVIEW"]))
    cfg["SHOWLOGO"]      = env.get("SENTINEL_HUB_SHOWLOGO", str(cfg["SHOWLOGO"])).lower() in ("1", "true", "yes")
    cfg["TIME_FROM"]     = env.get("SENTINEL_HUB_TIME_FROM", cfg["TIME_FROM"])
    cfg["TIME_TO"]       = env.get("SENTINEL_HUB_TIME_TO", cfg["TIME_TO"])
    cfg["MAX_ZOOM"]      = int(env.get("SENTINEL_HUB_MAX_ZOOM", cfg["MAX_ZOOM"]))

    # NEW overrides
    maxcc_env = env.get("SENTINEL_HUB_MAXCC", None)
    cfg["MAXCC"] = int(maxcc_env) if maxcc_env is not None else cfg["MAXCC"]
    cfg["PRIORITY"] = env.get("SENTINEL_HUB_PRIORITY", cfg["PRIORITY"])

    return cfg


# ======================================================
# App configuration
# ======================================================
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
        # repo root = one level above this script's dir
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        downloads_root = repo_root / "downloads"
        raw_dir = downloads_root / "raw"
        upscaled_dir = downloads_root / "upscaled"

        results_root = repo_root / "results"
        results_raw = results_root / "raw"
        results_up = results_root / "upscaled"

        ckpt_path = repo_root / "training" / "pretrained" / "ConvNext" / "convnext_ships.pt"

        for d in (raw_dir, upscaled_dir, results_raw, results_up):
            d.mkdir(parents=True, exist_ok=True)

        return cls(repo_root, downloads_root, raw_dir, upscaled_dir,
                   results_root, results_raw, results_up, ckpt_path)


# ======================================================
# Map component
# ======================================================
class MapView:
    """Folium map + draw plugin + provider base layer."""
    def __init__(self, provider: Provider, default_center=(20, 0), default_zoom=3):
        self.provider = provider
        self.default_center = default_center
        self.default_zoom = default_zoom

    def render(self):
        m = folium.Map(location=self.default_center, zoom_start=self.default_zoom,
                       tiles=None, control_scale=True)
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
        return st_folium(m, height=560, width=None,
                         returned_objects=["last_active_drawing", "all_drawings"])


# ======================================================
# Controller
# ======================================================
class AppController:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.catalog = ProviderCatalog()

    def _extract_bbox(self, draw_result: dict) -> tuple[float, float, float, float] | None:
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

    def _maybe_build_sentinel_hub_provider(self, base_provider: Provider) -> Provider:
        """
        Build Provider (tiles/max_zoom) from local secrets TOML or env.
        If INSTANCE_ID missing, return base_provider (empty tiles) and caller will fallback to EOX.
        """
        cfg = get_sh_config_from_local_or_env()
        instance = (cfg.get("INSTANCE_ID") or "").strip()
        if not instance:
            st.sidebar.warning(
                "Sentinel Hub not configured. Place secrets at:\n"
                f"{Path(__file__).resolve().parent / '.streamlit' / 'secrets.toml'}\n"
                "or set SENTINEL_HUB_INSTANCE_ID."
            )
            return base_provider  # tiles == "" → caller will fallback to EOX

        url = build_sentinel_hub_template(
            instance_id=instance,
            layer=cfg.get("LAYER", "1_TRUE_COLOR"),
            style=cfg.get("STYLE", "default"),
            fmt=cfg.get("FORMAT", "image/jpeg"),
            tms=cfg.get("TILEMATRIXSET", "PopularWebMercator256"),
            time_from=(cfg.get("TIME_FROM") or None),
            time_to=(cfg.get("TIME_TO") or None),
            preview=cfg.get("PREVIEW", 0),
            showlogo=cfg.get("SHOWLOGO", False),
            maxcc=cfg.get("MAXCC", None),            # NEW
            priority=cfg.get("PRIORITY", None),      # NEW
        )
        return Provider(
            key=base_provider.key,
            name=f"Sentinel Hub — {cfg.get('LAYER', '1_TRUE_COLOR')}",
            tiles=url,
            attribution=base_provider.attribution,
            max_zoom=int(cfg.get("MAX_ZOOM", 18)),
            tile_size=base_provider.tile_size,
        )

    # --------------------------------------------------
    # Sidebar
    # --------------------------------------------------
    def sidebar(self):
        st.sidebar.markdown("### Provider & Quality")
        sel = st.sidebar.selectbox(
            "Provider",
            options=[
                ProviderKey.ESRI,
                ProviderKey.OSM,
                ProviderKey.SENTINEL_EOX,
                ProviderKey.SENTINEL_HUB,
            ],
            index=0,
            format_func=lambda k: self.catalog.get(k).name,
        )
        provider = self.catalog.get(sel)

        # If Sentinel Hub selected, auto-build from local TOML (relative) + env
        if sel == ProviderKey.SENTINEL_HUB:
            # show where we’re reading from
            st.sidebar.caption(
                f"Config path: `{(Path(__file__).resolve().parent / '.streamlit' / 'secrets.toml')}`"
            )
            built = self._maybe_build_sentinel_hub_provider(provider)
            if not built.tiles:
                st.sidebar.info("Falling back to EOX mosaic (z≤13).")
                provider = self.catalog.get(ProviderKey.SENTINEL_EOX)
            else:
                provider = built

            # Quick date override (optional UI)
            with st.sidebar.expander("Date range (optional)", expanded=False):
                today = date.today()
                d_from = st.date_input("From", value=date(today.year, 1, 1))
                d_to = st.date_input("To", value=today)
                cfg = get_sh_config_from_local_or_env()
                instance = (cfg.get("INSTANCE_ID") or "").strip()
                if instance:
                    provider = Provider(
                        key=provider.key,
                        name=provider.name,
                        tiles=build_sentinel_hub_template(
                            instance_id=instance,
                            layer=cfg.get("LAYER", "1_TRUE_COLOR"),
                            style=cfg.get("STYLE", "default"),
                            fmt=cfg.get("FORMAT", "image/jpeg"),
                            tms=cfg.get("TILEMATRIXSET", "PopularWebMercator256"),
                            time_from=d_from.isoformat(),
                            time_to=d_to.isoformat(),
                            preview=cfg.get("PREVIEW", 0),
                            showlogo=cfg.get("SHOWLOGO", False),
                            maxcc=cfg.get("MAXCC", None),
                            priority=cfg.get("PRIORITY", None),
                        ),
                        attribution=provider.attribution,
                        max_zoom=provider.max_zoom,
                        tile_size=provider.tile_size,
                    )

        zoom = st.sidebar.slider(
            "Zoom (higher = more detail)",
            1, provider.max_zoom, min(15, provider.max_zoom)
        )

        st.sidebar.markdown("### Output")
        out_fmt = st.sidebar.radio("Format", ["PNG", "JPG"], index=0, horizontal=True)
        jpg_q = st.sidebar.slider("JPG quality", 1, 95, 90)
        out_name = st.sidebar.text_input("Output file name (optional)", "")

        st.sidebar.markdown("### Processing")
        real_esrgan = st.sidebar.checkbox(
            "RealESRGAN",
            value=True,
            help="If checked: upscale ×4 and detect on UPSCALED (only upscaled kept). "
                 "If unchecked: detect on RAW (only raw kept)."
        )
        stride = st.sidebar.select_slider("Sliding-window stride",
                                          options=[56, 64, 96, 112, 128, 160, 192],
                                          value=112)
        thresh = st.sidebar.slider("Ship threshold", 0.1, 0.95, 0.5, 0.05)

        st.sidebar.markdown("---")
        go_btn = st.sidebar.button(
            "⬇️ Download → (Optional) Upscale → 🚢 Detect",
            use_container_width=True
        )

        return provider, zoom, out_fmt, jpg_q, out_name, real_esrgan, stride, thresh, go_btn

    # --------------------------------------------------
    # Main runner
    # --------------------------------------------------
    def run(self):
        st.set_page_config(page_title="AOI Downloader + (RealESRGAN) + ConvNeXt Detection",
                           layout="wide")
        st.title("🛰 AOI → (Real-ESRGAN ×4) → 🚢 ConvNeXt Ship Detection")

        provider, zoom, out_fmt, jpg_q, out_name, real_esrgan, stride, thresh, go_btn = self.sidebar()
        st.caption("Draw a rectangle on the map, then click the action button. "
                   "Below: Left = model input, Right = detection overlay.")

        top_left, top_right = st.columns([3, 2], gap="large")
        with top_left:
            map_view = MapView(provider=provider)
            draw_state = map_view.render()
        with top_right:
            st.subheader("Progress")
            status_area = st.empty()
            progress_bar = st.progress(0)

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
                zoom=zoom, out_name=out_name, out_format=out_fmt.lower(),
                jpg_quality=jpg_q, provider_key=provider.key.value,
                provider_url=provider.tiles, provider_attr=provider.attribution,
            )

            def _tick(pct, msg):
                progress_bar.progress(min(100, pct))
                status_area.info(msg)

            try:
                _tick(3, f"📡 Fetching tiles @ z={aoi.zoom} from {provider.name} …")
                stitcher = TileStitcher(tile_size=provider.tile_size)
                stitched = stitcher.stitch(aoi,
                                           progress_cb=lambda f: progress_bar.progress(min(100, int(3 + f * 40))))
                _tick(45, "🧵 Writing RAW image …")
                raw_path = save_output_image(stitched, aoi, self.cfg.raw_dir)

                if real_esrgan:
                    _tick(55, "🔼 Upscaling with Real-ESRGAN ×4 …")
                    sr = SuperSampler()
                    upscaled_path = sr.process_image(raw_path, self.cfg.upscaled_dir)
                    input_image_path = Path(upscaled_path)
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

        # --------------------------------------------------
        # Bottom row → input vs result
        # --------------------------------------------------
        bottom_left, bottom_right = st.columns(2, gap="large")
        with bottom_left:
            st.markdown("#### Model Input")
            if input_image_path and Path(input_image_path).exists():
                st.image(str(input_image_path), use_column_width=True)
                with open(input_image_path, "rb") as f:
                    st.download_button(
                        "Download Input Image", f,
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
                        "Download Detection Overlay", f,
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
