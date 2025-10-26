from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ProviderKey(str, Enum):
    ESRI = "esri"
    OSM = "osm"
    SENTINEL_EOX = "sentinel_eox"
    SENTINEL_HUB = "sentinel_hub"  # auto from local ./.streamlit/secrets.toml (relative to gui script)


@dataclass(frozen=True)
class Provider:
    key: ProviderKey
    name: str
    tiles: str
    attribution: str
    max_zoom: int
    tile_size: int = 256


class ProviderCatalog:
    """
    Providers:
      • Esri World Imagery (z≤19)
      • OpenStreetMap (z≤19)
      • Sentinel-2 Cloudless (EOX, mosaic up to z≈13)
      • Sentinel Hub (tiles/zoom assembled by the GUI from local secrets/env)
    """
    def __init__(self):
        self._providers = {
            ProviderKey.ESRI: Provider(
                key=ProviderKey.ESRI,
                name="Esri World Imagery",
                tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
                       "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
                attribution=("Tiles © Esri — Source: Esri, Maxar, Earthstar "
                             "Geographics, and the GIS User Community"),
                max_zoom=19,
            ),
            ProviderKey.OSM: Provider(
                key=ProviderKey.OSM,
                name="OpenStreetMap",
                tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                attribution="© OpenStreetMap contributors",
                max_zoom=19,
            ),
            ProviderKey.SENTINEL_EOX: Provider(
                key=ProviderKey.SENTINEL_EOX,
                name="Sentinel-2 Cloudless (EOX, z≤13)",
                tiles=("https://tiles.maps.eox.at/wmts"
                       "?layer=s2cloudless_3857&style=default"
                       "&tilematrixset=PopularWebMercator256"
                       "&Service=WMTS&Request=GetTile&Version=1.0.0"
                       "&Format=image/jpeg"
                       "&TileMatrix={z}&TileCol={x}&TileRow={y}"),
                attribution=("Sentinel-2 Cloudless © EOX IT Services GmbH (CC BY 4.0); "
                             "contains modified Copernicus Sentinel data."),
                max_zoom=13,
            ),
            # NOTE: The GUI will construct the tiles URL and max_zoom for Sentinel Hub.
            ProviderKey.SENTINEL_HUB: Provider(
                key=ProviderKey.SENTINEL_HUB,
                name="Sentinel Hub (auto from local secrets)",
                tiles="",
                attribution="Contains Copernicus Sentinel data; served via Sentinel Hub.",
                max_zoom=18,
            ),
        }

    def get(self, key: ProviderKey) -> Provider:
        return self._providers[key]
