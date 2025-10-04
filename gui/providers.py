from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ProviderKey(str, Enum):
    EOX = "eox"
    ESRI = "esri"
    OSM = "osm"


@dataclass(frozen=True)
class Provider:
    key: ProviderKey
    name: str
    tiles: str
    attribution: str
    max_zoom: int
    tile_size: int = 256


class ProviderCatalog:
    """Fixed set of 3 providers, no custom XYZ, no upscaling."""
    def __init__(self):
        self._providers = {
            ProviderKey.ESRI: Provider(
                key=ProviderKey.ESRI,
                name="Esri World Imagery",
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attribution="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
                max_zoom=19,
            ),
            ProviderKey.EOX: Provider(
                key=ProviderKey.EOX,
                name="EOX Sentinel-2 Cloudless (2019)",
                tiles="https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2019_3857/default/g/{z}/{y}/{x}.jpg",
                attribution="© EOX Sentinel-2 Cloudless (2019)",
                max_zoom=17,
            ),
            ProviderKey.OSM: Provider(
                key=ProviderKey.OSM,
                name="OpenStreetMap (for testing)",
                tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                attribution="© OpenStreetMap contributors",
                max_zoom=19,
            ),
        }

    def get(self, key: ProviderKey) -> Provider:
        return self._providers[key]
