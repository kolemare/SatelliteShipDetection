from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ProviderKey(str, Enum):
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
