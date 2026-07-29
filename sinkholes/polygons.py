"""Binary masks <-> polygons, and pixel-space -> lon/lat georeferencing."""

from typing import Optional

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import geometry_mask, shapes
from shapely.geometry import Polygon, shape


def mask_array_to_polygons(mask_array: np.ndarray) -> gpd.GeoDataFrame:
    """Connected value-1 regions of a binary mask as pixel-space polygons."""
    polys = [
        shape(geom)
        for geom, value in shapes(mask_array, transform=Affine.identity())
        if value == 1
    ]
    return gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")


def pixel_polygons_to_lonlat(
    polyg_gdf: gpd.GeoDataFrame,
    x_start: float,
    y0: float,
    dx: float,
    dy: float,
) -> gpd.GeoDataFrame:
    """Map pixel-space polygons to lon/lat (EPSG:4326).

    ``x_start`` is the geographic longitude of pixel column 0 of the array the
    polygons were traced on, ``y0`` the latitude of row 0. When the array was
    cropped from a wider scene, x_start must include that offset
    (origin_x + offset_cols * dx) — a wrong value silently shifts every
    exported polygon.
    """
    out = []
    for polyg in polyg_gdf["geometry"]:
        out.append(
            Polygon([(x_start + x * dx, y0 - y * dy) for x, y in polyg.exterior.coords])
        )
    return gpd.GeoDataFrame(geometry=out, crs="EPSG:4326")


def remove_no_data_predictions(
    polygons_gdf: gpd.GeoDataFrame,
    intf: np.ndarray,
    th: float = 0.7,
    nodata_value: float = 0.5,
) -> gpd.GeoDataFrame:
    """Drop pixel-space polygons whose interior is mostly no-data pixels.

    A polygon is kept when at most ``th`` of its pixels equal ``nodata_value``
    (the normalised no-data code).
    """
    transform = Affine(1, 0, 0, 0, -1, 0)
    kept = []
    for poly in polygons_gdf.geometry:
        inside = geometry_mask([poly], out_shape=intf.shape, transform=transform, invert=True)
        total = int(inside.sum())
        ratio = float((intf[inside] == nodata_value).sum()) / total if total else 0.0
        if ratio <= th:
            kept.append(poly)
    return gpd.GeoDataFrame(geometry=kept, crs=polygons_gdf.crs)
