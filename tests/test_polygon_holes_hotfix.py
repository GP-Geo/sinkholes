import geopandas as gpd
import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import MultiPolygon

from sinkholes.polygons import mask_array_to_polygons, pixel_polygons_to_lonlat


@pytest.mark.parametrize("multi", [False, True])
def test_polygon_holes_survive_transform_and_shapefile_export(tmp_path, multi):
    mask = np.ones((5, 5), np.uint8)
    mask[1:4, 1:4] = 0
    ring = mask_array_to_polygons(mask).geometry.iloc[0]
    geometry = MultiPolygon([ring, translate(ring, xoff=10)]) if multi else ring
    source = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
    transformed = pixel_polygons_to_lonlat(source, 35.4, 31.7, .01, .02)
    path = tmp_path / "prediction.shp"
    transformed.to_file(path)
    for frame in (transformed, gpd.read_file(path)):
        geom = frame.geometry.iloc[0]
        parts = list(geom.geoms) if multi else [geom]
        assert len(parts) == (2 if multi else 1)
        assert all(len(p.interiors) == 1 for p in parts)
        assert geom.area == pytest.approx(16 * len(parts) * .01 * .02)
        assert geom.bounds == pytest.approx((35.4, 31.6, 35.55 if multi else 35.45, 31.7))
        assert frame.crs.to_epsg() == 4326
