import numpy as np
from rasterio.features import geometry_mask
import geopandas as gpd
from affine import Affine
transform = Affine(1, 0, 0, 0, -1, 0)

# Example polygon
def remove_no_data_predictions(polygons_gdf,intf,th=0.7):


        filtered_geometries = []
        transform = Affine(1, 0, 0, 0, -1, 0)

        for poly in polygons_gdf.geometry:
            # Rasterize the polygon
            polygon_mask = geometry_mask(
                [poly],
                out_shape=intf.shape,
                transform=transform,
                invert=True
            )

            # Extract pixels in the polygon
            values_in_polygon = intf[polygon_mask]

            # Compute the ratio of "no-data" pixels
            matching_pixels = np.sum(values_in_polygon == 0.5)
            total_pixels = np.sum(polygon_mask)

            ratio = matching_pixels / total_pixels if total_pixels > 0 else 0

            # Add polygon to the filtered list if it doesn't exceed the threshold
            if ratio <= th:
                filtered_geometries.append(poly)

        # Create a new GeoDataFrame with filtered polygons
        filtered_gdf = gpd.GeoDataFrame(
            geometry=filtered_geometries,
            crs=polygons_gdf.crs
        )

        return filtered_gdf


