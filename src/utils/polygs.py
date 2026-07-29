# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import os

from get_intf_info import *
from shapely.geometry import Polygon

import numpy as np
from affine import Affine
import matplotlib.pyplot as plt

import rasterio
from rasterio.features import shapes

import rasterio.features
from shapely.geometry import shape, polygon, box

import geopandas as gpd


def mask_array_to_polygons(mask_array):
    transform = Affine.identity()    # Generate polygons from the mask array
    polygons = shapes(mask_array, transform=transform)
    shapely_polygons = [shape(geom) for geom, value in polygons if value == 1]
    polygons_gpd = gpd.GeoDataFrame(geometry=shapely_polygons,crs="EPSG:4326")
    return polygons_gpd

def plg_indx2longlat(polyg_gdf, intf_coords, x_start=None):
    # x_start: geographic longitude of reconstruction pixel column 0.
    # Should be x0_frame + offset_x*dx (e.g. 35.25 + 3000*dx for South).
    # Falls back to intf_coords[6] (x4000) for legacy callers.
    if x_start is None:
        x_start = intf_coords[6]
    y0, dx, dy = intf_coords[1], intf_coords[2], intf_coords[3]
    polyg_list = polyg_gdf['geometry'].tolist()
    polyg_longlat = []
    for polyg in polyg_list:
        new_coords = [(x_start + x * dx, y0 - y * dy) for x, y in polyg.exterior.coords]
        polyg = Polygon(new_coords)
        polyg_longlat.append(polyg)

    polyg_longlat_gdf = gpd.GeoDataFrame(geometry=polyg_longlat, crs='EPSG:4326')
    return polyg_longlat_gdf
if __name__ == '__main__':
    plt.rcParams['backend'] = 'Qt5Agg'

    # EDIT 2026-07-29: followed test_full_intf.py out of 'pred_outputs2/'. CHANGELOG.md #14
    directory_path = 'outputs/predictions/'
    dir_polygs = directory_path #+ 'polygs/'
    outpath = dir_polygs + 'polygs_longlat/'
    os.makedirs(outpath, exist_ok=True)
    file_list = os.listdir(dir_polygs)
    for file in file_list:
        if file.endswith('.shp'):
            polyg_gdf = gpd.read_file(dir_polygs + file)
            intf = file[:17]
            intf_coords = get_intf_coords(intf)
            polyg_longlat = plg_indx2longlat(polyg_gdf,intf_coords)
            polyg_longlat.to_file(outpath+intf+'_predicted_polygons_longlat.shp')


            polygs_longlat_gdf = gpd.read_file(outpath+intf+'_predicted_polygons_longlat.shp')
            list_of_polygons = polygs_longlat_gdf.geometry.tolist()


            image = np.load( directory_path + intf +'_image.npy', allow_pickle=True)
            fig, ax1 = plt.subplots(1, 1)
            x0, y0, dx, dy, ncells, nlines, x4000, x8500, intf_lidar_mask,num_nz,bo = get_intf_coords(intf)
            extent = [x4000, x4000+dx*image.shape[1], y0 - dy * image.shape[0], y0]

            ax1.imshow(image, extent=extent)
            for p in list_of_polygons:
                x, y = p.exterior.xy
                ax1.plot(x, y)
            plt.show()


