# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
from rasterio.features import shapes
from affine import Affine
import rasterio
from rasterio.features import rasterize

import rasterio.features
from shapely.geometry import shape, Polygon, mapping

from shapely.geometry import Polygon, LineString

import argparse
from os import listdir, makedirs, path

import numpy as np

import numpy as np
from shapely.geometry import Polygon
from rasterio.features import rasterize
import logging


def count_zero_pixels_inside_polygon(image, shapely_polygon):
    """
    Counts the number of zero-value pixels inside a given Shapely Polygon using rasterization.

    Parameters:
    - image: 2D NumPy array (grayscale image).
    - shapely_polygon: Shapely Polygon object.

    Returns:
    - count: Number of pixels with value 0 inside the polygon.
    """
    # Create a blank mask
    mask = rasterize([(shapely_polygon, 1)], out_shape=image.shape, fill=0, dtype=np.uint8)

    # Count zero pixels inside the polygon
    zero_pixel_count = np.sum(image[mask == 1] == 0)

    return zero_pixel_count


def get_args():

    parser = argparse.ArgumentParser(description='clean edges from mask patches')
    parser.add_argument('--patches_path', type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/' , help='patch H, patch W')

    parser.add_argument('--patch_size', nargs='+', type=int, default=[200, 100], help='patch H, patch W')
    parser.add_argument('--eleven_days_diff', type=str, default='True')
    parser.add_argument('--strides_per_patch', type=int, default=2)
    return parser.parse_args()
if __name__ == '__main__':
    plt.rcParams['backend'] = 'Qt5Agg'

    args = get_args()
    H , W = args.patch_size[0], args.patch_size[1]
    mask_patch_path = args.patches_path + 'mask_patches'+ '_H' + str(H) + '_W' + str(W) +'_strpp' + str(args.strides_per_patch) + '_11days' '/'
    patch_data_path = args.patches_path + 'data_patches'+ '_H' + str(H) + '_W' + str(W) +'_strpp' + str(args.strides_per_patch) + '_11days' '/'
    cleaned_dir = mask_patch_path+'cleaned/'
    cleaned_data_dir = patch_data_path+'cleaned/'
    if os.path.exists(cleaned_dir):
        shutil.rmtree(cleaned_dir)  # Deletes
        # the existing directory and all its contents
    if os.path.exists(cleaned_data_dir):
        shutil.rmtree(cleaned_data_dir)
    # Create a new, empty directory
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(cleaned_data_dir, exist_ok=True)

    files = [item for item in listdir(mask_patch_path) if path.isfile(path.join(mask_patch_path, item))and 'nonz' not in item]
    data_files = ['data'+item[4:] for item in files]

    for n,file in enumerate(files):
        patch_boundary = LineString([(0, 0), (W, 0), (W, H), (0, H), (0, 0)])

        patch_mask = np.load(mask_patch_path + file).astype(np.float32)
        patch_data = np.load(patch_data_path+data_files[n]).astype(np.float32)
        orig_shape = patch_mask.shape
        if len(orig_shape) == 4:
            patch_mask = np.reshape(patch_mask, (orig_shape[0] * orig_shape[1], orig_shape[2], orig_shape[3]))
            patch_data = np.reshape(patch_data, (orig_shape[0]*orig_shape[1], orig_shape[2], orig_shape[3]))
        transform = Affine.identity()  # Create an identity transform
        threshold = 20
        zdata_threshold = 0.7
        topolg_transform = Affine.identity()
        cleaned_data =[]
        cleaned_mask = []
        cleaned_nonz_maks = []
        cleaned_nonz_data = []
        for i in range(patch_mask.shape[0]):
            if not np.any(patch_mask[i]):
                cleaned_mask.append(patch_mask[i])
                cleaned_data.append(patch_data[i])
                continue
            # plt.imshow(patch_data[i,:,:], cmap='grey')
            # plt.show()
            shapes = rasterio.features.shapes(patch_mask[i], patch_mask[i] > 0)
            polygons = [shape(geom) for geom, value in shapes if value > 0]
            polygons_copy = polygons.copy()
            # for p in polygons:
            #     coords_e = p.exterior.coords
            #     coords_list = list(coords_e)
            #     x, y = zip(*coords_list)
            #     plt.gca().set_aspect('equal', adjustable='box')
            #
            #     plt.plot(x,y)
            #     plt.xlim(0,100)
            #     plt.ylim(200,0)
            #
            #     plt.show()
            # plt.show()

            removed = False
            for item in (polygons):
                min_x = min([point[0] for point in item .exterior.coords])
                max_x = max([point[0] for point in item.exterior.coords])
                min_y = min([point[1] for point in item .exterior.coords])
                max_y = max([point[1] for point in item.exterior.coords])

                if count_zero_pixels_inside_polygon(patch_data[i],item)/item.area > zdata_threshold:
                    polygons_copy.remove(item)
                    removed = True

                elif item.area < threshold and item.touches(patch_boundary) or (max_x-min_x<5 and (max_x==W or min_x==1)) or (max_y-min_y<5 and (max_y==H or min_y==1)):
                    polygons_copy.remove(item)
                    removed = True

            transform = Affine.identity()
            # Rasterize the polygons
            if len(polygons_copy)>0:
                cleaned_mask_patch = rasterio.features.rasterize(
                    [(mapping(polygons_copy), 1) for polygons_copy in polygons_copy],
                    out_shape=(H, W),
                    transform=transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8
            )
                cleaned_mask.append(cleaned_mask_patch)
                cleaned_data.append(patch_data[i])
                cleaned_nonz_maks.append(cleaned_mask_patch)
                cleaned_nonz_data.append(patch_data[i])
            else:
                cleaned_mask_patch = np.zeros_like(patch_data[0])
                cleaned_data.append(patch_data[i])
                cleaned_mask.append(cleaned_mask_patch)

            if removed and False:
                fig, [ax1,ax2,ax3] = plt.subplots(1,3, sharex=True,sharey=True,figsize=(10,10))
                ax1.imshow(patch_data[i], cmap='jet')
                ax1.set_title('original mask')
                ax2.imshow(patch_mask[i], cmap='gray')
                ax2.set_title('original mask')
                ax3.imshow(cleaned_mask_patch, cmap='gray')
                ax3.set_title('cleaned mask')
                plt.show()
        cleaned_data = np.array(cleaned_data)
        cleaned_mask  = np.array(cleaned_mask)
        print('finished cleaning inft'+str(n))
        if len(orig_shape) ==4:
            cleaned_mask = np.reshape(cleaned_mask,(orig_shape[0],orig_shape[1],orig_shape[2],orig_shape[3]))
        np.save(mask_patch_path+'cleaned/'+file.split('.')[0] +'_cleaned.npy', cleaned_mask)
        np.save(mask_patch_path+'cleaned/'+'mask_patches_nonz'+file[12:].split('.')[0] +'_cleaned.npy', cleaned_nonz_maks)
        np.save(patch_data_path+'cleaned/'+'data_patches_nonz'+file[12:].split('.')[0] +'_cleaned.npy', cleaned_nonz_data)

        reloaded = np.load(mask_patch_path+'cleaned/'+file.split('.')[0] +'_cleaned.npy')
        reloaded_nonz_data =  np.load(patch_data_path+'cleaned/'+'data_patches_nonz'+file[12:].split('.')[0] +'_cleaned.npy')
        reloaded_nonz_mask =  np.load(mask_patch_path+'cleaned/'+'mask_patches_nonz'+file[12:].split('.')[0] +'_cleaned.npy')
        orig_nonz_mask = np.load(mask_patch_path+'mask_patches_nonz'+file[12:])
        orig_nonz_data = np.load(patch_data_path+'data_patches_nonz'+file[12:])

        print('shape of saved cleaned data is' + str(np.shape(reloaded)))
        print('shape of orig nonz data is' + str(np.shape(orig_nonz_data)))
        print('shape of orig nonz mask is' + str(np.shape(orig_nonz_mask)))
        print('shape of saved reloaded nonz data is' + str(np.shape(reloaded_nonz_data)))
        print('shape of saved reloaded nonz_mask is' + str(np.shape(reloaded_nonz_mask)))
        assert reloaded.shape == orig_shape, 'Not as original!'