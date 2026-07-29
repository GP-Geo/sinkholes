# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import geopandas as gpd
import pandas as pd

mask2019 = gpd.read_file('LiDAR2019_polyg.shp')
mask2020 = gpd.read_file('LiDAR2020_polyg.shp')
mask2021 = gpd.read_file('LiDAR2021_polyg.shp')
mask2022 = gpd.read_file('LiDAR2022_polyg.shp')

mask2019['source'] = 'LiDAR2019'
mask2020['source'] = 'LiDAR2020'
mask2021['source'] = 'LiDAR2021'
mask2022['source'] = 'LiDAR2022'
united_gdf = pd.concat([mask2019, mask2020, mask2021, mask2022], ignore_index=True)
united_gdf.to_file(_bootstrap.asset('lidar_mask_polygs.shp'))

# Verify the result
print(united_gdf)