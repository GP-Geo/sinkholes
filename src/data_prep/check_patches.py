# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import argparse
from os import listdir
import numpy as np
import json
import logging
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--input_patch_dir', type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/data_patches_H200_W100_strpp2_11days/',
                    help='full unw files directory for input')
with open (_bootstrap.asset('intf_coord.json'), 'r') as f:
    intf_mdata = json.load(f)
for intf in intf_mdata:
    if 'nonz_num' not in intf_mdata[intf]:
        intf_mdata[intf]['nonz_num'] ='none'

args = parser.parse_args()
intf_path = args.input_patch_dir+'data_patches_nonz'
intfs = [[file,file[len('data_patches_nonz_'):len('data_patches_nonz_')+17]]for file in listdir(args.input_patch_dir) if 'nonz' in file]
for item in intfs:
    file_name = item[0]
    intf_name = item[1]
    data = np.load(args.input_patch_dir+file_name)
    nonz_patches_num = data.shape[0]
    if intf_mdata[intf_name]['north'] >31.5:
        intf_part = 'north_part'
    else:
        intf_part = 'south_part'

    logging.info(f'{intf_name} is {intf_part} and has {nonz_patches_num} nonz' )
    intf_mdata[intf_name]['nonz_num'] = nonz_patches_num
with open(_bootstrap.asset('intf_coord.json'), 'w') as file:
    json.dump(intf_mdata, file,indent=4)
