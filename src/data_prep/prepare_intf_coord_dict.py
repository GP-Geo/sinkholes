# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
from os import listdir
import argparse
import json
from get_intf_info import *
import logging


parser = argparse.ArgumentParser(description='Prepare patches of intrfrgrm data')

parser.add_argument('--intf_dir', type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/')
parser.add_argument('--out_dir', type=str, default='./')

args = parser.parse_args()
intf_dict = dict()
for file in listdir(args.intf_dir):
    if file.endswith('.ers'):
        print(file)
        with open(args.intf_dir + file, 'r') as f:
            for line in f:
                for line in f:
                    if 'NrOfLines' in line:
                        NLINES = int(line.strip().split()[-1])
                    if 'NrOfCellsPerLine' in line:
                        NCELLS = int(line.strip().split()[-1])
                    if 'Northings' in line:
                        y0 = float(line.strip().split()[-1])
                    if 'Eastings' in line:
                        x0 = float(line.strip().split()[-1])
                    if 'Ydimension' in line:
                        dY = float(line.strip().split()[-1])
                    if 'Xdimension' in line:
                        dX = float(line.strip().split()[-1])
                    if 'ByteOrder' in line:
                        byte_order = line.strip().split()[-1]
        intfrgrm_name = file.split('.')[0][9:17] + file.split('.')[0][24:33]
        intf_lidar_mask = get_intf_lidar_mask(intfrgrm_name)
        print(intf_lidar_mask)
        if intf_lidar_mask == 'no_mask':
            logging.info('Note: No LiDAR mask for {}. Please decide what todo with it.'.format(intfrgrm_name))
        intf_dict[intfrgrm_name] = {'north': y0,'east':x0, 'nlines': NLINES, 'ncells':NCELLS, 'dy' : dY, 'dx':dX,'byte_order':byte_order, 'lidar_mask':intf_lidar_mask}


with open(args.out_dir + 'intf_coord.json', 'w') as json_file:
    json.dump(intf_dict, json_file, indent=4)###