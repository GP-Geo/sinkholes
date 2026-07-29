# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import argparse
import os
import logging
import sys
import random
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
import numpy as np

def str2bool(arg):
    if arg.lower() == 'true':
        arg = True
    else:
        arg = False
    return arg
def get_args():
    parser = argparse.ArgumentParser(description='create intf partition')
    parser.add_argument('--in_path',  type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/')
    parser.add_argument('--eleven_days_20',  type=str, default='True')
    parser.add_argument('--patch_size',  nargs = '+', type = int, default=[200,100], help='patch H, patch W')
    parser.add_argument('--val_percent',  type=int, help='validation set portion in percents', default=10)



    return parser.parse_args()
if __name__ == '__main__':

    job_time =  datetime.now().strftime("%d_%m_%Hh%M")



    log_file =  'job_intf_partition_' + job_time +'.log'
    file_handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(file_handler)


    random.seed()
    args = get_args()
    args.eleven_days_21 = str2bool(args.eleven_days_21)
    in_path = args.in_path + 'data_patches'+'_H{}'.format(args.patch_size[0]) + '_W{}'.format(args.patch_size[1])+'_11days/' if args.eleven_days_20 else '/'
    if not os.path.exists(in_path):
        logging.info('the patches you requested do not exist - please check for errors or create the data')
        sys.exit()
    full_intf_list = [file[13:30] for file in os.listdir(in_path) if 'nonz' not in file]
    logging.info('full intf list: {}'.format(full_intf_list))
    random.shuffle(full_intf_list)
    logging.info('intf list after shuffle: {}'.format(full_intf_list))
    num_patches = []
    for intf in full_intf_list:
        num_patches.append(np.load(
            in_path + 'data_patches_nonz_' + intf + '_H{}'.format(args.patch_size[0]) + '_W{}'.format(
                args.patch_size[1]) + '.npy').shape[0])

    nonz_patches_sum = sum(num_patches)

    index = np.where((np.cumsum(np.array(num_patches))/nonz_patches_sum)>=(100 - args.val_percent)/100)[0][0]
    val_set = full_intf_list[index+1:]
    train_set = full_intf_list[:index+1]
    partition_file = 'partition_' + job_time + '.json'

    logging.info('\ntrain set: {}'.format(train_set)  +'\nval set: {}'.format(val_set)
                 + '\nval set in intf is {} of full set'.format(len(val_set)/len(full_intf_list))
                 +'\nWriting sets to file: ' + partition_file
                 +'\nJob Info in file: ' + log_file)

    partition = {
        "train": train_set,
        "val": val_set,
        "log file": log_file

    }

    with open(partition_file, 'w') as file:
        json.dump(partition, file, indent=4)

    # Reading back the data to verify
    with open(partition_file, 'r') as file:
        loaded_data = json.load(file)
        logging.info(loaded_data)