# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluate import evaluate
from unet import *
import logging

from attn_unet import *
from device_utils import get_device  # EDIT 2026-07-27, CHANGELOG.md #1
# EDIT 2026-07-28: ConvLSTM U-Net evaluation. CHANGELOG.md #6
# EDIT 2026-07-29: build_convlstm_unet / pop_model_config are no longer called here —
# factory.build_from_checkpoint owns construction. CHANGELOG.md #12
# EDIT 2026-07-29: shared checkpoint -> model factory. CHANGELOG.md #12
from factory import architecture_from_flags, build_from_checkpoint


def str2bool(arg):
    if arg.lower() == 'true':
        arg = True
    else:
        arg = False
    return arg





# Load the test dataset from a pickle file
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='test')


    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--aux_model',type=str,default= None)
    parser.add_argument('--test_intfs', type=str, default='False')
    parser.add_argument('--unet_attn', action='store_true')
    parser.add_argument('--add_attn',action='store_true')
    parser.add_argument('--th', type=float, default=0.7)
    parser.add_argument('--b', type=int, default=5)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--k_prevs', type=int, default=0)
    # EDIT 2026-07-28: ConvLSTM U-Net. CHANGELOG.md #6
    parser.add_argument('--convlstm_unet', action='store_true',
                        help='The checkpoint is a ConvLSTM U-Net (see --convlstm_unet in '
                             'train_sinkholes_unet.py)')
    parser.add_argument('--treat_nodata_regions', action='store_true',
                        help='The checkpoint was trained with validity channels, i.e. '
                             '2 channels per timestep')

    return parser.parse_args()


    return parser.parse_args()
args = get_args()
test_intfs = str2bool(args.test_intfs)

with open(args.test_data_path, 'rb') as file:
    test_data = pickle.load(file)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


if hasattr(test_data, 'ids') and test_intfs:
    test_intfs = test_data.ids
    test_image_data = test_data.image_data
    test_mask_data = test_data.mask_data
    logging.info('test_intfs: {}'.format(test_intfs))


# EDIT 2026-07-27: was `torch.device('cuda' if ... else 'cpu')`; get_device() adds MPS. CHANGELOG.md #1
device = get_device()
logging.info(f'Loading model {args.model}')
logging.info(f'Using device {device}')

# EDIT 2026-07-28: architecture selection is now explicit, and the checkpoint is read
# *before* the model is built so a ConvLSTM U-Net can be rebuilt from the config it
# carries (hidden size, kernel, bilinear, channels per timestep). The UNet and
# AttentionUNet branches construct exactly what they constructed before.
# CHANGELOG.md #6
# EDIT 2026-07-29: the mutual-exclusion check moved into factory.architecture_from_flags
# so every script enforces it identically. CHANGELOG.md #12


# EDIT 2026-07-29: the per-script if/else chain moved into factory.build_from_checkpoint,
# so this script, test_full_intf.py, inspect_run.py and both predict scripts now share one
# construction path. Architecture is detected from the weights; the flags override.
# CHANGELOG.md #12
state_dict = torch.load(args.model, map_location=device)
loaded = build_from_checkpoint(
    state_dict,
    arch=architecture_from_flags(
        convlstm_unet=args.convlstm_unet,
        attn_unet=args.unet_attn,
        add_attn=args.add_attn,
    ),
    n_channels=args.k_prevs + 1,
    n_classes=1,
    bilinear=False,
    treat_nodata_regions=args.treat_nodata_regions,
)
net = loaded.model
mask_values = loaded.mask_values
logging.info(f'Architecture {loaded.architecture} ({loaded.n_channels} in-channels)')
net.to(device=device)
net.load_state_dict(state_dict)
net.eval()
if args.aux_model is not None:
    state_dict = torch.load(args.aux_model, map_location=device)
    net1 = build_net(state_dict)
    net1.to(device=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net1.load_state_dict(state_dict)
    net1.eval()




logging.info('Model loaded!')

test_dir = 'test_data/'

test_dice_score =  evaluate(net, test_loader, device, amp=False, is_local=True, out_path=test_dir,epoch = 1,mode = 'test',net_aux=net,th = args.th,buffer = args.b,plot=args.plot)
logging.info(f'test dice score is: {test_dice_score}')




