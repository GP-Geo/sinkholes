# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import argparse
import logging
import os
import random
import sys

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
#from data_loading import *
#from mac_gpu import *
import io
from get_intf_info import *
import os
from torch.utils.data import Subset
from device_utils import get_device, memory_format_for  # EDIT 2026-07-27, CHANGELOG.md #1


# import wandb
from evaluate import evaluate
from unet import UNet
# EDIT 2026-07-28: temporal architecture, opt-in via --convlstm_unet. CHANGELOG.md #6
from convlstm_unet import CONFIG_KEY as CONVLSTM_CONFIG_KEY, ConvLSTMUNet
from sinkholes_data_loading import *
import logging
from dice_score import dice_loss

from datetime import datetime
import pickle
from attn_unet import *
import torch.nn.functional as F

# EDIT 2026-07-27: Ultralytics-style CLI reporter. CHANGELOG.md #3
import math
import time
from train_reporter import (
    BestTracker,
    EpochTable,
    ResultsCSV,
    banner,
    human_time,
    plot_curves,          # EDIT 2026-07-27, CHANGELOG.md #5
    quiet_root_console,
    save_prediction_grid,  # EDIT 2026-07-27, CHANGELOG.md #5
    setup_logger,
)

EPS = 1e-6

# EDIT 2026-07-27: one logger name shared by __main__ and train_model; setup_logger
# is idempotent, so the second call returns the already-configured logger.
# CHANGELOG.md #3
REPORTER_NAME = 'sinkholes.train'

def masked_bce_with_logits(logits, y_float, V, pos_w: float = 1.0):
    """
    logits:  (B,1,H,W)
    y_float: (B,1,H,W) in {0,1}
    V:       (B,1,H,W) validity mask in {0,1}
    pos_w:   >1.0 increases loss on positive pixels
    """
    pw = torch.as_tensor([pos_w], device=logits.device, dtype=logits.dtype)  # per-class (C=1)
    per_pix = F.binary_cross_entropy_with_logits(
        logits, y_float, reduction='none', pos_weight=pw
    )
    w = V.detach()
    return (per_pix * w).sum() / w.sum().clamp(min=1.0)

def masked_dice_loss_binary(logits, y_float, V):
    """
    Soft Dice on probs with validity mask.
    """
    p = torch.sigmoid(logits)                      # (B,1,H,W)
    num = (2.0 * (p * y_float * V)).sum()
    den = ((p * V).sum() + (y_float * V).sum() + EPS)
    return 1.0 - (num / den)

def masked_ce_plus_softdice_multiclass(logits, y_long, V):
    """
    CrossEntropy (masked) + mean soft Dice over classes (masked).
    logits: (B,C,H,W), y_long: (B,H,W), V: (B,1,H,W)
    """
    # CE (per-pixel) then mask
    ce_per_pix = F.cross_entropy(logits, y_long, reduction='none')   # (B,H,W)
    ce = (ce_per_pix * V.squeeze(1)).sum() / (V.sum().clamp(min=1.0))

    # Soft Dice over classes
    probs = F.softmax(logits, dim=1)                      # (B,C,H,W)
    y1 = F.one_hot(y_long, num_classes=logits.shape[1]).permute(0,3,1,2).float()  # (B,C,H,W)

    probs = probs * V                                     # mask both
    y1    = y1 * V

    num = (2.0 * (probs * y1)).sum(dim=(0,2,3))           # per-class
    den = (probs.sum(dim=(0,2,3)) + y1.sum(dim=(0,2,3)) + EPS)
    dice = 1.0 - (num / den).mean()                       # mean over classes

    return ce + dice

# EDIT 2026-07-27: the per-batch loss, lifted verbatim out of the training loop
# so validation can report the *same* objective. Previously the loss existed only
# inline in the loop, which meant a comparable `val/loss` could not be produced
# without duplicating it — and a duplicate silently drifts the first time either
# branch is tuned. Numerics are unchanged, including the hardcoded `pos_w=8.0`
# on the masked path (which deliberately differs from `--pos_w`) and the
# r_tol/lam constants. CHANGELOG.md #5
def segmentation_loss(logits, images, true_masks, model, args, criterion):
    """Loss for one batch.

    `images` is (B, T, H, W), or (B, 2T, H, W) when `--treat_nodata_regions` is
    on, in which case the second half is the per-time validity mask; `V_any` is
    derived here rather than passed in so the function is self-contained and the
    validation caller does not have to reconstruct it. `true_masks` may arrive as
    (B, H, W) from the training loop or (B, 1, H, W) from `evaluate`.
    """
    if true_masks.dim() == 4 and true_masks.shape[1] == 1:
        true_masks = true_masks.squeeze(1)  # evaluate() unsqueezes before calling

    if args.treat_nodata_regions:
        T = images.shape[1] // 2
        V = images[:, T:, ...]                     # (B,T,H,W), 1=valid, 0=nodata
        V_any = V.max(dim=1, keepdim=True).values  # (B,1,H,W)
    else:
        V_any = None

    if model.n_classes == 1:
        # Binary segmentation
        y_float = true_masks.float().unsqueeze(1)  # (B,1,H,W)

        if args.treat_nodata_regions:
            loss_bce = masked_bce_with_logits(logits, y_float, V_any, pos_w=8.0)
            loss_dice = masked_dice_loss_binary(logits, y_float, V_any)

            # no-data FP suppression (outside GT, with tolerance)
            r_tol = 2  # pixels; try 1–3
            lam = 0.2  # weight; tune 0.1–0.5

            y_dil = F.max_pool2d(y_float, kernel_size=2 * r_tol + 1, stride=1,
                                 padding=r_tol)  # tolerate small holes
            M_fp = (1.0 - V_any) * (1.0 - y_dil)  # penalize invalid pixels not near GT

            if M_fp.sum() > 0:
                bce_fp = F.binary_cross_entropy_with_logits(
                    logits, torch.zeros_like(y_float), reduction='none'
                )
                bce_fp = (bce_fp * M_fp).sum() / M_fp.sum().clamp(min=1.0)
            else:
                bce_fp = logits.new_tensor(0.0)

            return loss_bce + loss_dice + lam * bce_fp

        # Unmasked baseline losses
        loss = criterion(logits.squeeze(1), y_float.squeeze(1))
        loss += dice_loss(torch.sigmoid(logits.squeeze(1)), y_float.squeeze(1), multiclass=False)
        return loss

    # Multiclass segmentation
    if args.treat_nodata_regions:
        # Expect masked CE + Soft Dice to broadcast V_any (B,1,H,W) across classes
        return masked_ce_plus_softdice_multiclass(logits, true_masks, V_any)
    return criterion(logits, true_masks)


def str2bool(arg):
    if arg.lower() == 'true':
        arg = True
    else:
        arg = False
    return arg
def train_model(
        args,
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        test_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0

):
    # 1. Create dataset
    patch_size = tuple(args.patch_size)
    H, W = patch_size

    image_dir = args.patches_dir + 'data_patches_H' + str(H) + '_W' + str(W) +'_strpp'+str(args.stride) + ('_11days' if args.train_on_11d_diff else '_all')
    mask_dir = args.patches_dir + 'mask_patches_H' + str(H) + '_W' + str(W)  +'_strpp'+str(args.stride) + ('_11days' if args.train_on_11d_diff else '_all')
    #
    if args.add_temporal or True:
        image_dir += '_Aligned'
        mask_dir += '_Aligned'

    if args.use_cleaned_patches:
        image_dir += '/cleaned'
        mask_dir += '/cleaned'

        #sys.exit(0)
    logging.info('input patch directories: {} and {}'.format(image_dir, mask_dir))
    assert os.path.exists(image_dir) and os.path.exists(image_dir), 'The data you are requesting does not exist, please check if you prepared it at the preparation stage'
    if args.nonz_only and args.partition_mode != 'spatial' and not args.add_temporal:
        pref, mask_pref = 'data_patches_nonz_', 'mask_patches_nonz_'
    else:
        pref, mask_pref = 'data_patches_', 'mask_patches_'
    start_intf_name = len(pref)
    intf_list = [
        file.split('.')[0][start_intf_name:start_intf_name + 17]
        for file in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, file)) and '.json' not in file and (
                ('nonz' in file and args.nonz_only and args.partition_mode != 'spatial'and not args.add_temporal) or
                ('nonz' not in file and (not args.nonz_only or args.partition_mode == 'spatial'or args.add_temporal))
        )
    ]
    with open(args.intf_dict_path, 'r') as json_file:
        coord_dict = json.load(json_file)
    tmp_list = []
    for intf in intf_list:
        if coord_dict[intf]['nonz_num'] != 'none':
            tmp_list.append(intf)
    intf_list = tmp_list

    n1 = len(intf_list)
    logging.info('Original list has {} nonz'.format(n1))
    if args.train_with_nonz_th:
        nonz_th_north, nonz_th_south = tuple(args.nonz_th)[0], tuple(args.nonz_th)[1]
        filtered_list = []
        for intf in intf_list:
            if (coord_dict[intf]['north'] > 31.5 and coord_dict[intf]['nonz_num'] > nonz_th_north) or (
                    coord_dict[intf]['north'] < 31.5 and coord_dict[intf]['nonz_num'] > nonz_th_south):
                filtered_list.append(intf)
        intf_list = filtered_list
        n2 = len(intf_list)
        logging.info('filtered list has {} nonz'.format(n2))
        logging.info('filtered by nonz threshold: removed {} intfs'.format(n1-n2))
    if args.add_temporal or True:
        with open(args.intf_dict_path, "r") as f:
            intf_info = json.load(f)
        prev_dict,updated_intf_list = find_11day_sequences(intf_info,k_prev=args.k_prevs, restrict_to=intf_list)
        intf_list = updated_intf_list
    if args.partition_mode == 'random_by_patch':
        logging.info('Creating Dataset: Randlomly partitioning by patches')


        if args.nonoverlap_tr_tst:
            train_set = SubsiDataset(args,image_dir,mask_dir,intf_list,dset='train')
            test_set = SubsiDataset(args,image_dir,mask_dir,intf_list,dset='test')
            val_set = test_set
            n_train = len(train_set)
            n_val = len(val_set)
            n_test = len(test_set)
        else:
            dataset = SubsiDataset(args,image_dir,mask_dir,intf_list)
            n_total = int(len(dataset))
            n_val = int(n_total * val_percent)
            n_test = int(n_total * test_percent)
            n_train = n_total - n_val - n_test
            train_set, temp_set = random_split(dataset, [n_train, n_total - n_train], generator=torch.Generator().manual_seed(0))
            val_set, test_set = random_split(temp_set, [n_val, n_test], generator=torch.Generator().manual_seed(0))

        logging.info('train val and test sets have {}, {}, {} samples'.format(len(train_set), len(val_set), len(test_set)) )
        buffer = io.BytesIO()
        pickle.dump(test_set, buffer)

        # Get the size in bytes
        size_in_bytes = buffer.tell()

        # Convert bytes to gigabytes
        size_in_gb = size_in_bytes / (1024 ** 3)

        logging.info(f"Estimated size of test_Dataset: {size_in_gb:.2f} GB")


        with open(outpath+'test_dataset_'+args.job_name+'.pkl', 'wb') as f:
            pickle.dump(test_set, f)

    elif args.partition_mode == 'random_by_intf':
        logging.info('Creating Dataset: Randomly partitioning by Interferograms !!!')
        unique_intf_list = intf_list


        if args.test_data_to_exclude is not None:
            with open(args.test_data_to_exclude, 'rb') as file:
                test_data = pickle.load(file)
            test_list = test_data.ids
        else:
            test_list = None

         # test_list = ['20190504_20190515','20191210_20191221','20200613_20200624','20201115_20201126','20210326_20210406','20210510_20210521']#hack
        if test_list is None:
            random.shuffle(unique_intf_list)
            n_val = int(len(unique_intf_list)*(val_percent))
            n_test = int(len(unique_intf_list)*(test_percent))
            n_train = len(unique_intf_list) - n_val - n_test
            if n_val == 0:
                logging.info('not enough data for partitioning by interferograms !!')
                sys.exit(0)
            train_list = unique_intf_list[:n_train]
            val_list = unique_intf_list[n_train:n_train+n_val]
            test_list = unique_intf_list[n_train+n_val:]

        else:
            tv_list = list(set(unique_intf_list) - set(test_list))
            logging.info(f' train + val sets include {len(tv_list)} intfs')
            logging.info(f' val percent is  {val_percent}')
            random.shuffle(tv_list)
            n_val = int(len(unique_intf_list) * (val_percent))
            n_train = len(tv_list) - n_val
            if n_val == 0:
                logging.info('not enough data for partitioning by interferograms !!')
                sys.exit(0)

            train_list = tv_list[:n_train]
            val_list = tv_list[n_train:]
        if args.preset_test_val_21:#hack
            val_list = ['20210407_20210418','20210418_20210429','20210304_20210315','20210326_20210406']
            test_list = ['20210806_20210817','20210623_20210704','20210622_20210703','20210714_20210725']


            train_list = list(set(unique_intf_list) - set(test_list)-set(val_list))
            # tmp_train_list = []
            # for intf in train_list:
            #     if intf[:4] != '2021':
            #         tmp_train_list.append(intf)
            # train_list = tmp_train_list

            logging.info(f' test interferograms: {test_list}')
            logging.info(f'val interferograms: {val_list}')
            logging.info(f'train interferograms: {train_list}')
        if is_running_locally and True: #hack for local debug
            train_list = ['20190204_20190215']
            test_list = ['20190205_20190216']
            val_list = ['20190216_20190227']



        train_set = SubsiDataset(args,image_dir,mask_dir,intrfrgrm_list=train_list,dset = 'train',seq_dict=prev_dict)
        val_set = SubsiDataset(args,image_dir,mask_dir,intrfrgrm_list=val_list,dset = 'val',seq_dict=prev_dict)
        test_set = SubsiDataset(args,image_dir, mask_dir, intrfrgrm_list=test_list,dset = 'test',seq_dict=prev_dict)
        logging.info('train intfs: ' + str(train_set.ids))
        logging.info('val intfs: ' + str(val_set.ids))
        logging.info('test intfs:' + str(test_set.ids))
        logging.info('num of intfs in train: ' + str(len(train_set.ids)))
        logging.info('num of intfs in val: ' + str(len(val_set.ids)))
        logging.info('num of intfs in test: ' + str(len(test_set.ids)))
        if (set(train_set.ids)).isdisjoint(set(val_set.ids)):
            logging.info( 'no commom train val intfs')
        if (set(train_set.ids)).isdisjoint(set(test_set.ids)):
            logging.info('no commom train test intfs')


        assert set(train_set.ids).isdisjoint(set(val_set.ids)) and set(train_set.ids).isdisjoint(set(test_set.ids)) and set(val_set.ids).isdisjoint(
            set(test_set.ids)), 'there are common intfs in lists!'

        logging.info('train val and test sets have {}, {}, {} samples'.format(len(train_set), len(val_set), len(test_set)))
        buffer = io.BytesIO()
        pickle.dump(test_set, buffer)

        # Get the size in bytes
        size_in_bytes = buffer.tell()

        # Convert bytes to gigabytes
        size_in_gb = size_in_bytes / (1024 ** 3)

        logging.info(f"Estimated size of test_Dataset: {size_in_gb:.2f} GB")

        with open(outpath+'test_dataset_'+args.job_name+'.pkl', 'wb') as f:
            pickle.dump(test_set, f)

    elif args.partition_mode == 'spatial':
        logging.info('Creating Dataset: by spatial partitioning !!!')
        if is_running_locally:
            intf_list = ['20191129_20191210']
        train_set = SubsiDataset(args,image_dir,mask_dir,intf_list, dset = 'train',augment = args.augment,seq_dict=prev_dict)
        valtmp_set = SubsiDataset(args,image_dir,mask_dir,intf_list, dset = 'val',seq_dict=prev_dict)
        n_train = len(train_set)
        n_valtmp = len(valtmp_set)
        n_val = n_valtmp//2
        n_test = n_valtmp - n_val
        val_set,test_set = random_split(valtmp_set, [n_val,n_test])
        # val_indices = list(range(n_val))  # First n_val samples → validation
        # test_indices = list(range(n_val, n_valtmp))  # The rest → test
        #
        # # Define subsets
        # val_set = Subset(valtmp_set, val_indices)
        # test_set = Subset(valtmp_set, test_indices)

        logging.info('train val and test sets have {}, {}, {} samples'.format(len(train_set), len(val_set), len(test_set)) )
        logging.info('Spatial partitioning: Val percent is ' + str(int(100*n_val/(n_train+n_val+n_test))) + '%')
        logging.info('Spatial partitioning: test percent is ' + str(int(100*n_test/(n_train+n_val+n_test))) + '%')
        with open(outpath+'test_dataset_'+args.job_name+'.pkl', 'wb') as f:
            pickle.dump(test_set, f)

    elif args.partition_mode == 'preset_by_intf':
        logging.info('Creating Dataset: preset partition by intf \n partiotion file: '+args.partition_file)
        with open(args.partition_file, 'r') as file:
            loaded_data = json.load(file)
        train_list = loaded_data['train']
        val_list = loaded_data['val']
        train_set = SubsiDataset(args, image_dir, mask_dir, intrfrgrm_list=train_list)
        val_set = SubsiDataset(args, image_dir, mask_dir, intrfrgrm_list=val_list)
        n_train = len(train_set)
        n_val = len(val_set)

    #     train_set, val_set = get_preset_partition()
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    val_loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_loader_args)


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_w], device=device))
    global_step = 0

    # EDIT 2026-07-27: reporter set-up — table, results.csv, best tracking.
    # All of this is skipped when --reporter False. CHANGELOG.md #3
    # EDIT 2026-07-27: sample counts, not the branch-dependent n_train/n_val.
    # In `random_by_intf` those hold *interferogram* counts, so the old
    # `tqdm(total=n_train)` showed 0/24 while counting to 7705 — the progress
    # percentage was meaningless. CHANGELOG.md #3
    n_train_samples = len(train_set)
    n_val_samples = len(val_set)
    # `preset_by_intf` never builds a test split, so this stays 'n/a' there.
    n_test_samples = len(test_set) if 'test_set' in locals() else 'n/a'

    rep = setup_logger(REPORTER_NAME) if args.reporter else None
    table = results_csv = None
    # EDIT 2026-07-27: best tracking and early stopping are NOT tied to
    # --reporter. A run needs its best weights whether or not the fancy console
    # is on, so the tracker is built unconditionally. CHANGELOG.md #4
    tracker = BestTracker('val/dice', mode='max', patience=args.patience)
    if rep is not None:
        # Pre-flight warnings the old output buried or never emitted at all.
        if device.type == 'cpu':
            rep.warning('No accelerator found — training on CPU. '
                        'Expect ~10x slower epochs than MPS/CUDA.')
        if n_val_samples == 0:
            rep.warning('Validation split is empty — no Dice will be computed '
                        'and best-checkpoint tracking is disabled.')
        # Resolved values, not raw CLI args — so silently-overridden defaults
        # (LOCAL_ENVIRONMENT's split override, the k_prevs chain filter, the
        # forced _Aligned suffix) are visible here rather than buried.
        # EDIT 2026-07-28: ConvLSTMUNet added to the arch label. CHANGELOG.md #6
        arch = 'ConvLSTMUNet' if args.convlstm_unet else (
            'AttentionUNet' if args.attn_unet else (
                'UNet+attn' if args.add_attn else 'UNet'))
        banner(
            rep,
            f'Sinkholes {arch} — {args.job_name}',
            device=device.type,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patch_size=f'{H}x{W} (stride {args.stride})',
            # EDIT 2026-07-28: for ConvLSTMUNet, spell out channels-PER-TIMESTEP and the
            # sequence length separately — they are different things. CHANGELOG.md #6
            channels=(
                f'{model.n_channels_per_timestep} ch/timestep x T={args.k_prevs + 1} '
                f'/ {model.n_classes} out (ConvLSTM hidden='
                f'{model.convlstm_hidden_channels}, k={model.convlstm_kernel_size})'
                if args.convlstm_unet else
                f'{model.n_channels} in / {model.n_classes} out'
                f' (k_prevs={args.k_prevs}, temporal={args.add_temporal})'
            ),
            target=('union over stack (legacy)' if args.union_temporal_mask
                    else 'latest timestep') if args.add_temporal else 'single intf',
            partition=args.partition_mode,
            train_val_test=f'{n_train_samples} / {n_val_samples} / '
                           f'{n_test_samples} patches',
            pos_weight=args.pos_w,
            patch_dir=image_dir,
            intf_dict=args.intf_dict_path,
            local_override=bool(is_running_locally),
            output_dir=outpath,
        )
        # EDIT 2026-07-27: added val/loss, val/IoU and val/F1. Existing readers key
        # off column names (inspect_run.py:44), so appending is safe. CHANGELOG.md #5
        columns = ['epoch', 'time', 'train/loss', 'val/loss', 'val/dice',
                   'val/IoU', 'val/F1', 'val/P', 'val/R', 'lr']
        table = EpochTable(columns, rep, header_every=25)
        results_csv = ResultsCSV(Path(outpath) / 'results.csv', columns)
        rep.info(f'Logging results to {outpath}results.csv')
        rep.info(f'Starting training for {epochs} epochs...')

    run_start = time.time()
    best_path = Path(dir_checkpoint) / 'best.pt'
    last_path = Path(dir_checkpoint) / 'last.pt'  # EDIT 2026-07-27, CHANGELOG.md #4
    interrupted = False
    last_epoch = 0

    #Begin training
    try:
      for epoch in range(1, epochs + 1):
        last_epoch = epoch
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        n_batches = 0
        with (tqdm(total=n_train_samples, desc=f'{epoch}/{epochs}', unit='img',
                   leave=False, disable=False) as pbar):
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                # images: (B, 2T, H, W) if args.treat_nodata_regions else (B, T, H, W)
                # EDIT 2026-07-28: for ConvLSTMUNet, dim 1 is TIME (times channels per
                # timestep), not a fixed channel count — T is not an architectural
                # parameter — so the flat-channel assert does not apply. CHANGELOG.md #6
                if isinstance(model, ConvLSTMUNet):
                    assert images.dim() == 4, (
                        f'ConvLSTMUNet expects a 4D batch (B, T*C, H, W) from the loader, '
                        f'got shape {tuple(images.shape)}.')
                    assert images.shape[1] % model.n_channels_per_timestep == 0, (
                        f'Loaded images have {images.shape[1]} channels, which is not '
                        f'divisible by {model.n_channels_per_timestep} channels per '
                        f'timestep (shape {tuple(images.shape)}). Check --k_prevs and '
                        f'--treat_nodata_regions.')
                else:
                    assert images.shape[1] == model.n_channels, \
                        f'Network has {model.n_channels} input channels, but loaded images have {images.shape[1]}.'

                # EDIT 2026-07-27: memory format now matches the model's (see CHANGELOG.md #1)
                images = images.to(device=device, dtype=torch.float32, memory_format=memory_format_for(device))
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    logits = model(images)  # (B,1,H,W) for binary; (B,C,H,W) for multiclass

                # EDIT 2026-07-27: the loss body moved to segmentation_loss() so
                # validation reports the same objective. CHANGELOG.md #5
                loss = segmentation_loss(logits, images, true_masks, model, args, criterion)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                loss_val = loss.item()
                # EDIT 2026-07-27: warn once per epoch on a non-finite loss, via
                # the reporter so tqdm.write() parks the bar first. CHANGELOG.md #3
                if not math.isfinite(loss_val):
                    if rep is not None and n_batches == 0:
                        rep.warning(f'Non-finite loss ({loss_val}) at step {global_step}, '
                                    f'epoch {epoch} — check the learning rate.')
                else:
                    epoch_loss += loss_val
                    n_batches += 1
                logging.info( '\n step: {}\n epoch: {}\n train loss: {:.8f} \n '.format(global_step,epoch,loss_val))

                pbar.set_postfix(**{'loss (batch)': loss_val})

        Path(dir_validation).mkdir(parents=True, exist_ok=True)
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)


        logging.info( 'Validation Round: {}'.format(epoch))
        # EDIT 2026-07-27: `val_metrics` collects pixel P/R for the table. Opt-in
        # on evaluate()'s side, so other callers are unaffected. CHANGELOG.md #3
        val_metrics = {} if args.reporter else None
        # EDIT 2026-07-27: ask evaluate() for the validation loss (same objective
        # as training) and for a few patches to render. Both are opt-in; when the
        # reporter is off nothing extra is computed. CHANGELOG.md #5
        val_loss_fn = (lambda lg, im, gt: segmentation_loss(lg, im, gt, model, args, criterion)) \
            if args.reporter else None
        want_samples = (args.reporter and args.sample_every
                        and (epoch % args.sample_every == 0 or epoch == 1))
        # Channel T-1 is the current interferogram (prevs…present ordering,
        # sinkholes_data_loading.py:176); any later channels are validity masks.
        # Without --add_temporal there is only one channel whatever --k_prevs says,
        # matching how num_c is built at :829-832.
        _cur_ch = args.k_prevs if args.add_temporal else 0
        val_samples = {'n': args.n_samples, 'channel': _cur_ch} if want_samples else None
        val_score = evaluate(model, val_loader, device, amp, is_local=is_running_locally,out_path=str(dir_validation),epoch = epoch,save_val = args.save_val,metrics_out=val_metrics,loss_fn=val_loss_fn,samples_out=val_samples)
        scheduler.step(val_score)

        # EDIT 2026-07-27: qualitative grid for this epoch. Plot failures are
        # logged and swallowed — a picture is not worth losing the run over.
        # CHANGELOG.md #5
        if val_samples and 'image' in val_samples:
            try:
                png = save_prediction_grid(
                    val_samples, Path(dir_validation) / 'preds' / f'epoch_{epoch:03d}.png',
                    epoch=epoch)
                if png is not None and epoch == 1:
                    rep.info(f'  validation samples → {png.parent}') if rep else None
            except Exception as exc:  # noqa: BLE001 - never fail a run on plotting
                (rep or logging).warning(f'Could not save the sample grid: {exc}')

        logging.info('Epoch {} Validation Dice score: {}'.format(epoch,val_score))
        if save_checkpoint:
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.dataset.mask_values if args.partition_mode == 'random_by_patch' and not args.nonoverlap_tr_tst else train_set.mask_values
            # EDIT 2026-07-28: ConvLSTMUNet checkpoints carry the config needed to rebuild
            # the exact architecture at inference. Stock UNet checkpoints are byte-for-byte
            # unchanged — this key is only added for the new model. CHANGELOG.md #6
            if isinstance(model, ConvLSTMUNet):
                state_dict[CONVLSTM_CONFIG_KEY] = model.config_dict()
            # EDIT 2026-07-27: --save_best_only skips the 124 MB per-epoch dump.
            # best.pt and last.pt are still written, so nothing is lost except the
            # ability to pick an arbitrary middle epoch. CHANGELOG.md #4
            if not args.save_best_only:
                torch.save(state_dict, str(dir_checkpoint / (args.job_name + 'checkpoint_epoch{}.pth'.format(epoch))))
                logging.info(f'Checkpoint {epoch} saved!')
            # `last.pt` always mirrors the most recent epoch.
            torch.save(state_dict, str(last_path))

        # EDIT 2026-07-27: best tracking + early stopping, independent of the
        # reporter. `row` is built either way; the table/CSV are the optional
        # part. CHANGELOG.md #3, #4
        row = {
            'epoch': f'{epoch}/{epochs}',
            'time': human_time(time.time() - run_start),
            'train/loss': epoch_loss / max(n_batches, 1),
            # EDIT 2026-07-27: val/loss, val/IoU, val/F1. CHANGELOG.md #5
            'val/loss': round(val_metrics['loss'], 6) if val_metrics and 'loss' in val_metrics else float('nan'),
            'val/dice': float(val_score),
            'val/IoU': round(val_metrics['iou'], 4) if val_metrics else float('nan'),
            'val/F1': round(val_metrics['f1'], 4) if val_metrics else float('nan'),
            'val/P': round(val_metrics['precision'], 4) if val_metrics else float('nan'),
            'val/R': round(val_metrics['recall'], 4) if val_metrics else float('nan'),
            'lr': optimizer.param_groups[0]['lr'],
        }
        if rep is not None:
            table.row(row)
            results_csv.append(row)

        if tracker.update(epoch, row):
            if save_checkpoint:
                torch.save(state_dict, str(best_path))
                msg = (f'  new best val/dice={tracker.best:.4f} @ epoch {epoch}'
                       f' — saved {best_path.name}')
            else:
                msg = (f'  new best val/dice={tracker.best:.4f} @ epoch {epoch}'
                       f' (not saved: save_checkpoint=False)')
            rep.info(msg) if rep is not None else logging.info(msg)

        if tracker.should_stop:
            msg = (f'Early stopping: no improvement in {tracker.patience} '
                   f'epochs (best val/dice={tracker.best:.4f} '
                   f'@ epoch {tracker.best_epoch})')
            rep.warning(msg) if rep is not None else logging.warning(msg)
            break

    # EDIT 2026-07-27: Ctrl+C now saves what has been learned and prints the
    # summary instead of dumping a traceback. CHANGELOG.md #3
    except KeyboardInterrupt:
        interrupted = True
        if rep is not None:
            rep.warning(f'Interrupted at epoch {last_epoch} — saving current weights.')
        if save_checkpoint:
            sd = model.state_dict()
            try:
                sd['mask_values'] = (train_set.dataset.mask_values
                                     if args.partition_mode == 'random_by_patch'
                                     and not args.nonoverlap_tr_tst
                                     else train_set.mask_values)
            except AttributeError:
                sd['mask_values'] = [0, 1]
            torch.save(sd, str(Path(dir_checkpoint) / 'interrupted.pt'))
            if rep is not None:
                rep.info(f'  saved {Path(dir_checkpoint) / "interrupted.pt"}')

    # EDIT 2026-07-27: leave `model` holding the BEST weights, not the last
    # epoch's. Without this the in-memory model after train_model() is whatever
    # the final epoch produced, which on the 2026-07-27 20-epoch run was 0.021
    # Dice worse than epoch 17. `mask_values` is not a parameter, so it is
    # stripped before load_state_dict. CHANGELOG.md #4
    if save_checkpoint and best_path.exists() and tracker.best_epoch:
        _best = torch.load(str(best_path), map_location=device)
        _best.pop('mask_values', None)
        _best.pop(CONVLSTM_CONFIG_KEY, None)  # EDIT 2026-07-28, CHANGELOG.md #6
        model.load_state_dict(_best)
        _msg = (f'Restored best weights from epoch {tracker.best_epoch} '
                f'(val/dice={tracker.best:.4f}) into the model')
        rep.info(_msg) if rep is not None else logging.info(_msg)

    # EDIT 2026-07-27: closing summary — best metric/epoch, wall time, artifact
    # paths. Runs whether the loop finished, early-stopped, or was Ctrl+C'd.
    # CHANGELOG.md #3
    if rep is not None:
        elapsed = time.time() - run_start
        rep.info('=' * 72)
        status = 'interrupted' if interrupted else 'complete'
        rep.info(f'Training {status} — {last_epoch} epochs in {human_time(elapsed)}'
                 f' ({human_time(elapsed / max(last_epoch, 1))}/epoch)')
        if tracker is not None and tracker.best_epoch:
            rep.info(f'Best val/dice {tracker.best:.4f} @ epoch {tracker.best_epoch}')
        rep.info(f'Weights   {dir_checkpoint}')
        if best_path.exists():
            rep.info(f'Best      {best_path}  (epoch {tracker.best_epoch})')
        if last_path.exists():
            rep.info(f'Last      {last_path}  (epoch {last_epoch})')
        rep.info(f'Results   {outpath}results.csv')
        # EDIT 2026-07-27: learning-curve figure from the CSV just written. Runs
        # after an interrupt too, so a Ctrl+C'd run still gets its plot.
        # CHANGELOG.md #5
        try:
            curves = plot_curves(Path(outpath) / 'results.csv',
                                 Path(outpath) / 'curves.png',
                                 best_epoch=tracker.best_epoch or None,
                                 title=args.job_name)
            if curves is not None:
                rep.info(f'Curves    {curves}')
        except Exception as exc:  # noqa: BLE001 - a plot must not fail a run
            rep.warning(f'Could not write curves.png: {exc}')
        _preds_dir = Path(dir_validation) / 'preds'
        if _preds_dir.exists():
            rep.info(f'Samples   {_preds_dir}  ({len(list(_preds_dir.glob("*.png")))} epochs)')
        rep.info(f'Test set  {outpath}test_dataset_{args.job_name}.pkl')
        rep.info('=' * 72)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--test', dest='test', type=float, default=10.0,
                        help='Percent of the data that is used as test (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--patch_size',  nargs = '+', type = int, default=[200,100], help='patch H, patch W')
    parser.add_argument('--stride',  type = int, default=2, help='train on patchs with given strides per window')

    parser.add_argument('--nonz_only', type = str, default='True', help='train only on non zero mask patches')

    parser.add_argument('--add_nulls_to_train', type = str, default='False', help='add patches with 0 sharp values')

    parser.add_argument('--patches_dir', type=str, default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/', help='path to patches')
    parser.add_argument('--partition_mode', type=str, default='random_by_patch', choices=['random_by_patch', 'random_by_intf','spatial','preset_by_intf'], help='partition mode')
    parser.add_argument('--partition_file', type=str, default=_bootstrap.asset('partition_20_05_13h45.json'), help=('preset partition file'))
    parser.add_argument('--preset_test_val_21',action='store_true', help='train on 19,20 test on 21')

    parser.add_argument('--train_on_11d_diff', type = str, default='True', help='')
    parser.add_argument('--job_name', type = str, default='', help='job name to add to output files')
    parser.add_argument('--intf_dict_path', type=str, default=_bootstrap.asset('intf_coord.json'), help='path to interferograms coord dict')
    parser.add_argument('--thresh_lat', type=float, default=31.4)
    parser.add_argument('--train_with_nonz_th', type = str, default='False', help='train only on non zero mask patches')
    parser.add_argument('--nonz_th',  nargs = '+', type = int, default=[350,150], help='north, south')
    parser.add_argument('--save_val', type = str, default='False', help='train only on non zero mask patches')
    parser.add_argument('--nonoverlap_tr_tst', type = str, default='False', help='')
    parser.add_argument('--retrain_with_fpz', type = str, default='False', help='')
    parser.add_argument('--test_data_to_exclude',type = str, default = None, help='path to previous run test dataset ')
    parser.add_argument('--use_cleaned_patches', action='store_true')
    parser.add_argument('--add_attn', action='store_true')
    parser.add_argument('--attn_unet', action='store_true')
    parser.add_argument('--augment', action='store_true',help='augment: only for spatial')
    parser.add_argument('--add_temporal', action='store_true')
    parser.add_argument('--k_prevs',type=int, default=2)
    parser.add_argument('--treat_nodata_regions', action='store_true')
    # EDIT 2026-07-28: ConvLSTM U-Net. CHANGELOG.md #6
    parser.add_argument('--convlstm_unet', action='store_true',
                        help='U-Net with a ConvLSTM over the bottleneck sequence. '
                             'Requires --add_temporal; mutually exclusive with '
                             '--attn_unet / --add_attn')
    parser.add_argument('--convlstm_hidden', type=int, default=0,
                        help='ConvLSTM hidden channels (0 = match the U-Net bottleneck, '
                             '1024 by default). Lower values cut parameters sharply')
    parser.add_argument('--convlstm_kernel', type=int, default=3,
                        help='ConvLSTM gate convolution kernel size (odd)')
    # EDIT 2026-07-28: legacy temporal target. CHANGELOG.md #7
    parser.add_argument('--union_temporal_mask', action='store_true',
                        help='Legacy: target = union of masks over the temporal stack. '
                             'Default is now the latest timestep only')

    parser.add_argument('--pos_w', type=float, default=1)
    parser.add_argument('--lidar_gate', action='store_true')
    parser.add_argument('--add_ring_negatives', action='store_true')
    parser.add_argument('--neg_ring_inner', type=int, default=1)
    parser.add_argument('--neg_ring_outer', type=int, default=3)
    parser.add_argument('--neg_per_pos', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=None)
    # EDIT 2026-07-27: reporter controls. --reporter False restores the previous
    # console output exactly. CHANGELOG.md #3
    parser.add_argument('--reporter', type=str, default='True',
                        help='Ultralytics-style CLI reporter (table + results.csv + best.pt)')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early-stop after N epochs without a val/dice improvement (0 = off)')
    # EDIT 2026-07-27: CHANGELOG.md #4
    parser.add_argument('--save_best_only', action='store_true',
                        help='Write only best.pt / last.pt, not one 124 MB checkpoint per epoch')
    # EDIT 2026-07-27: qualitative sample grids. --sample_every 0 turns them off.
    # CHANGELOG.md #5
    parser.add_argument('--sample_every', type=int, default=1,
                        help='Save a validation input/GT/prediction grid every N epochs (0 = off)')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of validation patches in each sample grid')






    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # Configure the logging system
    logging.basicConfig(level=logging.INFO)  # Set the logging level (e.g., INFO)
    now = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    args.job_name = args.job_name + '_' + now
    outpath = './outputs/' + args.job_name + '/'
    os.makedirs(outpath, exist_ok=True)
    # Create a FileHandler and specify the log file name

    log_file = outpath + args.job_name + '_' + now + '.log'
    file_handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(file_handler)
    logging.info("train job started")

    # EDIT 2026-07-27: hand the console to the reporter. The root logger's
    # FileHandler stays at INFO, so outputs/<job>/<job>_<ts>.log keeps exactly the
    # content it had before (including the per-step loss lines); only the root
    # *console* handler is raised to WARNING. --reporter False skips this
    # entirely and the previous console output returns. CHANGELOG.md #3
    if str2bool(args.reporter):
        quiet_root_console(logging.WARNING)
        setup_logger(REPORTER_NAME, log_file=Path(outpath) / 'logs' / 'reporter.log')
    is_running_locally = os.environ.get('LOCAL_ENVIRONMENT', False)

    if is_running_locally:
        logging.info("Running locally")
    else:
        logging.info("Running on a remote server")


    args.train_on_11d_diff = str2bool(args.train_on_11d_diff)
    args.nonz_only = str2bool(args.nonz_only)
    args.train_with_nonz_th = str2bool(args.train_with_nonz_th)
    args.add_nulls_to_train = str2bool(args.add_nulls_to_train)
    args.save_val = str2bool(args.save_val)
    args.nonoverlap_tr_tst = str2bool(args.nonoverlap_tr_tst)
    args.retrain_with_fpz = str2bool(args.retrain_with_fpz)
    args.reporter = str2bool(args.reporter)  # EDIT 2026-07-27, CHANGELOG.md #3

    dir_checkpoint = Path(outpath + 'checkpoints/')
    dir_validation = Path(outpath + 'validation/')
    dir_test = Path(outpath)

    # EDIT 2026-07-27: was `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`,
    # which never selected Apple Silicon. get_device() adds MPS between CUDA and CPU;
    # CUDA and CPU behaviour is unchanged. See CHANGELOG.md #1.
    device = get_device()
    #device = get_default_device()
    logging.info(f'Using device {device}')
    if args.add_temporal:
        num_c = args.k_prevs+1
    else:
        num_c=1
    if args.treat_nodata_regions:
        num_c = num_c*2


    # EDIT 2026-07-28: explicit three-way architecture selection. Conflicting flags are a
    # hard error rather than a silent precedence rule. CHANGELOG.md #6
    if args.convlstm_unet:
        _conflicts = [f for f, on in (('--attn_unet', args.attn_unet),
                                      ('--add_attn', args.add_attn)) if on]
        if _conflicts:
            logging.error(f'--convlstm_unet cannot be combined with {", ".join(_conflicts)}: '
                          f'they select different architectures. Pass exactly one.')
            sys.exit(2)
        if not args.add_temporal:
            logging.error('--convlstm_unet needs a temporal sequence but --add_temporal is '
                          'not set, so the loader would hand it a single interferogram. '
                          'Pass --add_temporal --k_prevs N (e.g. --k_prevs 2).')
            sys.exit(2)
        # num_c above is the FLAT channel count (T, or 2T with --treat_nodata_regions).
        # ConvLSTMUNet is built from channels PER TIMESTEP instead; T stays dynamic.
        n_ch_per_t = 2 if args.treat_nodata_regions else 1
        model = ConvLSTMUNet(n_channels_per_timestep=n_ch_per_t,
                             n_classes=args.classes,
                             bilinear=args.bilinear,
                             convlstm_hidden_channels=args.convlstm_hidden,
                             convlstm_kernel_size=args.convlstm_kernel)
    elif args.attn_unet:
        model = AttentionUNet(n_channels=num_c, n_classes=args.classes,bilinear=args.bilinear)
    else:
        model = UNet(n_channels=num_c, n_classes=args.classes, bilinear=args.bilinear,add_attn=args.add_attn)
    # EDIT 2026-07-27: was an unconditional `.to(memory_format=torch.channels_last)`.
    # MPS cannot autograd through channels_last (RuntimeError in backward), so the
    # layout is now chosen per device. Unchanged on CUDA and CPU. See CHANGELOG.md #1.
    model = model.to(memory_format=memory_format_for(device))

    # EDIT 2026-07-28: say "per timestep" for ConvLSTMUNet — its n_channels is the channel
    # count of ONE interferogram, not of the batch. CHANGELOG.md #6
    _ch_desc = (f'{model.n_channels} input channels per timestep '
                f'(sequence length T={args.k_prevs + 1}, not fixed by the architecture)'
                if args.convlstm_unet else f'{model.n_channels} input channels')
    logging.info(f'Network:\n'
                 f'\t{_ch_desc}\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    model.to(device=device)
    try:
        train_model(
            args,
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            test_percent = args.test / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
