# Dead Sea Sinkhole Detection — Training & Testing Pipeline

Semantic segmentation of sinkhole related subsidence areas in Sentinel-X SAR interferograms of the Dead Sea coast, using a U-Net (optionally an Attention U-Net)
trained on interferogram patches with manually mapped subsidence polygons as ground truth.

## Repository Layout

```
├── src/
│   ├── models/       unet, unet_parts, attn_unet, convlstm_unet
│   ├── data_prep/    prepare_*, create_intf_partition, clean_patches,
│   │                 check_patches, untie_mask_polygs
│   ├── training/     train_sinkholes_unet, train_reporter, evaluate,
│   │                 dice_score, sinkholes_data_loading
│   ├── inference/    test, test_full_intf, predict_new_intf(_vA),
│   │                 evaluate_full_intf_output, inspect_run, remove_no_Data_predictions
│   ├── utils/        device_utils, get_intf_info, polygs, lidar
│   └── _bootstrap.py path shim — see below
├── assets/           intf_coord.json, lidar_mask_polygs.*, lidar_intf_mask.txt, partition_*.json
├── docs/             PIPELINE_MANUAL, LOCAL_WALKTHROUGH, PROJECT_SUMMARY, TRAINING_RUNS, CHANGELOG
├── tests/            pytest suite
├── data/             local patches (gitignored)
└── outputs/          run artifacts (gitignored)
```

Modules import each other by bare name (`from unet import UNet`) regardless of subfolder.
`src/_bootstrap.py` makes that work: entry-point scripts import it via a short header, and it
puts every `src/*` folder on `sys.path`. It also exposes `asset()`, which resolves the files in
`assets/` absolutely — so scripts run correctly from any working directory.

Run scripts by path, e.g. `python src/training/train_sinkholes_unet.py --help`.

## Pipeline Overview

```
interferograms + mapped subsidence polygons : (full interferograms: /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/
                                              data patches: /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/data_patches...
                                         GT mask patches: /home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/mask_patches...)
        │
        ▼
1. Patch preparation      prepare_intrfrgrm_pathches.py
        │
        ▼
2. Training               train_sinkholes_unet.py
        │
        ├──▶ 3a. Patch-level testing        test.py
        │
        ├──▶ 3b. Full-interferogram testing test_full_intf.py
        │
        └──▶ 4. Inference on new data       predict_new_intf.py
```

## 1. Data Preparation

- **`prepare_intf_coord_dict.py`** — builds `intf_coord.json`, a dictionary of geographic metadata (origin, pixel size, grid dimensions) per interferogram. Needed by all later stages.
- **`prepare_intrfrgrm_pathches.py`** — the main patch generator:
  1. Loads each interferogram and rasterizes the mapped subsidence polygons (.shp file) into a binary ground-truth mask.
  2. Cuts both into fixed-size patches (default 200×100) with a configurable stride (default 2)
  3. Saves all patches, plus a separate set of "non-zero" patches (patches whose mask contains at least one subsidence pixel) used for training.

## 2. Training

**`train_sinkholes_unet.py`** trains the segmentation network:


Main behavior:

- **Network** — standard U-Net (`unet.py`), U-Net with bottleneck attention (`--add_attn`),
  Attention U-Net (`--attn_unet`, `attn_unet.py`), or **ConvLSTM U-Net** (`--convlstm_unet`,
  `convlstm_unet.py`) which runs the encoder per timestep and a ConvLSTM over the bottleneck
  sequence instead of stacking frames as channels. The ConvLSTM is the best-scoring model to date
  — see [`docs/TRAINING_RUNS.md`](docs/TRAINING_RUNS.md).
- **Partition modes** — random by patch, random by interferogram, spatial split, or a preset partition file (for intf partition)
- **Temporal context** — optionally stacks the *k* previous interferograms of the same frame as extra input channels (`--add_temporal`), giving the network the deformation history of each pixel.
the GT in that case is the unified GT mask. (currently hard-coded should be made configurable)
- **Loss** — masked BCEwl + Dice loss. `--pos_w` gives higher weight to positive (subsidence) pixels
- **Outputs** — a checkpoint (`*checkpoint_epoch<N>.pth`) per epoch, a log file per job, and optionally a pickled test/validation dataset for later evaluation.

Run `python src/training/train_sinkholes_unet.py -h` for the full list of options.

## 3. Testing

> **Architecture is detected from the checkpoint.** Every script below inspects the saved weights
> and builds the matching network (`factory.py`), so U-Net, Attention U-Net and ConvLSTM
> checkpoints all work with no extra flag. `--attn_unet` / `--add_attn` / `--convlstm_unet` still
> work as an explicit override; one that contradicts the weights is reported as an error rather
> than failing with a key mismatch.

### Patch-level (`test.py`)

Loads a saved test dataset (pickle from training) and a checkpoint, and reports Dice score, pixel-level precision/recall, 
and object-level (OL) precision/recall over the test patches. For the OL metrics, predicted objects are matched to ground-truth objects using an overlap threshold (`--th`) and a buffer in pixels (`--b`):

python src/inference/test.py --test_data_path <test_set.pkl> --model <checkpoint.pth>


### Full-interferogram (`test_full_intf.py`)

The main evaluation path. relevant for by-intf parition. For each test interferogram it:

1. loads the interferogram patches and runs the model on all of them (only patches inside the LiDAR coverage region — the valid-region mask from `lidar_mask_polygs.shp` — are predicted).
2. Reconstructs the full-scene prediction map using a stride of `--data_stride` (overlapping patches can be blended with a Hann window). 
3. Thresholds the prediction (`recon_th`) and converts connected regions to polygons (`polygs.py`).
4. Outputs: reconstructed arrays (interferogram, prediction (confidence map), GT mask) and the predicted polygons per interferogram.
5. With `--merge_polygs`, all per-intf polygons are also merged into one combined shapefile. (useful only for overall runs)

`evaluate_full_intf_output.py` computes precision/recall-style statistics from the saved full-interferogram outputs, and `remove_no_Data_predictions.py` filters out predictions that fall in no-data regions.

## 4. Predicting on New Interferograms

**`predict_new_intf.py`** runs a trained model on interferograms *without* ground truth: it crops the scene to the model grid, reconstructs the full prediction, and exports the predicted sinkhole polygons as a shapefile for GIS use. `predict_new_intf_vA.py` is an alternative version of this script — **prefer vA**, it fixes a LiDAR-mask indexing bug in v1 (`docs/PROJECT_SUMMARY.md §6.5`).

Both accept any architecture, detected from the checkpoint as described above.

## Repository Layout

| File | Role |
|---|---|
| `unet.py`, `unet_parts.py`, `attn_unet.py`, `convlstm_unet.py` | Network architectures |
| `factory.py` | Checkpoint → model. Detects the architecture from the weights; the single place any inference script builds a network. Add a new architecture here. |
| `sinkholes_data_loading.py` | PyTorch dataset / data loading |
| `dice_score.py` | Dice metric and loss |
| `evaluate.py` | Validation-time evaluation utilities |
| `get_intf_info.py` | Interferogram metadata & 11-day sequence lookup |
| `polygs.py` | Mask ↔ polygon conversion, pixel ↔ lon/lat |
| `assets/intf_coord.json` | Per-interferogram coordinate dictionary |
| `assets/partition_*.json` | Saved train/val splits |
| `assets/lidar_mask_polygs.*` | LiDAR coverage polygons — valid-region mask for prediction (shapefile) |

## Requirements

Python 3 with: `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `rasterio`, `geopandas`, `shapely`, `scikit-image`, `tqdm`.
