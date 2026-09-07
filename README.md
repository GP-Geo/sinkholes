# Dead Sea Sinkhole Detection

Semantic segmentation of subsidence (sinkhole-precursor) areas in Sentinel-1 InSAR
interferograms of the Dead Sea coast. Wrapped-phase scenes are cut into aligned patch
grids, a U-Net-family model is trained against manually mapped subsidence polygons, and
full scenes are reconstructed into LiDAR-gated predictions exported as georeferenced
shapefiles for GIS use.

Everything runs through one CLI:

```bash
python -m sinkholes <command>        # from the repo root, or
pip install -e . && sinkholes <command>
```

| command | what it does |
|---|---|
| `prepare-metadata` | parse `.ers` headers into the interferogram coordinate dictionary |
| `prepare-patches` | cut `.unw` scenes + GT polygons into aligned patch grids |
| `count-positives` | fill per-interferogram positive-patch counts into the dictionary |
| `clean-patches` | optional QC: drop no-data / edge-sliver mask polygons |
| `make-partition` | write a reproducible train/val partition JSON |
| `prepare-local-subset` | make a partial patch download training-ready |
| `train` | train UNet / UNet+attention / AttentionUNet / ConvLSTM U-Net |
| `test-patches` | patch-level metrics on a run's pickled test split |
| `eval-scenes` | full-scene reconstruction → confidence maps, polygons, arrays |
| `eval-outputs` | object-level metrics + figures over saved eval-scenes outputs |
| `predict` | predict polygons on new raw `.unw` scenes (no ground truth) |
| `inspect-run` | learning curve + decision-threshold sweep for a finished run |
| `curves` | regenerate `curves.png` for a finished run |
| `architectures` | list the registered model architectures |

`sinkholes <command> --help` lists every flag. The full stage-by-stage reference,
including the conventions the science depends on, is
[`docs/PIPELINE.md`](docs/PIPELINE.md).

Where to look for what:

| Question | File |
|---|---|
| What do we know? What should I not repeat? | [`docs/RESULTS.md`](docs/RESULTS.md) |
| What did each batch settle? | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| The object-level numbers | [`docs/PREDICTIONS.md`](docs/PREDICTIONS.md) |
| Per-run registry, and attention selectivity | [`docs/MODEL_RUNS.md`](docs/MODEL_RUNS.md) |
| What is on disk, what is safe to delete | [`docs/OUTPUTS.md`](docs/OUTPUTS.md) |
| How to run a stage | [`docs/PIPELINE.md`](docs/PIPELINE.md) |

**Object-level F1 decides everything.** Patch `val/dice` ranks these models
*backwards* and is a training-health signal only — see `RESULTS.md` finding ②.

## Layout

```
sinkholes/            the package
├── cli.py            command dispatch (one entry surface)
├── geo.py            frame origins, grid cropping — the georeferencing constants
├── meta.py           interferogram metadata + 11-day temporal chains
├── normalise.py      phase normalisation and the 0.5 no-data convention
├── polygons.py       masks ↔ polygons, pixel → lon/lat
├── models/           UNet, AttentionUNet, ConvLSTM U-Net + checkpoint factory
├── dataprep/         patchify, the dataset, partitioning, prep commands
├── training/         losses, the training loop, evaluation, run reporter
└── inference/        shared scene reconstruction + the evaluation/predict commands
scripts/              cluster-side wrappers — the package does the work
├── submit_all.sh     the shield in front of bsub; only jobs that never ran
├── tidy_outputs.sh   file a finished run under outputs/<date>/ with its proper name
├── train/            LSF templates, one per architecture
├── eval/             run_eval.sh (2-stage scoring), rescore.sh, run_probe.sh
├── data/             patch generation and dataset verification
└── viewer/           interferogram viewer
assets/               committed data assets (intf_coord.json, partitions, LiDAR coverage)
docs/                 see the table above; reference/ holds the paper and status deck
tests/                pytest suite (pins the behavioural invariants)
data/, outputs/, models/, test_data/   local data and run artifacts (gitignored)
```

## Requirements

Python ≥ 3.10 with torch, numpy, scipy, pandas, matplotlib, tqdm, geopandas, pyogrio,
rasterio, shapely, affine, scikit-image — `pip install -e .` installs them. CUDA and
Apple Silicon (MPS) are picked up automatically; CPU works but is ~10x slower.

## Quickstart on a local subset

With a partial download of patch grids (full `(ny, nx, H, W)` `.npy` grids under
`data/patches/{images,masks}` plus `data/metadata/nonz_indices.json`):

```bash
# one-time: derive the positive-only patch files + a corrected metadata dictionary
python -m sinkholes prepare-local-subset

# a ~30-second training smoke test (3 interferograms)
python -m sinkholes train --epochs 1 --batch_size 8 \
  --patches_dir data/patches/train_ready/ \
  --intf_dict_path data/metadata/intf_coord_local.json \
  --partition_mode random_by_intf --k_prevs 0 \
  --train_intfs 20190204_20190215 --val_intfs 20190216_20190227 \
  --test_intfs 20190205_20190216 --job_name smoke

# patch-level metrics on the run's held-out test split
python -m sinkholes test-patches \
  --test_data_path outputs/smoke_<ts>/test_dataset_smoke_<ts>.pkl \
  --model outputs/smoke_<ts>/checkpoints/best.pt --k_prevs 0

# full-scene reconstruction → polygons (~3 min on MPS, ~1.3 GB of arrays)
python -m sinkholes eval-scenes --model outputs/smoke_<ts>/checkpoints/best.pt \
  --input_patch_dir data/patches/train_ready/ \
  --intf_source intf_list --intf_list 20190205_20190216 --k_prevs 0
```

Every evaluation command detects the architecture from the checkpoint's weights, so
UNet, AttentionUNet and ConvLSTM checkpoints all load with no extra flag.

## Tests

```bash
python -m pytest
```

Data-dependent tests skip themselves when the local subset is absent. The end-to-end
regression check is in `docs/PIPELINE.md` ("Regression oracle"): full-scene inference
with `models/run_v2_best.pt` on `20190205_20190216` must produce **464 polygons**.
