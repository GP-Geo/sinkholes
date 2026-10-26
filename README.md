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
| `merge-lidar` | merge the per-year LiDAR coverage shapefiles into one |
| `make-benchmark-partitions` | write the temporal benchmark partitions (AOI window, chain separation) |
| `prepare-local-subset` | make a partial patch download training-ready |
| `train` | train UNet / UNet+attention / AttentionUNet / ConvLSTM / temporal-attention U-Net |
| `test-patches` | patch-level metrics on a pickled split **or** a partition JSON |
| `eval-scenes` | full-scene reconstruction → confidence maps, polygons, arrays |
| `eval-outputs` | object-level metrics + figures over saved eval-scenes outputs |
| `eval-snapshots` | the same RTh protocol for many checkpoints from one data pass (model selection) |
| `hpo` | the plain-vs-ctx50 Optuna study: `init`, `adopt`, `approve`, `step` (dry run / `--submit`), `status`, `coverage`, `report`, `panel` -- see [`docs/HPO.md`](docs/HPO.md) |
| `relabel` | test-set relabelling: QGIS workspace, validation, change review, re-scoring -- see [`docs/RELABELING.md`](docs/RELABELING.md) |
| `predict` | predict polygons on new raw `.unw` scenes (no ground truth) |
| `attention-probe` | where a `tattn_unet`'s attention lands over a long, gappy history |
| `curves` | regenerate `curves.png` for a finished run |
| `architectures` | list the registered model architectures |

`sinkholes <command> --help` lists every flag. The full stage-by-stage reference,
including the conventions the science depends on, is
[`docs/PIPELINE.md`](docs/PIPELINE.md).

Where to look for what:

| Question | File |
|---|---|
| The project in brief, what we know, which model is best, what not to repeat | [`docs/RESULTS.md`](docs/RESULTS.md) |
| How to run a stage, and the current training recipe | [`docs/PIPELINE.md`](docs/PIPELINE.md) |
| Which partition file, and why | [`assets/PARTITIONS.md`](assets/PARTITIONS.md) |
| The HPO / long-training study | [`docs/HPO.md`](docs/HPO.md) |
| The relabelled test set | [`docs/RELABELING.md`](docs/RELABELING.md) |
| What the 11-day dataset holds, and how it is rebuilt | [`docs/DATASET_11DAY.md`](docs/DATASET_11DAY.md) |
| What is on disk, what is safe to delete | [`docs/OUTPUTS.md`](docs/OUTPUTS.md) |
| The full lab notebooks (every batch, run and evaluation to 2026-10-15) | tag `notebook-2026-10-15` |

**Object-level F1 decides everything.** Patch `val/dice` ranks these models
*backwards* and is a training-health signal only — see `RESULTS.md` finding ②.

**Where things stand (2026-10-25).** The story of the project (the question, the starting
point in the published benchmark, what was done, where it stands and what is next) is the
first section of [`docs/RESULTS.md`](docs/RESULTS.md). In one line: plain temporal attention
over the current interferogram and the ten before it, on filled history chains, is the best
model (`outputs/2026-10-06/temporal_k10_th350_fill_tattn_posw8_30e`: object F1 0.8114 on the
24 generation-5 test scenes, precision 0.834, recall 0.790). The plain-vs-ctx50 HPO study
(`studies/k10t2026_hpo_v2`, `docs/HPO.md`) is in its first phase: plain leads (0.770 vs 0.757)
and, at equal precision, also recalls more (`RESULTS.md` ⑫).

## Layout

```
sinkholes/            the package
├── cli.py            command dispatch (one entry surface)
├── geo.py            frame origins, grid cropping — the georeferencing constants
├── meta.py           interferogram metadata + 11-day temporal chains
├── normalise.py      phase normalisation and the 0.5 no-data convention
├── polygons.py       masks ↔ polygons, pixel → lon/lat
├── models/           UNet, AttentionUNet, ConvLSTM + temporal-attention U-Net,
│                    and the checkpoint factory that detects which is which
├── dataprep/         patchify, the dataset, partitioning, prep commands
├── training/         losses, the training loop, evaluation, run reporter
├── inference/        shared scene reconstruction + the evaluation/predict commands
├── relabel/          test-set relabelling and re-scoring (docs/RELABELING.md)
└── hpo/              the Optuna study controller (docs/HPO.md)
scripts/              cluster-side wrappers — the package does the work
├── submit_all.sh     the shield in front of bsub; only jobs that never ran
├── tidy_outputs.sh   file a finished run under outputs/<date>/ with its proper name
├── train/            LSF templates, one per architecture
├── eval/             run_eval.sh (2-stage scoring), rescore.sh, run_probe.sh
├── data/             patch generation and dataset verification
├── relabel/          QGIS workspace helpers, eval_v2.sh
├── hpo/              the engine check
└── viewer/           interferogram viewer
assets/               committed data assets (intf_coord.json, partitions, LiDAR coverage)
docs/                 see the table above; reference/ holds the benchmark paper
tests/                pytest suite (pins the behavioural invariants)
data/, outputs/, models/, test_data/   local data and run artifacts (gitignored);
                     data/scenes_11day is the scene symlink farm scripts/data/*.sh read
studies/              HPO study roots -- ledger, runs, evaluations (gitignored, protected)
logs/                 LSF stdout/stderr of the scripts/ jobs (gitignored)
```

## Requirements

Python ≥ 3.10 with torch, numpy, scipy, pandas, matplotlib, tqdm, geopandas, pyogrio,
rasterio, shapely, affine, scikit-image — `pip install -e .` installs them;
`pip install -e '.[hpo]'` adds Optuna for the `hpo` controller (the GPU jobs never import
it). CUDA and Apple Silicon (MPS) are picked up automatically; CPU works but is ~10x
slower. On WEXAC the environment is `/home/labs/rudich/pinkas/.conda/envs/sinkholes`, with
an editable install of this checkout.

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
UNet, AttentionUNet, ConvLSTM and temporal-attention checkpoints all load with no extra
flag — and so does the patch geometry, including the large-context runs' `300×200` input
with its `200×100` target (`io_geometry`, see `docs/PIPELINE.md`).

## Tests

```bash
python -m pytest
```

Data-dependent tests skip themselves when the local subset is absent; the `hpo` tests
need Optuna. On macOS, if DataLoader workers abort with `OMP: Error #15`, run with
`KMP_DUPLICATE_LIB_OK=TRUE`. The end-to-end
regression check is in `docs/PIPELINE.md` ("Regression oracle"): full-scene inference
with `models/run_v2_best.pt` on `20190205_20190216` must produce **464 polygons**.
