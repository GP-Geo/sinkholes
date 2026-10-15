# Partition registry

`assets/` holds **four generations** of temporal partition, each a superset of the rules of
the one below it. Nothing selects a generation for you — every entry point demands an
explicit path, and a number is only comparable with numbers on the same generation and the
same scene list.

**New work uses generation 5**; the HPO study uses 5-t2026. Generation 4 stays because two
focus models trained on it and its k5 test list is the official relabelled test set;
generation 3 because the relabel workflow reads it.

| Generation | Files | What it is |
|---|---|---|
| **5** (current) | `partition_temporal_k{5,10}[_testeval]_clean_fill_th350x200.json` | generation 4 with chains that **reach past missing acquisitions** |
| 5-t2026 (the HPO study) | `partition_temporal_k{5,10}[_testeval]_test2026_fill_th350x200.json` | generation 5 re-split: **test 2026, val 2022–23, train the rest**, fully separated |
| 4 | `partition_temporal_k{5,10}[_testeval]_clean_th350x200.json` | generation 3 plus a **label-quality threshold on targets** |
| 3 | `partition_temporal_k{5,10}[_testeval]_clean.json` | the clean benchmark: AOI, all years, no threshold |

`scripts/eval/run_eval.sh` and `run_probe.sh` pick them with `GEN=fill|t2026|th350|3`.

---

## Generation 5 — `*_clean_fill_th350x200.json` (filled chains, 2026-10-06) — current

Generation 4 rebuilt after `find_11day_sequences` stopped demanding a gap-free chain. A
missing acquisition is now skipped and the walk continues to an older one, up to
`3k` slots back (`FILL_LOOKBACK_FACTOR`), and each chain records its real `offsets`, which
the temporal attention receives as frame ages. Same command, cut, AOI, bounds, seed and
threshold as generation 4; only the chain rule differs.

Where no slot is missing the filled chain is the gap-free one, so **every split of
generation 4 is a subset of the same split here** (checked at write time). A generation-5
run can therefore be scored on the generation-4 test list too: that is the like-for-like
number, and the full generation-5 test list is the coverage gain.

| File | train / val / test intfs | positives in window |
|---|---|---|
| `partition_temporal_k5_clean_fill_th350x200.json` | 130 / 12 / 24 | 44,114 / 3,379 / 8,893 |
| `partition_temporal_k10_clean_fill_th350x200.json` | 121 / 11 / 24 | 41,032 / 3,162 / 8,893 |
| `partition_temporal_k5_testeval_clean_fill_th350x200.json` | 142 / 24 (test-as-val) | 47,493 / 8,893 |
| `partition_temporal_k10_testeval_clean_fill_th350x200.json` | 132 / 24 (test-as-val) | 44,194 / 8,893 |

Only `--tattn_unet` reads the ages. A ConvLSTM trained here sees uneven chains as evenly
spaced, and `train` warns when it does.

```bash
sinkholes make-benchmark-partitions \
  --intf_dict assets/intf_coord.json \
  --patches_dir $D/patches --seed 0 \
  --aoi 31.25 31.75 35.38 35.46 --axis temporal \
  --temporal_bounds 20240101 20250101 \
  --nonz_th 350 200 --suffix clean_fill_th350x200 --out_dir assets/
```

---

## Generation 5-t2026 — `*_test2026_fill_th350x200.json` (block split, 2026-10-13)

Generation 5's scenes, chains and threshold, re-split by **date blocks** instead of two
boundaries: `--temporal_blocks val:20220101:20231231 test:20260101:20261231`. Everything
outside the blocks is train, so training reaches **2024 and Jan–Oct 2025**: its newest
target moves from 2023-12 to 2025-10, and the gap to the test scenes shrinks from over a
year to about four months.

**Full separation.** With train on both sides of a held-out block, chains cross set
boundaries in both directions, so the rule is stricter than generation 5's: no target takes
an input that is **another split's target**, in any direction. The held-out blocks are kept
whole and train pays — a train scene is dropped when it feeds a held-out block (the lead-in)
or reads from one (the lead-out). `provenance.separation` reads `full` here and
`held_out_chains` for a `--temporal_bounds` run. Generation 5 does not meet the full rule:
3 of its test scenes (`20250122`, `20250223`, `20250328`) take val scenes as inputs, which
its design allows.

| File | train / val / test intfs | positives in window |
|---|---|---|
| `partition_temporal_k5_test2026_fill_th350x200.json` | 144 / 11 / 10 | 48,511 / 3,731 / 3,958 |
| `partition_temporal_k10_test2026_fill_th350x200.json` | 131 / 11 / 10 | 43,944 / 3,731 / 3,958 |
| `partition_temporal_k5_testeval_test2026_fill_th350x200.json` | 155 / 10 (test-as-val) | 52,242 / 3,958 |
| `partition_temporal_k10_testeval_test2026_fill_th350x200.json` | 142 / 10 (test-as-val) | 47,675 / 3,958 |

Unused at k10: `20240102`, `20240103`, `20240216`, `20240319` (their chains hold val
scenes) and `20250818`, `20251024`, `20251103`, `20251126` (they feed the 2026 test scenes).

Three things to know before reading a number off it:

- **The test list is 10 scenes of one season** (Feb–Apr 2026), 8 of them relabelled. It is
  a subset of generation 5's 24, with the same chains and window, so a generation-5 model is
  compared by re-averaging its per-scene results over these 10: the fill k10 leader scores
  **0.8660** there (original labels, RTh 0.125), its no-fill twin 0.8499. Per-scene sd is
  0.07, so a gap under ~0.02–0.03 is noise.
- **Feb–Apr is the season the models already do best on** (0.862 mean per-scene F1 against
  0.726 for the May–Dec test scenes of generation 5). The weak summer scenes are not tested.
- **Nine relabelled 2025 scenes become training targets.** Training reads the original
  labels from the patch trees; the corrections help only if those scenes' mask patches are
  rebuilt from the corrected export.

```bash
sinkholes make-benchmark-partitions \
  --intf_dict assets/intf_coord.json \
  --patches_dir $D/patches --seed 0 \
  --aoi 31.25 31.75 35.38 35.46 --axis temporal \
  --temporal_blocks val:20220101:20231231 test:20260101:20261231 \
  --nonz_th 350 200 --suffix test2026_fill_th350x200 --out_dir assets/
```

---

## Generation 4 — `*_clean_th350x200.json`

Generation 3's temporal family with one thing added: `--nonz_th 350 200`, a **per-region
whole-scene positive-patch threshold on targets**. A scene enters train/val/test only when
`nonz_num` exceeds the threshold for its region (north when the frame origin is above lat
31.5, else south). Generated 2026-09-10 by the same command, same AOI, same seed, same
`20240101 / 20250101` temporal bounds.

| File | train / val / test | AOI positives (train/val/test) |
|---|---|---|
| `partition_temporal_k5_clean_th350x200.json` | 100 / 11 / 20 | 35,558 / 3,184 / 7,675 |
| `partition_temporal_k10_clean_th350x200.json` | 74 / 10 / 17 | 27,191 / 2,967 / 6,918 |
| `partition_temporal_k5_testeval_clean_th350x200.json` | 111 / 20 (test-as-val) | 38,742 / 7,675 |
| `partition_temporal_k10_testeval_clean_th350x200.json` | 84 / 17 (test-as-val) | 30,158 / 6,918 |

Four properties of the threshold, each easy to get wrong:

- **Targets only; predecessors are exempt.** The 11-day chains are built over the whole
  dictionary *before* the filter runs, so a thinly labelled scene still supplies an input
  frame to a kept scene. Filtering the chains too would silently shorten every stack that
  reaches back across such a date.
- **Whole-scene, not in-window.** Quality is judged on `nonz_num` over the entire scene, on
  purpose: an in-window count would confuse "badly digitised" with "few sinkholes in this
  band". Since an in-window count can never exceed the whole-scene one, the filter cannot
  drop a scene that has more than the threshold's worth of positives inside the window.
- **It is a different generation, and stamped as one.** `provenance.generation` reads
  `all_years_clean_nonz350-200`, not `all_years_clean` — a threshold-restricted file holds
  fewer scenes than the bare name promises, so it must not be indistinguishable from the
  full family.
- **The equivalent train-time flag does nothing here.** `train --train_with_nonz_th` filters
  the *discovered* interferogram list, which `preset_by_intf` never reads. On a preset
  partition, label quality is a property of the file.

**What the threshold is worth, measured.** Re-averaging three finished generation-3 ctx50
runs over only the above-threshold scenes moves object F1 by **+0.013 to +0.024** at every
RTh, for all three models — three to four times the ±0.006 noise floor. That is a level
shift from an easier scene list, **not** model quality, and it is the amount by which a
generation-4 number is inflated against a generation-3 one before any model difference is
counted.

**350 is not a clean separation, and it was known not to be.** Against the k5 list: 17 of
20 scored scenes are common with generation 3, 3 are added (no 10-previous chain, so k10
cannot hold them) and 3 dropped at 346 / 343 / 339 positives. The three dropped are the
low-precision cluster (0.61–0.63 against 0.82–0.92 for the retained northern scenes) — but
two more of that cluster survive at 356 and 359, including the worst scene in the set at
0.530 precision. The real gap in the data sits between **359 and 496**. A future threshold
has somewhere better to go.

The `--axis` flag writes one family or both; generation 4 is temporal-only because that is
the axis the September batches ran on.

```bash
sinkholes make-benchmark-partitions \
  --intf_dict assets/intf_coord.json \
  --patches_dir $D/patches --seed 0 \
  --aoi 31.25 31.75 35.38 35.46 --axis temporal \
  --temporal_bounds 20240101 20250101 \
  --nonz_th 350 200 --suffix clean_th350x200 --out_dir assets/
```

## Generation 3 — `*_clean.json`

Built by `sinkholes make-benchmark-partitions` on 2026-08-18. **All years 2019–2026**,
11-day, every split restricted to the shoreline **AOI lat 31.25–31.75 / lon 35.38–35.46**
(the geo family of this generation also cut the frame overlap at 31.4°; it is retired,
below). Quality is enforced spatially by the AOI rather than by dropping years.

| File | train / val / test | AOI positives (train/val/test) |
|---|---|---|
| `partition_temporal_k5_clean.json` | 163 / 21 / 27 | 42,985 / 5,917 / 9,152 |
| `partition_temporal_k10_clean.json` | 122 / 16 / 22 | 32,590 / 4,735 / 8,001 |
| `*_testeval_clean.json` (×2) | parent's test list under `"val"` | — |

`sinkholes relabel` reads these threshold-free files for *candidate* scenes — test scenes
that relabelling might lift over generation 4's threshold. The official relabelled set is
generation 4's k5 test list (20 scenes, `docs/RELABELING.md`).

Two keys every generation from 3 on carries:

- **`aoi_window`** — per split, the lat **and** lon box its patches are restricted to. Three
  consumers must agree on it (`dataprep/dataset.py`, `inference/scenes.py`,
  `inference/outputs.py`) or a model is scored on ground it trained on.
- **`provenance`** — `{generation, years, k_prevs, aoi, seed, ...}`. Trust the file's own
  `provenance` over any table here.

## Retired 2026-10-15

Removed from `assets/` with every model trained on them archived. Recover any file with
`git show notebook-2026-10-15:assets/<file>`; the full write-up of each is in that tag's
`assets/PARTITIONS.md`.

| Generation | Files | Why it went |
|---|---|---|
| 3, geo | `partition_geo_k{5,10}[_testeval]_clean.json` | the geo axis was dropped for temporal on 2026-09-03 |
| 3b | `partition_{geo,temporal}_k{5,10}[_testeval]_pre2023.json` | the 2019–2022 archive; it answered its question (the 2023–2026 training years buy nothing measurable, `docs/RESULTS.md` ⑪) |
| 2 | `partition_{geo,temporal}_k{5,10}[_testeval].json` | no AOI, and the geo split put the frame-overlap band in train *and* hold-out |
| 1 | `partition_20_05_*.json` | 2019–2021, train/val only |

## No implicit selection

- `sinkholes train --partition_mode preset_by_intf` **errors** without `--partition_file`.
- `scripts/train/train_{convlstm,control,tattn}.sh` require `PARTITION` in the environment
  (`${PARTITION:?…}`).
- The `K_PREVS`-vs-partition guards in those scripts match `*_k5.json` **and** `*_k5_*.json`.

## Conventions

- `_testeval` variants hold the parent's **test** list under the `"val"` key, so a training
  entry point that reads only train/val evaluates on test. `load_partition_split` refuses a
  missing key rather than returning an empty list, for this reason.
- `_k5` files list only interferograms with a valid 5-predecessor 11-day chain, `_k10` only
  those with 10. `K_PREVS` must match the filename or the run trains on a smaller set than
  intended.
- Keys other than `train` / `val` / `test` are metadata; `PARTITION_SPLITS` in
  `sinkholes/dataprep/partition.py` is the authority on which keys are lists.
