# Partition registry

`assets/` holds **three generations** of interferogram partition. They coexist on purpose:
the older two are what every published number was produced on, so deleting them would make
`docs/MODEL_RUNS.md`, `docs/RESULTS.md` and `docs/PREDICTIONS.md` unreproducible. Nothing
selects a generation for you — every entry point now demands an explicit path.

**New work uses generation 3.** Generations 1 and 2 are frozen: read them, re-run them,
never extend them.

---

## Generation 3 — `*_clean.json` (current)

Built by `sinkholes make-benchmark-partitions` per `docs/PLAN_CLEAN_BENCHMARK.md`.
**All years 2019–2026**, 11-day, frame overlap removed by a latitude cut at **31.4°**, every
split restricted to the shoreline **AOI lat 31.25–31.75 / lon 35.38–35.46**.

The 2019–2022 year restriction that this family was first designed around was **overturned**
on 2026-08-18 (plan §0a): quality is enforced spatially by the AOI, not by dropping years.

| File | train / val / test | crossview | AOI positives (train/val/test) |
|---|---|---:|---|
| `partition_geo_k5_clean.json` | 127 / 34 / 51 | 117 | 32,515 / 4,093 / 5,789 |
| `partition_geo_k10_clean.json` | 107 / 23 / 35 | 98 | 27,443 / 2,649 / 3,904 |
| `partition_temporal_k5_clean.json` | 163 / 21 / 27 | — | 42,985 / 5,917 / 9,152 |
| `partition_temporal_k10_clean.json` | 122 / 16 / 22 | — | 32,590 / 4,735 / 8,001 |
| `*_testeval_clean.json` (×4) | parent's test list under `"val"` | — | — |

Generated 2026-08-18 by `sinkholes make-benchmark-partitions`; counts are the real ones the
command printed, not projections. `crossview` is geo-only and **evaluation-only** — it shares
interferograms with `train` by design (plan §5b).

Two keys generation 3 has and the others do not:

- **`aoi_window`** — per split, the lat **and** lon box its patches are restricted to. Three
  consumers must agree on it (`dataprep/dataset.py`, `inference/scenes.py`,
  `inference/outputs.py`) or a model is scored on ground it trained on.
- **`provenance`** — `{years, k_prevs, cut_lat, aoi, seed, ...}`.

*Counts are from the plan's re-derivation (§5a); the generator prints the real table when it
runs. Trust the file's own `provenance`, not this row.*

## Generation 2 — `partition_{geo,temporal}_k{5,10}.json` — **frozen**

The 2019–**2026** benchmark. Every result currently in `MODEL_RUNS.md`, `RESULTS.md` and
`PREDICTIONS.md` is on these, and all associated outputs live under
`outputs/archive_2019_2026_noisy_data/`.

| File | train / val / test |
|---|---|
| `partition_geo_k5.json` | 109 / 25 / 26 |
| `partition_geo_k10.json` | 91 / 17 / 18 |
| `partition_temporal_k5.json` | 106 / 30 / 24 |
| `partition_temporal_k10.json` | 76 / 30 / 20 |
| `partition_geo_k5_testeval.json` | 134 / 26 (test-as-val) |
| `partition_geo_k10_testeval.json` | 108 / 18 |
| `partition_temporal_k5_testeval.json` | 136 / 24 |
| `partition_temporal_k10_testeval.json` | 106 / 20 |

Why superseded, in one line each:

- They carry 2023–2026 interferograms, where the best model's object-level precision falls
  to 0.44 while recall holds at 0.91 — the background of the newer scenes, not the objects.
- The geo split divides by **frame**, so the 31.25–31.44° band (~21 km) imaged by both
  frames sits in train *and* in the hold-out.

Each file now carries a `provenance` block recording exactly this. The interferogram lists
were not touched when it was added.

## Generation 1 — `partition_20_05_*.json` — **frozen**

The earliest family, 2019–2021, **train/val only — no test split**.

| File | train / val | Notes |
|---|---|---|
| `partition_20_05_13h45.json` | 144 / 16 | was the implicit default in `train.py` until 2026-08-18 |
| `partition_20_05_13h55.json` | 144 / 17 | |
| `partition_20_05_16h53.json` | 145 / 16 | parent of the two below |
| `partition_20_05_16h53_k5_compatible.json` | 89 / 25 | chain-valid to k=5; 12,396 / 58,947 val samples |
| `partition_20_05_16h53_k10_compatible.json` | 63 / 19 | chain-valid to k=10; 10,403 / 49,121 val samples |

The `_compatible` pair carries its own metadata keys (`derived_from`, `promotion_seed`,
`val_percent_actual`, …) and predates the `provenance` convention; they were left as-is.

---

## No implicit selection

As of 2026-08-18 nothing picks a partition for you:

- `sinkholes train --partition_mode preset_by_intf` **errors** without `--partition_file`.
  It used to fall back to `partition_20_05_13h45.json` — a generation-1 file — silently.
- `scripts/train/train_{convlstm,control,tattn}.sh` require `PARTITION` in the environment
  (`${PARTITION:?…}`) instead of defaulting to a generation-2 file.
- The `K_PREVS`-vs-partition guards in those scripts match `*_k5.json` **and** `*_k5_*.json`,
  so the generation-3 names are guarded too. The old glob would have skipped them silently.

## Conventions

- `_testeval` variants hold the parent's **test** list under the `"val"` key, so a training
  entry point that reads only train/val evaluates on test. `load_partition_split` refuses a
  missing key rather than returning an empty list, for this reason.
- `_k5` files list only interferograms with a valid 5-predecessor 11-day chain, `_k10` only
  those with 10. `K_PREVS` must match the filename or the run trains on a smaller set than
  intended.
- Keys other than `train` / `val` / `test` are metadata; `PARTITION_SPLITS` in
  `sinkholes/dataprep/partition.py` is the authority on which keys are lists.
