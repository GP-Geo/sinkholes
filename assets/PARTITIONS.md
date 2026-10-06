# Partition registry

`assets/` holds **four generations** of interferogram partition. They coexist on purpose:
the older ones are what every published number was produced on, so deleting them would
make `docs/MODEL_RUNS.md`, `docs/RESULTS.md` and `docs/PREDICTIONS.md` unreproducible.
Nothing selects a generation for you — every entry point demands an explicit path.

**New work uses generation 4.** Generations 1–3 are frozen: read them, re-run them, never
extend them.

| Generation | Files | What it is |
|---|---|---|
| **4** (current) | `partition_temporal_k{5,10}[_testeval]_clean_th350x200.json` | generation 3 plus a **label-quality threshold on targets** |
| 3 | `partition_{geo,temporal}_k{5,10}[_testeval]_clean.json` | the clean benchmark: AOI + latitude cut, all years |
| 3b | `partition_{geo,temporal}_k{5,10}[_testeval]_pre2023.json` | the same, archive stopped at 2022-12-31 |
| 2 | `partition_{geo,temporal}_k{5,10}[_testeval].json` | frame-split geo, no AOI |
| 1 | `partition_20_05_*.json` | 2019–2021, train/val only |

Generations 3 and 3b differ **only** in which years the archive contains — same generator,
cut, AOI, seed and val share — so a run on one is readable against its twin on the other
and the difference is the archive. **Do not mix them inside one comparison.**

---

## Generation 4 — `*_clean_th350x200.json` (current)

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
the axis the September batches run on. `scripts/eval/run_eval.sh`'s `GEN=th350` comment
still says *"ONLY temporal_k5 exists in this family"* — that was true when it was written
and is not now; the k10 pair was generated afterwards and `evalk10` scores against it.

```bash
sinkholes make-benchmark-partitions \
  --intf_dict assets/intf_coord.json \
  --patches_dir $D/patches --seed 0 \
  --aoi 31.25 31.75 35.38 35.46 --axis temporal \
  --temporal_bounds 20240101 20250101 \
  --nonz_th 350 200 --suffix clean_th350x200 --out_dir assets/
```

## Generation 3 — `*_clean.json`

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

## Generation 3b — `*_pre2023.json` (the 2019–2022 archive)

Same generator, same **cut at 31.4°**, same **AOI lat 31.25–31.75 / lon 35.38–35.46**, same
seed and same 40:60 geo val:test share as the `*_clean.json` family. **One thing differs:**
`--years 2019 2022`, so the archive stops at 2022-12-31. Generated 2026-08-20.

Why it exists: `docs/PLAN_CLEAN_BENCHMARK.md` §1 measured object-level precision at 0.91 on
2021 scenes and 0.44 on 2024–2025 ones while recall held at 0.91–0.93 — the newer
*background*, not the objects. §0a kept those years anyway and recorded the mechanism as
**unexplained and the risk as knowingly accepted**. This family is the other arm of that
decision: it makes "how much of the false-positive rate is the newer archive" a measurement.

| File | train / val / test | crossview | AOI positives (train/val/test) |
|---|---|---:|---|
| `partition_geo_k5_pre2023.json` | 90 / 24 / 34 | 86 | 21,432 / 2,576 / 3,686 |
| `partition_geo_k10_pre2023.json` | 71 / 15 / 22 | 67 | 16,690 / 1,462 / 2,377 |
| `partition_temporal_k5_pre2023.json` | 77 / 15 / 48 | — | 24,817 / 6,078 / 5,652 |
| `partition_temporal_k10_pre2023.json` | 50 / 6 / 35 | — | 16,409 / 2,632 / 3,890 |
| `*_testeval_pre2023.json` (×4) | parent's test list under `"val"` | — | — |

Counts are the ones the generator printed, not projections. 150 interferograms are
chain-valid at k=5 in this archive and 109 at k=10, against 214 / 166 for all years.

Three things to know before using them:

- **The temporal boundaries are `20210101 / 20210701`, not clean's `20240101 / 20250101`** —
  train 2019–2020, val 2021H1, test 2021H2 onward. Clean's bounds put every held-out scene
  in years this archive does not contain. This layout came from a sweep of all 320 viable
  monthly boundary pairs (plan §5); §5a explains why it is the wrong layout for the full
  archive. **The two are not interchangeable.**
- **`temporal_k10` val is 6 interferograms / 2,632 positives.** That is the risk plan §11
  flags as the plan's biggest, and `--add_val_negatives` — the padding it named as the
  mitigation — was removed on 2026-08-20. Six scenes drive the plateau schedule, early
  stopping and `best.pt` for that arm: treat its `val/dice` as noise and judge it at object
  level. The documented fallback is to let **val-only** chains reach back into train (§5),
  which the generator does not do today.
- **`provenance.generation` is `2019_2022_clean`, not `all_years_clean`**, and
  `provenance.years_filter` is `[2019, 2022]`. A year-restricted file is not stamped as the
  all-years generation — `benchmark_partitions.generation_label` enforces that.

Reproduce with:

```bash
sinkholes make-benchmark-partitions \
  --intf_dict assets/intf_coord.json \
  --patches_dir $D/patches --cut_lat 31.4 --seed 0 \
  --aoi 31.25 31.75 35.38 35.46 --geo_val_share 0.4 \
  --years 2019 2022 --temporal_bounds 20210101 20210701 \
  --suffix pre2023 --out_dir assets/
```

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
