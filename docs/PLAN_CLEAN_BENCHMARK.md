# Plan — rebuild the benchmark

*Written 2026-08-17 as "clean 2019–2022". **Premise revised 2026-08-18: the benchmark now covers
all 2019–2026 interferograms**, with quality enforced by a spatial AOI rather than by a year
filter. Renamed from `PLAN_CLEAN_2019_2022.md`. Supersedes the partition family in
`assets/partition_{geo,temporal}_k{5,10}.json` for every new experiment. Nothing here changes a
finished run or a published number.*

The four current partitions put 2023–2026 interferograms into val/test, and §1 records the
precision collapse that motivated restricting the years. That restriction has been **overturned
by decision** (§0a): the AOI transfers cleanly to every year, the newer years are no less dense
per labelled scene, and both bookkeeping explanations for §1 were disproved. The benchmark
therefore keeps all 437 interferograms, removes the frame overlap with a **latitude cut at
31.4°**, restricts every split to the **shoreline AOI**, rebuilds four partitions
(geo k5/k10, temporal k5/k10), adds the paper's **RTh reconstruction protocol** as a second
evaluation track, and quarantines the old outputs.

**Settled:** all years 2019–2026 · AOI lat 31.25–31.75 / lon 35.38–35.46, baked into all four
partitions · cut at 31.4° · one interferogram per split · six training runs · stride-4 patches
already exist.

---

## 0a. The year decision — read this before §1

**Decided 2026-08-18: the benchmark uses all years, 2019–2026.**

§1 below is the original argument for restricting to 2019–2022 and is **kept as written** because
its evidence is real and still unexplained. What changed is the weight given to it:

| Considered | Outcome |
|---|---|
| AOI transfers to the newer years? | **Yes** — 90–98% of positives inside the box in every year (§10) |
| Newer years dilute positive density? | **No** — density per labelled scene rises slightly (§10) |
| Unlabelled scenes poison the splits? | **No** — disproved; 164 empty interferograms, none in any split (§10) |
| Newer scenes scored without a LiDAR gate? | **No** — disproved; `no_mask` falls back to the union (§10) |
| Newer scenes worse covered? | **No** — nodata flat across all eight years (§10) |
| **§1's precision collapse explained?** | **No.** Still unexplained. Phase-noise measurement deliberately deferred. |

**The risk is accepted knowingly:** the mechanism that halved precision from 2021 to 2024 has not
been identified, so it may contaminate the new benchmark the way it contaminated the old one. The
mitigations are the AOI, the LiDAR gate, and the fact that scene-level results are reported per
year — so if the collapse reappears it will be visible rather than averaged away.

**Consequences, all reflected below:** the "subset dictionary" of §3 is no longer a subset; the
`lidar_mask != "no_mask"` assertion must be **dropped**, not kept; §5's partition sizes and the
temporal boundaries are re-derived in §5a; and partition files lose the `_2019_2022` suffix.

---

## 0. Start here — state as of 2026-08-18

Every design question is now closed. No decision is waiting on you; the next move is code.

| | |
|---|---|
| **Done** | Stage 7 — old partitions preserved in place with `provenance`, `assets/PARTITIONS.md` written, implicit partition selection removed everywhere (§9). Old outputs pruned — 1.34 TB freed, metrics/figures/checkpoints intact (§8). Both split axes settled: geo cut 31.4° with `--geo_val_share 0.4`, temporal bounds `20210101 / 20210701` (§5). |
| **Next** | Step 5 — submit the six training runs (`submit_all.sh` needs a `clean22` kind pointing at the new partitions). Steps 3 and 4 are **done**: `grid_window`, per-split `aoi_window`, all three consumers agreeing, and the eight `*_clean.json` partitions generated and audited. 283 tests pass. |
| **Adopted** | §10 — AOI **lat 31.25–31.75 / lon 35.38–35.46**, baked into all four partitions, geo and temporal alike. Keeps 97%/91% of positives, deletes 45%/85% of scored North/South background, hold-out 6.1× denser, and measured 3× (North) to 20–41× (South) cleaner inside the box than outside. **Step 3 must therefore write `grid_window` with longitude bounds, not `grid_row_window`.** |
| **Loose end** | Move `outputs/predictions{,_positives}/` into the archive once pid 24957 is dead or the machine has rebooted (§8, *Status*). Cosmetic — do not let it gate step 2. |

The one command that encodes both split decisions, for when step 4 arrives:

```bash
sinkholes make-benchmark-partitions \
  --intf_dict assets/intf_coord.json \
  --patches_dir $D/patches --cut_lat 31.4 --seed 0 \
  --aoi 31.25 31.75 35.38 35.46 \
  --geo_val_share 0.4 --temporal_bounds 20240101 20250101 \
  --out_dir assets/
```

As of 2026-08-18 the first code and asset changes have landed, all of them from §9:
`assets/PARTITIONS.md`, the `provenance` stamps, and the removal of the default
partition from `train.py` and the three `train_*.sh`. Steps 2–4 are still untouched.

---

## 1. The evidence for restricting the years

`geo_k5_convlstm_ring3_3x`, the project's best model, on its own 18-scene geo test list,
object-level at confidence 0.5:

| Year of scene | Scenes | Mean precision | Mean recall |
|---|---:|---:|---:|
| 2021 | 3 | **0.91** | 0.93 |
| 2023 | 2 | 0.60 | 0.99 |
| 2024 | 7 | **0.44** | 0.91 |
| 2025 | 6 | **0.44** | 0.76 |

Recall is flat at 0.91–0.93 from 2021 to 2024 while precision halves. The model finds the
same objects; it is the *background* of the newer scenes that generates detections. One
scene, `20241127_20241208`, scores precision **0.09**.

**2019–2022 = 234 eleven-day interferograms** (2019: 59, 2020: 53, 2021: 58, 2022: 64),
197 labelled, 55,229 positive patches.

**The LiDAR filter is already satisfied by the year filter** — `no_mask` starts in 2024.
Keep it in the code as an **assertion**, not a selector.

---

## 2. What is already on disk — no patch regeneration needed

| Tree | Grid per scene | Ids present (2019–2022) | Size per scene |
|---|---|---:|---|
| `*_H200_W100_strpp2_11days_Aligned` | `(193, 89, 200, 100)` | 234 / 234 | 1.37 GB + 0.34 GB mask |
| `*_H200_W100_strpp4_11days_Aligned` | `(386, 177, 200, 100)` | 234 / 234 | 5.45 GB + 1.37 GB mask |

The stride-4 tree is what the RTh protocol needs: 386 × 177 = **68,322 tiles** per scene,
16 per interior pixel — the paper's quarter-patch stride. Both trees reconstruct onto the
same 19,500 × 4,525 canvas.

`DATASET_11DAY.md` calls that tree stale because it predates `sub_20260701.shp`. For this
year range it is not:

```
11-day polygons, 2019-01-01 .. 2022-12-31
  sub_20231001.shp : 22,980 polygons over 197 date pairs
  sub_20260701.shp : 22,980 polygons over 197 date pairs
  date pairs with differing counts: 0
```

Re-run that check as a gate in the eval script rather than trusting it once.

---

## 3. Stage 1 — the dictionary audit

*Originally "build `assets/intf_coord_2019_2022.json`". No longer applicable:*

> **Superseded by §0a.** With the range set to 2019–2026 this "subset" is the whole dictionary,
> so no new file is needed — `assets/intf_coord.json` is the input. What survives of this stage is
> the **audit**, and the LiDAR assertion must be **inverted**: 112 interferograms are `no_mask`
> and they are now in scope.

```python
keep = load_coord_dict()                                    # all 437
assert all(m["nonz_num"] != "none" for m in keep.values())  # 'none' means never digitised
# NOT an assertion any more -- 112 scenes are no_mask from mid-2024 and are deliberately kept.
no_mask = [i for i, m in keep.items() if m["lidar_mask"] == "no_mask"]
print(f"{len(no_mask)} scenes fall back to the LiDAR union gate")   # 112
```

Everything downstream takes `--intf_dict_path assets/intf_coord.json`. Chain containment is no
longer enforced by the dictionary but by the **temporal boundaries** (§5a) and the chain rule —
which matters more now, because chains can reach across the whole archive. `make-subset-dict` is
**not needed**; what is needed is that the audit above runs as a gate in the eval script.

---

## 4. Stage 2 — the latitude cut at 31.4°

### The overlap being removed

```
lat  31.79 ┬──────────────────────────────── North frame (aligned origin 31.79)
     31.44 │           ┬──────────────────── South frame (aligned origin 31.44)
     31.40 ├───────────┼─────  ← the cut
     31.25 ┴───────────┤       North frame ends (~19,440 aligned rows)
     31.17             ┴────── South scored extent ends
```

**31.25 → 31.44 is imaged by both frames — 0.19°, about 21 km.** Today's geo partition
splits by frame, so that band is in training *and* in test. The cut removes it.

Any line inside **[31.25, 31.44]** makes the sides disjoint. **31.4** — the paper's
large-dataset value (§3.3) — sits at the top of that range, which is what makes the
hold-out big: it hands the shared band to the *southern* side.

| Cut | k5 train | k5 hold-out | k10 train | k10 hold-out | hold-out per scene (median) |
|---|---:|---:|---:|---:|---:|
| 31.30 | 28,550 | 2,647 | 22,249 | 1,629 | 52 |
| 31.35 | 24,824 | 5,801 | 19,340 | 3,574 | 114 |
| **31.40** | **22,570** | **7,040** | **17,605** | **4,325** | **137** |

At 31.4 the hold-out is **larger than today's geo_k5 val set (5,846)**, and every South
scene keeps a median of 137 positive tiles instead of 52.

### Grid rows, both frames

One grid row advances the patch top by 100 px = 0.002777°.

| Frame | Aligned origin | Rows fully ≥ 31.4 | Rows fully ≤ 31.4 | Straddling (dropped) |
|---|---|---|---|---|
| North | 31.79 | `0 … 138` | `141 …` | 139, 140 |
| South | 31.44 | `0 … 12` | `15 …` | 13, 14 |

A patch is kept only when its whole 200-pixel height is on one side.

### What the cut costs, measured

All 2019–2022 interferograms, stride-2 tree, from `nonz_indices.json`:

| | Intfs | Total positives | ≥ 31.4 | ≤ 31.4 | straddling |
|---|---:|---:|---:|---:|---:|
| North frame | 123 | 38,405 | **27,212** | 10,116 | 1,077 |
| South frame | 111 | 16,824 | 4,073 | **11,816** | 935 |

Bold cells are usable under "one interferogram, one split": North above the line trains,
South below the line is held out. The discard is now **10,116 North positives** (below the
line) rather than the South's — the direct consequence of moving the cut up, and the reason
the hold-out nearly triples.

### Implementation — one window, three consumers

```python
# sinkholes/geo.py
def grid_row_window(frame, lat_min, lat_max, patch_size, stride) -> tuple[int, int]:
    """Half-open grid-row range whose patches lie entirely within [lat_min, lat_max]."""
```

Derived from `FRAME_ORIGINS`, **not** the per-scene raw `north`. Consumed in three places,
which must agree or a model is scored on ground it trained on:

1. **`dataprep/dataset.py`** — restrict loaded patches to the window. The single-frame
   positives-only path reads the `nonz` files, which carry no coordinates, so it must switch
   to grid + `nonz_indices.json` when a window is active. Ring negatives draw their annulus
   inside the window too.
2. **`inference/scenes.py`** — skip tiles outside the window, as `positive_tiles` does
   (`reconstruct.py:189`).
3. **`inference/outputs.py`** — crop the canvas to the window before scoring, **replacing**
   the hard-coded "South frame → northern half" at `outputs.py:133`. The geo hold-out band
   becomes `[31.17, 31.40]`.

> **Latent bug, do not reuse.** `SubsiDataset._load_spatial` (`dataset.py:413`) computes its
> threshold row from the *raw* scene `north`, while the grids on disk are aligned to
> `FRAME_ORIGINS` — up to ~700 rows out. Move it onto the new window function in the same
> change.

The window travels inside the partition JSON, per split:

```json
{
  "train": ["20190310_20190321", "..."],
  "val":   ["..."],
  "test":  ["..."],
  "lat_window": {"train": [31.40, 90.0], "val": [-90.0, 31.40], "test": [-90.0, 31.40]},
  "provenance": {"years": [2019, 2022], "k_prevs": 5, "cut_lat": 31.4, "seed": 0}
}
```

---

## 5. Stage 3 — the four partitions

Predecessors must also be 2019–2022, which costs interferograms at the start of 2019:
**150 chain-valid at k=5** (91 North / 59 South), **109 at k=10** (72 / 37).

### geo_k5 / geo_k10

Train = North frame, rows ≥ 31.4. Hold-out = South frame, rows ≤ 31.4.

| | Train | Hold-out | Split |
|---|---|---|---|
| k=5 | 91 intfs · **22,570** | 59 intfs · **7,040** | 76 : 24 |
| k=10 | 72 intfs · **17,605** | 37 intfs · **4,325** | 80 : 20 |

The train:hold-out ratio here is **fixed by the frames** — it cannot be pushed to 9:1
without letting a scene sit in two splits, which you ruled out. What is adjustable is the
val:test division inside the hold-out. Take **40 : 60** rather than 50 : 50 — validation
only has to select a checkpoint, the test list is what gets quoted:

| | val | test |
|---|---|---|
| k=5 | ~24 intfs · ~2,800 | ~35 intfs · ~4,240 |
| k=10 | ~15 intfs · ~1,730 | ~22 intfs · ~2,595 |

Two rules to preserve: **k10 ⊂ k5 per split** (assign at k5, intersect with the k10-valid
set) and a `_testeval` variant per partition.

### temporal_k5 / temporal_k10 — hold-out-heavy, and why the boundaries sit in 2021

Two structural facts drive this, both measured:

```
chain-valid k=5, by month   J  F  M  A  M  J  J  A  S  O  N  D   total
                     2019   0  0  2  3  3  5  6  4  6  3  3  4     39
                     2020   1  0  2  2  0  0  5  6  4  6  6  6     38
                     2021   6  4  6  3  2  4  6  3  3  0  0  0     37
                     2022   6  3  6  6  6  3  4  2  0  0  0  0     36

positive patches per year   2019: 19,220   2020: 17,838   2021: 14,849   2022: 3,322
```

There are **no 11-day chains between 2021-10 and 2022-02, and none starting after
2022-08**, and 2022 holds only 6% of the positives. A hold-out placed late is therefore
patch-starved no matter how many scenes it contains, and one placed in a chain gap is empty.
The boundaries have to move *earlier*, into the patch-rich years, to give val and test real
mass. A sweep of all 320 viable monthly boundary pairs (both k, val and test ≥ 6 intfs and
≥ 1,500 positives) gives:

| Layout | k5 train : val : test | k10 | note |
|---|---|---|---|
| 2019–21H1 / 21H2+22Q1 / 22Q2+ | 91 : 7 : 2 | 92 : 5 : 3 | train-heavy, hold-out starved — **rejected** |
| 2019–21H1 / 21H2 / 2022 | 91 : 0 : 8 | 92 : 0 : 8 | val dies in the chain gap |
| 2019–21Q3 / 21Q4+22H1 / 22H2 | 92 : 8 : 0 | 93 : 7 : 0 | test dies after 2022-08 |
| 2019–20 / 21-02→21-07 / 21-08+ | 74 : 15 : 11 | 78 : 10 : 12 | most train-heavy of the live options |
| 2019–20 / 21H1 / 21-08+ | 68 : 21 : 11 | 72 : 16 : 12 | best k10 val (9 intfs) |
| **2019–20 / 21H1 / 21H2+22** | **68 : 17 : 15** | **72 : 11 : 17** | **chosen** |

**Settled: train < 2021-01-01, val < 2021-07-01, test ≥ 2021-07-01.** Whole calendar years
for train, the first half of 2021 for val, everything from 2021-07 on for test:

| | Train (2019-01 → 2020-12) | Val (2021-03 → 2021-06) | Test (2021-07 → 2022-08) |
|---|---|---|---|
| k=5 | 77 intfs · 26,076 pos | 15 · 6,417 | 48 · 5,865 |
| k=10 | 50 · 17,223 | 6 · 2,747 | 35 · 4,004 |

Against the rejected layout this is **2.5× the val positives and 6.5× the test positives**,
and the test set goes from 21 scenes to 48 — which matters more than the patch count, because
the temporal axis is ranked on scene-level object metrics. Test spans 2021-07-03 → 2022-08-26
and is 12 scenes of 2021 plus 36 of 2022, so it still carries the year-shift the temporal
axis exists to measure.

The price is train: 77 interferograms / 26,076 positives at k5, against 102 / 35,777 before.
Training now sees only 2019–2020. That is the trade you asked for, and it is a real 27% cut
in training patches — watch the first run's train dice against the geo arm, which keeps its
full-year training pool.

**Chain containment.** A val or test interferogram is kept only when its whole chain
(current + k predecessors) starts on or after the **val** boundary — i.e. no held-out
sample's input was ever a training target. Chains reaching from test back into val are
allowed: val is held out too, so nothing leaks from training. This rule is what trims val
from the 21 (k5) / 24 (k10) interferograms inside the window down to 15 / 6.

Two things to accept, both consequences of the archive rather than the design:

- **k10 val is 6 interferograms** (2021-04-29 → 2021-06-23) / 2,747 positives — the same
  count as before but twice the patches, because the window now sits in a densely digitised
  stretch. Still thin for driving the LR schedule and early stopping. `--add_val_negatives`
  doubles the sample count; if it proves unstable, the documented fallback is to let *val
  only* chains reach back into train, which restores **24 interferograms / 9,452 positives**
  at k10 (k5: 25 / 9,701) while the test split stays strict.
- **k10 train is 50 interferograms.** The k5/k10 comparison on the temporal axis is now also
  a 77-vs-50 scene comparison; keep that in mind before reading a k10 deficit as a
  context-length result.

### 5a. Re-derived for 2019–2026 — supersedes the sizes above

*All figures above in §5 were computed on 2019–2022 and are kept for the record only. These are
the live numbers: all years, chain-contained, counted as **positives inside the AOI**.*

**Chain-valid interferograms** (`find_11day_sequences`, all years):

| k | Chain-valid | North | South | AOI positives |
|---|---:|---:|---:|---:|
| 5 | **214** | 128 | 86 | 59,113 |
| 10 | **166** | 108 | 58 | 47,202 |

Against §5's 2019–2022 figures (k5 150 / k10 109) that is **+43% and +52% more scenes**. Chain
validity by year at k5: 2019:39 2020:38 2021:37 2022:36 2023:13 2024:24 2025:17 2026:10 — the
newer years contribute fewer chains than their scene count suggests, because a chain needs six or
eleven consecutive 11-day acquisitions and the newer archive has more gaps.

**Geo** is unchanged in method — train = North above 31.4°, hold-out = South below — and simply
gets bigger: 38,101 train / 15,189 hold-out AOI positives (§10), still ~71:29, with val:test
split 40:60 inside the hold-out.

**Temporal boundaries must be re-derived.** The settled `20210101 / 20210701` were chosen when the
archive ended in 2022 and 2022 held only 6% of positives. Applied to 2019–2026 they now yield
**train 77 / test 112 interferograms** — a hold-out larger than the training set, which is not a
benchmark. Sweep of candidate layouts (intfs / AOI positives):

| Layout | k | Train | Val | Test |
|---|---|---|---|---|
| 2019–20 / 21H1 / 21H2+ *(old, now broken)* | 5 | 77 / 24,817 | 15 / 6,078 | **112 / 25,164** |
| 2019–21 / 2022 / 2023+ | 5 | 114 / 36,378 | 27 / 2,213 | 64 / 19,512 |
| | 10 | 82 / 26,993 | **17 / 394** | 57 / 17,996 |
| 2019–22 / 2023 / 2024+ | 5 | 150 / 39,601 | 13 / 3,384 | 51 / 16,128 |
| 2019–22 / 2023–24 / 2025+ | 5 | 150 / 39,601 | 37 / 10,360 | 27 / 9,152 |
| **2019–23 / 2024 / 2025+** | **5** | **163 / 42,985** | **21 / 5,917** | **27 / 9,152** |
| | **10** | **122 / 32,590** | **16 / 4,735** | **22 / 8,001** |

**Recommended: train < 2024-01-01, val < 2025-01-01, test ≥ 2025-01-01.** It gives the largest
training pool at both k, a val set that is healthy at k10 (**16 interferograms**, against the 6
that §11 flags as the plan's biggest risk — that risk is retired by this layout), and a test set
of 22–27 scenes with 8–9k positives.

It also puts the test set squarely on **2025–2026**, which is where the LiDAR mask falls back to
the union and digitisation is thinnest. That is deliberate: it is the year-shift the temporal axis
exists to measure, and under the §0a decision it is exactly the regime the benchmark is being
extended to cover. If §1's unexplained collapse is real, this layout is where it will surface —
which is the point.

**Still to check before generating:** the 2019–21 / 2022 / 2023+ layout dies at k10 (val 394
positives), confirming that val must not sit in a chain-sparse window. Re-run the sweep if the
AOI or the chain rule changes.

### 5b. `geo_crossview` — the North band below the cut

*Added 2026-08-18. The 31.4° cut discards every North patch below the line. That band is
**larger than the geo hold-out itself** — 11,011 AOI positives at k5 against 9,882, and 9,205
against 6,553 at k10, a quarter of all North AOI positives. It is now kept as a fourth
evaluation set.*

**It can never be training data.** Within the AOI the band is lat **31.25–31.40**, and the South
hold-out is lat **31.25–31.40** — the identical 17 km strip, the same sinkholes, imaged from the
other orbit. Training on it would put hold-out ground into train, which is the exact leak the cut
exists to prevent. Evaluation is the only legitimate use.

| Split | Frame | Latitude | k5 intfs / AOI pos | k10 |
|---|---|---|---|---|
| train | North | 31.40 – 31.75 | 128 / 32,515 | 108 / 27,443 |
| val | South | 31.25 – 31.40 | 40% of hold-out | — |
| test | South | 31.25 – 31.40 | 60% of hold-out | — |
| **crossview** | **North** | **31.25 – 31.40** | **117 / 11,011** | **98 / 9,205** |

**Reported separately, never merged into test.** Merging would score the same physical sinkholes
twice from two frames, making object counts and scene averages correlated rather than
independent.

#### What it actually measures — read before quoting it

**116 of the 128 crossview interferograms are also training interferograms.** They contribute
their above-cut patches to train and their below-cut patches here. That is a deliberate exception
to "one interferogram, one split", and it is only acceptable because crossview is never used to
select a checkpoint or to report a headline number.

So crossview is **seen scenes on unseen ground**, while test is **unseen scenes on the same
unseen ground**:

| | Scenes | Ground |
|---|---|---|
| test (South) | never seen | never trained on |
| crossview (North band) | **seen in training** | never trained on |

The two are scored on *identical ground*, so the difference between them isolates one variable:
**how much the model gains from having seen the acquisition before.** That makes it a
memorisation probe, not a generalisation measure.

- crossview ≈ test → the model is reading sinkhole morphology. The benchmark is measuring what it
  should.
- crossview ≫ test → the model is exploiting scene-specific signal — atmospheric screen,
  processing artefacts — that happens to correlate with the labels. The bands are adjacent with
  no buffer and phase screens correlate over tens of km, so this is a live possibility.

That second case is worth having a probe for: it is a candidate mechanism for §1's unexplained
precision collapse, which no measurement so far has accounted for. A large crossview–test gap
would be the first real evidence for it.

**Never** quote crossview as the model's score, and never average it with test.

#### Implementation

The geo partitions gain a `crossview` list and its own window entry:

```json
{
  "train": ["..."], "val": ["..."], "test": ["..."],
  "crossview": ["...North ids, mostly the same as train..."],
  "aoi_window": {
    "train":     [31.40, 31.75, 35.38, 35.46],
    "val":       [31.25, 31.40, 35.38, 35.46],
    "test":      [31.25, 31.40, 35.38, 35.46],
    "crossview": [31.25, 31.40, 35.38, 35.46]
  }
}
```

`PARTITION_SPLITS` in `dataprep/partition.py` is currently `("train", "val", "test")` and
`load_partition_split` rejects anything else — it must learn `crossview`. Training must **never**
load it: it is evaluation-only, and the overlap-check in `train.py` that errors on a scene in two
splits has to exempt it explicitly, with a comment pointing here. Temporal partitions have no
crossview; the concept is geo-only.

### 5c. Generated 2026-08-18 — the real numbers

`sinkholes make-benchmark-partitions` has been run and the eight files are in `assets/`.
Counts are what the command printed, positives **inside each split's window**:

| Partition | train | val | test | crossview |
|---|---|---|---|---|
| geo_k5 | 127 / 32,515 | 34 / 4,093 | 51 / 5,789 | 117 / 11,011 |
| geo_k10 | 107 / 27,443 | 23 / 2,649 | 35 / 3,904 | 98 / 9,205 |
| temporal_k5 | 163 / 42,985 | 21 / 5,917 | 27 / 9,152 | — |
| temporal_k10 | 122 / 32,590 | **16 / 4,735** | 22 / 8,001 | — |

They match §5a's independently derived sweep, which is the cross-check that matters: the
generator reproduces numbers worked out by separate code and separate reasoning.

**One rule conflict surfaced and was resolved.** "k10 ⊆ k5 per split" and chain containment
disagree: three scenes (`20240227_20240309`, `20240319_20240330`, `20240320_20240331`) have
5-step chains starting after the val boundary but **10-step chains reaching back into 2023** —
training territory. Keeping them would have put a training target into a held-out sample's
input. **Containment outranks the subset rule**; the three are dropped at k10, which is safe
for the subset rule because dropping only ever removes. Without this the k10 temporal val would
have been 19 interferograms instead of 16 and would have leaked.

The geo val:test division is by **positive mass**, not scene count, so `--geo_val_share 0.4`
means what it says whatever the per-scene spread is (realised: 4,093 of 9,882 = 41%).

### The generator

None of the current partition files has a script in the repo — `make-partition` only does a
positive-patch-share train/val cut. Add:

```bash
sinkholes make-benchmark-partitions \
  --intf_dict assets/intf_coord.json \
  --patches_dir $D/patches --cut_lat 31.4 --seed 0 \
  --aoi 31.25 31.75 35.38 35.46 \
  --geo_val_share 0.4 --temporal_bounds 20240101 20250101 \
  --out_dir assets/
```

writing all eight files (`partition_{geo,temporal}_k{5,10}_clean.json` + `_testeval`)
with `aoi_window`, `provenance`, and — for the two geo partitions — the `crossview` list of §5b, and printing the count tables so a partition is never
adopted without its sizes being seen. It must count positives **inside the window**, not
`nonz_num` (which is whole-scene) — the subtle bug this command exists to prevent.

Pin it with a test: splits disjoint, k10 ⊆ k5, every held-out chain starting after the val
boundary, no train patch inside the hold-out latitude band.

---

## 6. Stage 4 — training

No code change beyond the row window and the new partition files.

| Job | Partition | Architecture | Negatives | Purpose |
|---|---|---|---|---|
| `clean_geo_k5_convlstm_ring3` | geo_k5 | ConvLSTM h256 | ring 1–3, 1:1 | **the anchor** |
| `clean_geo_k5_convlstm_ring3_3x` | geo_k5 | ConvLSTM h256 | ring 1–3, 3:1 | best config on the old data |
| `clean_geo_k5_single_ring3` | geo_k5 | single-frame U-Net | ring 1–3, 1:1 | temporal-context control |
| `clean_geo_k10_convlstm_ring3` | geo_k10 | ConvLSTM h256 | ring 1–3, 1:1 | k5 vs k10 |
| `clean_temporal_k5_convlstm_ring3` | temporal_k5 | ConvLSTM h256 | ring 1–3, 1:1 | the temporal axis |
| `clean_temporal_k10_convlstm_ring3` | temporal_k10 | ConvLSTM h256 | ring 1–3, 1:1 | k5 vs k10 there |
| `clean_temporal_k5_single_ring3` | temporal_k5 | single-frame U-Net | ring 1–3, 1:1 | temporal-context control, temporal axis |
| `clean_geo_k5_tattn_ring3` | geo_k5 | attention, fuse 0 | ring 1–3, 1:1 | attention vs recurrence |
| `clean_geo_k5_tattn_hybrid_ring3` | geo_k5 | attention + ConvLSTM | ring 1–3, 1:1 | the hybrid, which tied ConvLSTM on old geo |
| `clean_temporal_k5_tattn_ring3` | temporal_k5 | attention, fuse 0 | ring 1–3, 1:1 | artefact suppression under year shift |
| `clean_temporal_k5_tattn_hybrid_ring3` | temporal_k5 | attention + ConvLSTM | ring 1–3, 1:1 | hybrid on the temporal axis |
| `clean_geo_k10_tattn_ring3` | geo_k10 | attention, fuse 0 | ring 1–3, 1:1 | attention at k=10 |
| `clean_geo_k10_tattn_hybrid_ring3` | geo_k10 | attention + ConvLSTM | ring 1–3, 1:1 | tattn-vs-hybrid at k=10 |
| `clean_temporal_k10_tattn_ring3` | temporal_k10 | attention, fuse 0 | ring 1–3, 1:1 | longest history under the year shift |

**Fourteen runs, ≈150–170 GPU-hours** at the 100-epoch budget (≈90–105 at the old 60) (was six / ≈30 as first planned; the attention arms, the
temporal single-frame control and the k=10 attention arms were added 2026-08-18). Submit all
of them with:

```bash
bash scripts/submit_all.sh clean22 --submit
```

Host memory is sized per job from a measurement — the old geo_k5 pool's ~41 GB peak is 1.61×
the raw sample bytes — and capped at 128 GB. Only the 3:1 arm sits near that cap.

Both attention arms use the **same** `TemporalAttentionUNet` (`train_tattn.sh`); they differ
only in `RECURRENCE` — `none` is pure attention, `convlstm` is the hybrid.

The attention arms sit at `FUSE_SKIPS=0`, matching the arms `PREDICTIONS.md`'s result came
from. `FUSE_SKIPS=2` exists and is one env var away, deliberately not queued.

Attention runs at **both depths**: k=10 is where the artefact-suppression argument should bite
hardest, since a longer history is more evidence about what is temporally persistent. Every
k=10 attention arm takes **gmem 48G** — the template warns below that for `K_PREVS>=10`, and
`submit_all.sh` overrides its 36G `#BSUB` directive on the command line. `temporal_k10` hybrid
is the one absent for symmetry: the tattn-vs-hybrid contrast already exists at k=10 on geo and
at k=5 on temporal, so both comparisons are covered once.

Both attention architectures run on **both axes** on purpose: their hypothesis is that
temporally inconsistent artefacts get suppressed while persistent subsidence survives, and the
temporal axis is where the year shift actually stresses that. Geo-only would leave the
architecture's own claim untested.

Hyperparameters are pinned across all fourteen by one shared `$C_HYP` override
(2026-08-18): **lr 1e-5, 100 epochs, patience 40**. That is 10x the learning rate every
previous run used, so the old loss curves are not a guide to what these should look like.
`EPOCHS=100` is a ceiling — the 2026-08-10 batch stopped all 13 of its jobs on patience, and a
10x learning rate should move the loss earlier still, so patience 40 is what will decide these
runs in practice. Patience was doubled with the epoch budget so early stopping stays a
proportional rule rather than becoming twice as aggressive.

Everything else is template default (b128, seed 42, plateau schedule, `--amp`,
`--save_best_only`, `--resume auto`), plus **`VAL_NEGS=yes` on every run** — a positives-only
val set cannot see the false positives that are the whole problem, and with `temporal_k10`'s
6-interferogram val it matters more, not less. New `submit_all.sh` kind: `clean22`.

---

## 7. Stage 5 — evaluation

### (a) Scene-level, stride 2 — continuity

The existing protocol on the new partitions; every model scored on the k10 test list so k5
and k10 share ground. Comparable to `PREDICTIONS.md` in protocol only — the scene list is
different, so these numbers replace that table rather than extend it.

### (b) RTh, stride 4 — the paper's protocol

| | Paper | Ours today |
|---|---|---|
| Stride | quarter patch — 16 tiles/pixel | half patch — 4 tiles/pixel |
| Confidence | **fraction of tiles voting 1** | mean predicted probability |
| Threshold | RTh ∈ {0.125, 0.25, 0.5} = ≥2, ≥4, ≥8 of 16 | `--recon_th` on the mean |
| Object tolerance | ITh 0.7 / b 5, and soft 0.5 / b 10 | ITh 0.7 / b 5 |
| Aggregation | ground-truth-area weighted | unweighted mean over scenes |

"RTh 0.25" is **not** our "recon_th 0.25" — one counts patches, the other averages
probabilities. Four code changes; **items 1 and 4 landed 2026-08-19**, items 2 and 3
are performance only and remain open:

1. ✅ **DONE 2026-08-19. Vote mode** in `reconstruct.py`: `average="vote"` binarises each
   tile at `vote_threshold` (0.5) before accumulating, divided by the per-pixel tile count.
   Values land on {0, 1/16, …, 1} — the paper's Confidence Factor exactly. Exposed as
   `eval-scenes --recon_average vote`, and as `PROTOCOL=rth` in `run_eval.sh`, which sets
   both stages together. Rejects `blend=hann` (a vote is unweighted) and, at the script
   level, `DATA_STRIDE != 4`. Covered by `tests/test_reconstruction.py`.
2. ⏳ **OPEN — performance only, no number changes.** **Batched tile inference.** One tile
   per forward pass today; 68,322 tiles per scene is ~4× a 3-hour eval. Mitigated for now
   by the AOI, which keeps only 13.7% of a geo canvas (9,360 tiles/scene), and by the
   12:00/16:00 walltimes the `eval5` rungs carry. Batch 32–64 when it becomes the
   bottleneck — also the cheapest speedup available to the stride-2 evals.
3. ⏳ **OPEN — performance only, no number changes.** **Banded streaming.** `scenes.py`
   loads each timestep's whole grid as float32: 5.47 GB × 11 = **60 GB** at k10 (verified
   on disk 2026-08-19). Worked around by the `eval5` host-memory rungs — 112 GB at k5,
   144 GB at k10 — rather than fixed. Process the grid in row bands, deciding the
   normalisation branch once per scene from a cheap first pass.
4. ✅ **DONE 2026-08-19. `eval-outputs`:** `--rth` sweeps RTh {0.125, 0.25, 0.375, 0.5}
   under both tolerance settings (ITh 0.7/b 5 primary, ITh 0.5/b 10 soft, also reachable
   via `--extra_tolerance ITH:BUFFER`); adds `weighted_recall`/`weighted_precision`
   (GT-area weighted) beside the unweighted means; and passes `round_ndigits=None` into
   `object_level_evaluate`, so the per-scene `round(…, 2)` no longer quantises scenes
   before they are averaged. The patch-level path keeps the old rounding by default.
   The JSON gains `threshold_kind`, `per_intf_by_tolerance` and `summary_by_tolerance`
   while `per_intf`/`summary` keep the historical schema. Covered by
   `tests/test_rth_protocol.py`.

Then the polygon-level analysis at RTh 0.375 — detection rate against area, roundness,
normalised phase σ, mean phase gradient and year. `object_level_evaluate` already collects
per-object features. Recall is the primary metric here, for the reason that bit us: the
reconstruction is dominated by unannotated ground, so precision partly measures the
digitisation.

### (b2) `geo_crossview` — the memorisation probe

Score the geo models on the `crossview` split (§5b) with the **same protocol as track (a)**, on
the same ground as the geo test set. Report it as its own row, next to test, never averaged in:

| Model | test (South) | crossview (North band) | gap |
|---|---|---|---|

The gap is the number of interest. A small gap says the benchmark is measuring sinkhole
morphology; a large one says the model is reading scene-specific signal, and the geo result needs
re-examining before it is quoted. See §5b for why this is not a generalisation measure.

### (c) Patch-level on the partition

`test-patches --partition_file … --split test`. Positives-only by definition — read
`POSITIVES_ONLY_EVAL.md` before quoting it against anything.

---

## 8. Stage 6 — quarantine and prune the old outputs

`outputs/` holds **1.3 TB**, all of it produced on scene lists containing 2023–2026 data.
Measured breakdown of the two prediction trees (1.34 TB, 3,307 files):

| File kind | Size | Files | Recomputable? |
|---|---:|---:|---|
| `_image.npy` (input stacks) | **793.6 GB** | 410 | yes — figures already rendered; metrics never read it |
| `_pred_th.npy` | **182.6 GB** | 546 | yes — one threshold on `_pred.npy` |
| `_gt.npy` | 182.6 GB | 546 | yes — rebuildable from the mask grids |
| `_pred.npy` (confidence) | 182.6 GB | 546 | **no** — 3 GPU-hours per scene to rebuild |
| figures, shapefiles, metrics JSON | 2.3 GB | 1,259 | keep |

**Decision: every `.npy` goes.** These rasters are predictions over scene lists that include
the noisy 2023–2026 interferograms; they will not be quoted, re-scored or re-figured, so the
confidence maps are not worth 183 GB of standing cost. What survives is what actually gets
read: metrics JSON, object-level results, curves, shapefiles, figures and checkpoints.

```bash
mkdir -p outputs/archive_2019_2026_noisy_data
mv outputs/predictions outputs/predictions_positives \
   outputs/2026-08-0[3-9] outputs/2026-08-1[01] \
   outputs/archive_2019_2026_noisy_data/
find outputs/archive_2019_2026_noisy_data -name "*.npy" -delete   # ≈1.34 TB, 2,048 files
```

The `mv` is a rename inside one filesystem — instant, and reversible. The `find -delete` is
not: it frees **≈1.34 TB** and leaves ~2.3 GB. The thing given up is re-scoring a past
evaluation at a different threshold or a different object-level rule without a GPU; accepted
deliberately, since no old number carries into the clean benchmark anyway.

### Status — executed 2026-08-17

**Done.** All 2,048 `.npy` deleted; ~1.34 TB recovered; no `.smbdelete*` debris. Surviving and
verified: 29 `olm_results_*.json`, 64 JSON, 2,595 figures, 75 CSV, plus shapefiles and
checkpoints. The six training-run directories (`2026-08-03` … `2026-08-11`) are inside
`outputs/archive_2019_2026_noisy_data/`.

**Not done — `outputs/predictions/` and `outputs/predictions_positives/` are still one level
up, not inside the archive.** A stranded macOS Virtualization service left over from Docker
Desktop (pid 24957 on 2026-08-17) holds an open handle on `predictions` *and on each of its
21 model subdirectories*, and SMB refuses to rename a directory with open handles. Moving the
children individually fails for the same reason; the fix is to kill the process (`kill -9`,
`-TERM` was ignored) or reboot, then:

```bash
cd outputs && mv predictions predictions_positives archive_2019_2026_noisy_data/
```

Nothing depends on this — the space is already recovered and no result is at risk. It only
decides whether the two trees read as quarantined. Check for a Finder window under `outputs/`
first; that re-blocks the rename on its own.

Also required, or the archive silently breaks things:

- `outputs/archive_2019_2026_noisy_data/README.md` — why it is quarantined (the §1 table),
  what was pruned, and the fact that **no raster survives**: any figure or re-score against
  these runs means re-running `run_eval.sh` on the old scene lists.
- Path prefixes in `MODEL_RUNS.md`, `RESULTS.md`, `PREDICTIONS.md`, `scripts/eval/PRESETS.md`
  and `submit_all.sh`'s `rescore` kind all point at `outputs/predictions/...` and need the
  new prefix.
- `docs/OUTPUTS.md` should record the pruning policy so the next batch is written prunable
  from the start (`--save_confidence` without the input stacks unless figures are wanted).

---

## 9. Stage 7 — preserve the old partitions and kill implicit selection

*Added 2026-08-18, done the same day.*

The clean partitions are new **files** (`*_clean.json`), not overwrites, so the old
family was never at risk of being clobbered — and all of it is git-tracked besides. The real
hazard is the opposite one: three generations of partition now sit side by side in `assets/`,
and a run that does not name one explicitly picks an old one **silently** and looks fine.

**Decision: keep every old file exactly where it is.** Moving them to a `legacy/` subdirectory
would read more cleanly, but it breaks the eight hard-coded paths below *and* every command
recorded in `MODEL_RUNS.md`, `PRESETS.md` and the archived run logs. Provenance goes inside
the files instead, and the ambiguity is removed at the entry points.

### What was found

| Generation | Files | Splits | Years |
|---|---|---|---|
| 1 · `partition_20_05_*` | 5 | train/val only | 2019–2021 |
| 2 · `partition_{geo,temporal}_k{5,10}{,_testeval}` | 8 | train/val/test | **2019–2026** |
| 3 · `*_clean` (this plan) | 8 | + `aoi_window`, `provenance` | 2019–2026 |

Eight sites defaulted to a generation-1 or -2 file: `train.py:471` (→
`partition_20_05_13h45.json`), `train_{convlstm,control}.sh` and `train_tattn.sh`, and
`submit_all.sh:108,109,192`; plus doc references in `PIPELINE.md`, `POSITIVES_ONLY_EVAL.md`,
`PREDICTIONS.md` and `train/PRESETS.md`.

### Done

1. **`assets/PARTITIONS.md`** — registry of all three generations: sizes, why 1 and 2 are
   frozen, the `_testeval` and `_k5`/`_k10` conventions, and the no-implicit-selection rule.
2. **`provenance` stamped into the 8 generation-2 files** — `generation: "2019_2026_noisy"`,
   year span, axis, `k_prevs`, split sizes, `superseded_by`, and a note recording both
   reasons they were retired (the 2023–2026 precision collapse of §1, and the geo split
   dividing by frame so the 31.25–31.44° band sits on both sides). Interferogram lists
   untouched — the diff is additive, only the closing brace moves. Generation 1 predates the
   convention and was left byte-identical; the registry covers it.
3. **No entry point selects a partition any more.** `train --partition_mode preset_by_intf`
   raises without `--partition_file`; the three `train_*.sh` use `${PARTITION:?…}` instead of
   a default. Both error messages point at `PARTITIONS.md`.
4. **Latent bug fixed in passing.** The `K_PREVS`-vs-partition guards globbed on `*_k5.json` /
   `*_k10.json`, which **does not match `partition_geo_k5_clean.json`**. Six case arms
   across the three scripts now also match `*_k5_*.json` / `*_k10_*.json`. Left as-is, every
   generation-3 run would have skipped the guard that stops k5 data being fed to a k10 model.
5. **Removed** `assets/{selected_30,south_15}_interferograms.txt`, unreferenced anywhere in
   code or docs. `first_25_chain.txt` and `north_15_interferograms.txt` are equally
   unreferenced and were left alone.

### Still to do

- `submit_all.sh:108,109,192` still bind `TEMPORAL_K5` / `GEO_K5` / `G10_BASE` to
  generation-2 paths. Those name the **archived** batches and should stay; the new `clean22`
  kind (§6) must define its own generation-3 constants rather than reuse them.
- The doc references in `PIPELINE.md`, `POSITIVES_ONLY_EVAL.md`, `PREDICTIONS.md` and
  `train/PRESETS.md` are examples against old results and stay correct — but each wants a
  line saying which generation it is quoting, folded into the §8 doc-prefix sweep.
- `make-benchmark-partitions` must write `provenance` with `generation: "all_years_clean"`,
  the `aoi` box and the temporal bounds, so generation 3 is self-identifying from day one.

---

## 10. Stage 8 — a clean shoreline AOI (measured and adopted 2026-08-18)

*Added 2026-08-18. Motivation: restrict training **and** scoring to a spatial box known to be
well-imaged and well-digitised, rather than only to good years. The legacy trees were surveyed
for an existing crop first (none was found), then the AOI was derived by bounding the positive
patches. The numbers below are measured, not projected.*

### What the legacy search turned up

Searched `/Volumes/rudich/Rudich_Collaboration/{sinkholes,deadsea_sinkholes_data}` for any
prior AOI. **No legacy crop was ever used to select training ground.** Four things exist, and
only one is spatial:

| Artefact | What it is | Reusable as an AOI? |
|---|---|---|
| `x0_N=35.37 / x0_S=35.32`, `nx=4500` in `prepare_intrfrgrm_pathches.py` | the **longitude** crop, already in production as `geo.FRAME_ORIGINS` + `X_CROP_COLS` | already applied — it *is* the canvas |
| `deadsea_sinkholes_data/crops_2019_2021/` + `save_crops_remote.py` | 87 × **289 × 289** crops, `PAD_DEG 0.004` around `Point(35.390708, 31.400092)` | no — ~0.9 km, a viewer/demo strip |
| `visualize_2023_intfs.py:40` `DEFAULT_LON 35.35–35.45`, `DEFAULT_LAT 31.38–31.47` | a hand-chosen "look here" window for inspecting 2023 scenes | **the only spatial statement of where the data looks good** — but undocumented and unmeasured |
| `clean_patches.py` (ported to `dataprep/clean_patches.py`), `has_consecutive_zeros`, `--treat_nodata_regions` | quality filters keyed on **zero-valued phase = nodata/decorrelation** | not spatial — per-polygon and per-patch |

So the project already agrees that *zero phase is the decorrelation signal* — `clean_patches`
drops a label polygon when `zero_frac > 0.7` inside it, `_add_null_patches` uses
`has_consecutive_zeros`, and `--treat_nodata_regions` appends a validity channel. What has
never been done is aggregating that signal **spatially** to answer "which ground is reliably
good across the whole 2019–2022 stack".

### The finding that matters — the good ground is the ground the 31.4° cut gives away

Both human-chosen windows point at the same place:

```
  crops_2019_2021 centre     35.3907, 31.4001
  2023 viewer default        35.35–35.45, 31.38–31.47
  the frame overlap band              31.25–31.44   <- §4
  the cut                                   31.40
```

The strip everyone actually looks at sits **inside the 31.25–31.44 overlap band**, straddling
the cut. §4 chose 31.4 precisely because it is at the top of that band and so hands the shared
strip to the southern side, at the cost of discarding **10,116 North positives below the
line**. Those discarded positives are in the best-imaged, most densely digitised ground in the
dataset. That is a defensible trade — a scene cannot sit in two splits — but it should be
recorded as a deliberate one, not discovered later.

### Measured — bound the positive patches, 2019–2022

Per the agreed approach: take the positive patches as the definition of interesting ground and
bound them with a rectangle. Source is `nonz_indices.json` on the stride-2 tree — already on
disk, no rasters read. A patch counts as inside only when its **whole** 200×100 footprint is,
the same rule as the latitude cut. Reproduce with:

```bash
python scripts/data/measure_aoi.py --years 2019 2022 --cut_lat 31.4
```

Where the positives actually are, per frame:

| Frame | Intfs | Positives | Latitude span | Longitude span |
|---|---:|---:|---|---|
| North | 109 | 38,405 | 31.2485 – 31.7456 | 35.3700 – 35.4950 |
| South | 88 | 16,824 | 30.9568 – 31.4400 | 35.3519 – 35.4352 |

**The belt is long in latitude and narrow in longitude**, so an AOI is mostly a *longitude*
trim. And 95% of South positives sit above **31.2345** — the 0.28° tail below that holds ~5%.

Positives kept vs scored grid cells kept:

| AOI (lat / lon) | North pos | North cells | South pos | South cells |
|---|---:|---:|---:|---:|
| none | 100% | 100% | 100% | 100% |
| 31.23–31.76 / 35.375–35.47 | 97.9% | 71.0% | 92.4% | 20.9% |
| **31.25–31.75 / 35.38–35.46** | **96.7%** | **57.0%** | **91.0%** | **17.4%** |
| 31.25–31.75 / 35.38–35.45 | 91.0% | 49.7% | 91.0% | 17.4% |
| 31.30–31.75 / 35.38–35.45 | 81.1% | 44.9% | 68.5% | 12.7% |

### It composes with the 31.4° cut — the tension in the section above does not survive contact

The worry was that an AOI confined to the overlap band and a cut splitting that band would
leave nothing. It does not happen: the belt runs from 31.25 to 31.75, so the cut still has
0.35° of North above it and 0.15° of South below it.

Geo split, train = North above 31.4, hold-out = South below 31.4 (the "none" row reproduces
§4's 27,212 / 11,816 exactly, which is what validates the grid arithmetic here):

| AOI | Train pos | North cells | Hold-out pos | South cells | Split |
|---|---:|---:|---:|---:|---:|
| none (plan as written) | 27,212 | 12,371 | 11,816 | 16,020 | 70:30 |
| 31.23–31.76 / 35.375–35.47 | 26,407 | 8,576 | 10,533 | 2,891 | 71:29 |
| **31.25–31.75 / 35.38–35.46** | **25,950** | **6,820** | **10,495** | **2,340** | **71:29** |
| 31.25–31.75 / 35.38–35.45 | 23,757 | 5,952 | 10,495 | 2,340 | 69:31 |

**Recommended AOI: lat 31.25–31.75, lon 35.38–35.46.** It costs 4.6% of training positives and
11% of hold-out positives, leaves the 70:30 frame ratio untouched, and removes 45% of the North
and 85% of the South *raw canvas*.

> **Superseded below.** Evaluation already applies a LiDAR gate, so the raw-canvas figures
> overstate the gain. Against the gated area — what is actually scored — the reduction is
> **34% (North) / 80% (South)**. Use those.

That last figure is the point of the whole exercise. Positive density — positives per scored
grid cell per labelled scene (109 North / 88 South), which is inversely what a false positive
has to land on:

| AOI | Train density | Hold-out density |
|---|---:|---:|
| none | 0.0202 | 0.0084 |
| 31.23–31.76 / 35.375–35.47 | 0.0282 | 0.0414 |
| **31.25–31.75 / 35.38–35.46** | 0.0349 | **0.0510** |

**The hold-out gets 6.1× denser.** §1 diagnosed the precision collapse as the *background*
generating detections while recall held flat; the geo hold-out is where scene-level precision
is quoted, and this deletes 85% of its background for 11% of its positives. This is a larger
lever on precision than the year restriction, and independent of it.

### Extended to all 437 interferograms (2019–2026)

*Requested 2026-08-18: fit the AOI on everything available, so the newer interferograms can be
used rather than discarded.* Re-run with `--years 2019 2026 --by_year`.

The belt does not move. Across all years the positives still occupy the same box, and the
recommended AOI holds **90–98% of positives in every single year**, including the ones the plan
was written to exclude:

| Year | Intfs | Labelled | Positives | Inside AOI | Kept |
|---|---:|---:|---:|---:|---:|
| 2019 | 59 | 59 | 19,220 | 18,351 | 95.5% |
| 2020 | 53 | 53 | 17,838 | 16,854 | 94.5% |
| 2021 | 58 | 49 | 14,849 | 14,021 | 94.4% |
| 2022 | 64 | **36** | 3,322 | 3,223 | 97.0% |
| 2023 | 63 | **15** | 4,121 | 3,985 | 96.7% |
| 2024 | 64 | **30** | 8,321 | 8,120 | 97.6% |
| 2025 | 58 | **19** | 6,893 | 6,232 | 90.4% |
| 2026 | 18 | 12 | 4,371 | 4,138 | 94.7% |

**The AOI is not a 2019–2022 artefact** — it was fitted on those years and transfers to every
other year without adjustment. Using all years raises the pool from 55,229 to **78,935**
positives (+43%), and under the AOI + 31.4° cut the geo split becomes:

| Years | Train pos | N cells | Hold-out pos | S cells | Density N/S |
|---|---:|---:|---:|---:|---|
| 2019–2022 | 25,950 | 6,820 | 10,495 | 2,340 | 0.0349 / 0.0510 |
| **2019–2026** | **38,101** | 6,820 | **15,189** | 2,340 | **0.0370 / 0.0532** |

Density is per labelled scene, so the newer years are **not** diluting it — it goes slightly
*up*. On this metric the newer interferograms look as good as the old ones.

### Two hypotheses about the newer years, both tested and both **wrong**

*Recorded because they are plausible, were briefly written into this plan as findings, and do
not survive contact with the code. Anyone re-reading §1 will think of them again.*

**Hypothesis A — "unlabelled scenes are scored as all-negative."** Ground truth does thin out
sharply (2019–2021: 161 of 170 interferograms carry polygons; 2022–2026: 112 of 267; 2023 alone
is 15 of 63). The inference was that an undigitised scene enters a split, contributes a blank
target, and turns real detections into false positives.

**It does not happen.** Two independent guards:

- `_load_temporal` (`dataset.py:317`) takes sample coordinates from `coord_tids = [intf_id]` —
  the current frame only. An interferogram with no entry in `nonz_indices.json` yields `rc == []`
  and returns `(None, None)`, which the caller skips at `dataset.py:252`. Predecessors are loaded
  as input context and **never contribute coordinates of their own**. So an empty interferogram
  can only ever appear as chain history, exactly as intended.
- The partitions never list one anyway. Of 437 interferograms, **164 have zero positive patches
  and none of them appears in any split** of any of the four partitions — train, val or test.

**Hypothesis B — "no LiDAR mask from mid-2024, so the whole canvas is scored."** 112
interferograms carry `lidar_mask == "no_mask"`, every one of 2025 and 2026.

**Also wrong.** `"no_mask"` is not in `rasterise_lidar_gates`'s null set
(`("", "none", "null")`, `reconstruct.py:84`), so it takes the lookup branch, finds no matching
`source`, and falls back to **ALL polygons** (`reconstruct.py:87`). A `no_mask` scene is gated on
the union of LiDAR2019–2022, not on nothing.

Measured as the fraction of the scored canvas each gate passes:

| Gate | North | South | Used by |
|---|---:|---:|---|
| LiDAR2019 | 25.2% | 31.3% | 2019 scenes |
| LiDAR2020 | 47.4% | 57.3% | 2020 scenes |
| LiDAR2021 | 62.4% | 77.3% | 2021 scenes |
| LiDAR2022 | 55.0% | 58.3% | 2022 scenes |
| **UNION (fallback)** | **65.7%** | **81.2%** | every `no_mask` scene, 2024–2026 |

The union *is* the widest gate — 2.6× the 2019 gate — so newer scenes are scored on more ground.
But it is only marginally wider than the **LiDAR2021** gate (65.7 vs 62.4 North, 81.2 vs 77.3
South), and §1 records 2021 scenes at precision **0.91**. Near-identical gate, twice the
precision. **Gate width does not explain the collapse.**

### Nodata measured on all 437 interferograms — the AOI is validated

`scripts/data/measure_nodata.py`, every scene, no skips (~1 h over SMB). Nodata is
`phase == 0` — the signal `clean_patches` and `--treat_nodata_regions` already use. "inside" is
the AOI, "outside" is the canvas the AOI would delete.

| Year | Frame | Scenes | Inside | Outside | Whole | Ratio out/in |
|---|---|---:|---:|---:|---:|---:|
| 2019 | North | 32 | 0.0280 | 0.0974 | 0.0590 | 3.48 |
| 2020 | North | 30 | 0.0215 | 0.0712 | 0.0437 | 3.32 |
| 2021 | North | 30 | 0.0232 | 0.0739 | 0.0459 | 3.19 |
| 2022 | North | 31 | 0.0247 | 0.0768 | 0.0480 | 3.10 |
| 2023 | North | 32 | 0.0241 | 0.0757 | 0.0472 | 3.14 |
| 2024 | North | 32 | 0.0252 | 0.0755 | 0.0477 | 2.99 |
| 2025 | North | 33 | 0.0255 | 0.0765 | 0.0483 | 3.00 |
| 2026 | North | 9 | 0.0287 | 0.0874 | 0.0550 | 3.04 |
| 2019 | South | 27 | 0.0093 | 0.2198 | 0.1825 | 23.8 |
| 2020 | South | 23 | 0.0077 | 0.2076 | 0.1722 | 26.9 |
| 2021 | South | 28 | 0.0074 | 0.2189 | 0.1816 | 29.8 |
| 2022 | South | 33 | 0.0065 | 0.2408 | 0.1999 | 37.3 |
| 2023 | South | 31 | 0.0060 | 0.2447 | 0.2031 | 41.1 |
| 2024 | South | 32 | 0.0125 | 0.2547 | 0.2125 | 20.3 |
| 2025 | South | 25 | 0.0124 | 0.2571 | 0.2144 | 20.8 |
| 2026 | South | 9 | **0.0543** | 0.2661 | 0.2292 | 4.90 |

**The AOI keeps the well-measured ground and deletes the holes.** North is ~3× cleaner inside
the box in every year; South is **20–41× cleaner**. The box was derived purely from where the
polygons are, with no reference to data quality — that it lands on the best-measured ground is
independent confirmation, not circular.

This is the check §10 asked for, and the AOI passes it.

### Two further corrections

**Nodata does not degrade in the newer years.** North inside the AOI is 0.021–0.029 in *every*
year from 2019 to 2026; outside it is 0.071–0.097 in every year. South is equally flat. Whatever
is wrong with 2023–2026, it is not coverage.

**An earlier claim in this plan was a sampling artefact and is withdrawn.** A quick scan of 11
scenes appeared to show nodata falling from ~0.18 in 2019–2022 to ~0.045 in 2023–2025, i.e. that
the newer interferograms were *better* covered. The sample was perfectly frame-confounded — all
six old scenes were South, all five new ones North — and the full table above shows the split is
**by frame, not by year**: North ~0.05, South ~0.20, both stable across all eight years. There
is no year effect in nodata at all.

The lesson for anything else measured here: **always disaggregate by frame before reading a
year trend.** The two frames differ by 4× on this metric, so any year sample that is not
frame-balanced will manufacture a trend.

### What this does and does not settle

It settles the AOI: the box is well-measured ground, in every year, on both frames.

It does not settle §1. Nodata is unwrapping failure and masking; **decorrelation is noisy phase,
which carries non-zero values and is invisible to a zero count**. §1's precision collapse
remains unexplained by anything measured so far — the two bookkeeping hypotheses were disproved
above, and coverage is now ruled out too. Testing it needs a phase-noise statistic (local
variance or coherence); **not run, deliberately deferred.**

**One thing to watch:** 2026 South is the single outlier — 0.0543 inside the AOI against
0.006–0.013 for every other South year, and a ratio of 4.9 against 20–41. Only 9 scenes, so it
may be noise, but if 2026 is ever used, measure it again first.

### Visual inspection — rectangle confirmed, band rejected

Six scenes rendered with the box drawn on (`scripts/data/plot_aoi.py`, PNGs in
`outputs/aoi_preview/`, three per frame, seeds 0 and 1). Reviewed 2026-08-18; **the AOI is
accepted as a rectangle and the work moves on.** Two things the numbers alone did not show,
recorded because both bound how the AOI may be described:

**1. The box contains open water, and the nodata scan could not see it.** The belt follows the
shoreline, which runs diagonally — lon ≈35.39 at the south end, ≈35.45 at the north. A rectangle
wide enough to hold the belt at every latitude necessarily contains a large block of Dead Sea;
in the North frame that is roughly half the box. Water is not nodata — it returns strong, fully
random phase — so the zero-count scan rated the box "3× cleaner than outside" while half of it
carries no usable signal. This is the decorrelation blind spot §10 warned about, found in the
one place it mattered.

**Consequence: quote the AOI's benefit against the LiDAR gate, not the raw canvas.** Evaluation
already applies a LiDAR mask, so the defensible figures are **34% (North) / 80% (South)** of
scored area removed. The 45%/85% computed against the full canvas overstates it and should not
be used.

**2. South retention has a tail that the 91% aggregate hides.**

| Frame | Scenes | Min | p05 | Median | Below 80% |
|---|---:|---:|---:|---:|---:|
| North | 151 | 91% | 93% | 98% | **0** |
| South | 122 | **19%** | 81% | 91% | **6** |

Worst is `20250121_20250201`: its cluster sits at lat 31.237–31.260, lon 35.364–35.394,
straddling the southern *and* western edge at once. **Five of the six sub-80% scenes are 2025.**
This matters more than the aggregate because scene-level object metrics are the ranking metric —
a scene missing three quarters of its ground truth returns a distorted score, not merely a
smaller one.

Both edges were swept and neither is worth moving: southern edge to 31.17 buys 0.6% retention
for +43% area; western edge to 35.35 buys 1.7% for +49%. That area is what the AOI exists to
remove.

**A shoreline-following band (per-latitude longitude window) was considered and declined** —
it would track the diagonal and drop most of the water, but the rectangle is accepted as good
enough and the added complexity was judged not worth it. Revisit only if scene-level precision
on the South hold-out disappoints.

**Carried forward as open items, not blockers:**

- The six South scenes below 80% retention — decide whether to exclude them rather than score
  them on a fraction of their ground truth. Check whether 2025 digitisation is drifting
  south-west, out of the box.
- 2026 South nodata inside the box is 0.054 against 0.006–0.013 for every other South year.

### Caveats — read before adopting

- **This bounds labelled ground, not good data.** The AOI was derived from where polygons are,
  so it presumes digitisation followed data quality. It does not measure decorrelation. The
  nodata-fraction map (`|phase| <= RAW_VALIDITY_TOL`, the test `treat_nodata` already uses) is
  still worth computing to confirm the box is *also* well-imaged — but it is now a check on a
  chosen AOI, not the thing that picks it.
- **Recall is measured inside the box only.** Any sinkhole in the discarded 3–11% becomes
  invisible rather than missed. Say so wherever AOI numbers are quoted, and keep one
  full-canvas eval per model so the discarded fraction stays honest.
- **Precision will improve for a reason that is partly bookkeeping.** Deleting empty ground
  raises precision without the model changing. AOI numbers are comparable to each other, never
  to the `PREDICTIONS.md` tables.
- **Grid height is not fixed.** §2 records "(193, 89) per scene"; measured from the metadata it
  varies **191–200 rows** (North median 192, South 195) because alignment crops to a common
  origin, not a common height. `nonz_indices` therefore contains row 193, which is out of range
  for a 193-row scene. Harmless for latitude-defined windows — row→lat is fixed by the aligned
  origin — but any code indexing a fixed grid height is wrong. Fold into the §4 window work.

### Decided 2026-08-18

- **The AOI is part of the benchmark, not a variant.** It is baked into all four partitions and
  recorded in `provenance`; there is no AOI-free clean partition and no fourth generation in
  `assets/` (§9). Six training runs as per §6, scored inside the box only. The consequence
  accepted with this: **no with-box/without-box number will be produced**, so the AOI's
  contribution cannot be quoted separately — it is part of how the benchmark is defined. Any
  future claim about its effect needs a deliberate extra evaluation.
- **The same AOI applies to all four partitions**, temporal as well as geo. This is what keeps
  the two axes comparable: a geo-vs-temporal difference then reflects the axis rather than how
  much empty background each was scored over.

**Consequence for step 3:** `grid_row_window(frame, lat_min, lat_max, ...)` must be written as
`grid_window(frame, lat_min, lat_max, lon_min, lon_max, ...)` returning a row **and** column
range, and the partition JSON key is `aoi_window` with four bounds rather than `lat_window` with
two. All three consumers (`dataprep/dataset.py`, `inference/scenes.py`, `inference/outputs.py`)
must agree on it, exactly as they must for the latitude cut. **AOI: lat 31.25–31.75,
lon 35.38–35.46.**

---

## 11. Risks and open calls

- ~~**`temporal_k10` val is 6 interferograms.**~~ **Retired by §5a** — the re-derived layout
  (train <2024, val <2025, test ≥2025) gives k10 a val set of **16 interferograms / 4,735 AOI
  positives**. The val-only chain-relaxation fallback is no longer needed.
- ~~**The temporal train pool shrank to 2019–2020.**~~ **Retired by §5a** — training now spans
  2019–2023 (k5 163 intfs / 42,985 AOI positives; k10 122 / 32,590), the largest pool the archive
  allows. Temporal and geo training pools are now comparable.
- **Geo test is patch-poor**: ~5,900 positives at k5 (60% of the 9,882 hold-out). Scene-level
  object metrics are the ranking metric on both axes; patch dice on test is a sanity check only.
  §5b's `geo_crossview` does **not** relieve this — it is a diagnostic, not extra test data.
- **`geo_crossview` shares 116 of 128 scenes with train** (§5b). It is the one deliberate breach
  of one-interferogram-one-split in the whole benchmark. If it ever leaks into checkpoint
  selection or a headline number, the geo result is void.
- **Gate the stride-4 ground-truth check.** If 2019–2022 is ever re-digitised, that tree
  needs a rebuild (~5 h, 1.6 TB) before any RTh number is trusted.
- **Old numbers do not carry over.** Every table in `MODEL_RUNS.md`, `RESULTS.md` and
  `PREDICTIONS.md` is on the old partitions. Clean-data results start a new table.
- **§1's precision collapse is unexplained and now inside the benchmark.** Under §0a the test
  split is 2025–2026, the regime where §1 measured precision 0.44. If the collapse is a property
  of the imagery it will land squarely on the temporal test set. Report scene-level metrics **per
  year** so it is visible rather than averaged away, and treat a temporal-axis precision drop as
  a candidate rediscovery of §1 before reading it as a model result.

---

## 12. Order of work

| # | Step | Output | Cost |
|---|---|---|---|
| 1 | ~~Quarantine + prune `outputs/`~~ **done 08-17** | 1.34 TB freed; two trees still to move (§8) | — |
| 2 | ~~Subset dictionary~~ → audit only (§0a: no subset, `intf_coord.json` is the input) | audit output | ~1 h |
| 3 | ~~**`grid_window`** + three consumers + tests~~ **done 08-18** | `geo.grid_window`, `aoi_window` in partitions, 29 tests | — |
| 4 | ~~`make-benchmark-partitions` + tests~~ **done 08-18** | 8 partition JSONs in `assets/`, 14 tests | — |
| 5 | Submit the fourteen training runs (`submit_all.sh clean22 --submit`) | checkpoints | ~150–170 GPU-h |
| 6 | Tracks (a), (b2) and (c) | continuity numbers + the crossview gap | ~3 h per run |
| 7 | Vote mode + batching + banded streaming | code | 2 days |
| 8 | Track (b): RTh sweep + polygon analysis | paper-comparable numbers | ~4 h per run |
| 9 | ~~Preserve old partitions + kill implicit selection~~ **done 08-18** | `assets/PARTITIONS.md`, provenance stamps, no defaults (§9) | — |
| 10 | ~~AOI: bounds, all-years transfer, nodata validation~~ **done 08-18** — adopted into all four partitions (§10) | lat 31.25–31.75 / lon 35.38–35.46 | — |

Steps 3, 4 and 7 are independent of the GPU queue and can run while nothing is training.
