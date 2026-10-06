# Relabelling the temporal test set (v2)

*Written 2026-10-05. Code: `sinkholes/relabel/`, `scripts/relabel/`. Workspace:
`/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/relabel_temporal_test_v2/`
(`/Volumes/rudich/Rudich_Collaboration/...` on a Mac). This file is copied into the workspace
as `README.md`.*

The goal: correct the ground truth of the official temporal test scenes **without touching
anything that already exists**. The workflow produces two separate things:

- **Output A**, the corrected GT used for evaluation.
- **Output B**, a change-review package a person outside the pipeline can check scene by
  scene.

Existing predictions are then re-scored against the corrected labels **without re-running
any model**.

Nothing here writes to `sub_20260701.*`, `patches/`, `assets/partition_*.json`, `outputs/`,
checkpoints or any evaluation directory. Every output lives under the workspace.

---

## 1. What is in scope (verified 2026-10-05)

**Source labels.** `deadsea_sinkholes_data/sub_20260701.shp` (EPSG:4326, 75,987 polygons,
Date fields `start_date`/`end_date`). Every current patch tree and evaluation was built from
it. This is not just assumed: `relabel eval` re-derives each saved `_gt.npy` from this file
and requires a bit-identical match. The `id` attribute is 0 on every 2025–26 row, so the
shapefile **FID** is the only polygon identifier there is, and it becomes `orig_fid` /
`orig_uid`.

**Official test set.** The `test` list of
`assets/partition_temporal_k5_clean_th350x200.json` (generation 4, k=5): **20 scenes**, all
from 2025–26, each with a complete k=5 chain whose grid files exist in every patch tree.
Its `_testeval_` twin uses the same 20 as its `val` list, and that twin is what every gen-4
evaluation ran on.

| Scene | Frame | GT polygons | In k10 test |
|---|---|---|---|
| 20250223_20250306 | South | 51 | ✓ |
| 20250328_20250408 | South | 68 | ✓ |
| 20250329_20250409 | North | 73 | ✓ |
| 20250511_20250522 | South | 74 | ✓ |
| 20250602_20250613 | South | 77 | ✓ |
| 20250705_20250716 | South | 70 | ✓ |
| 20250818_20250829 | South | 72 | ✓ |
| 20250819_20250830 | North | 85 | ✓ |
| 20250830_20250910 | North | 151 | ✓ |
| 20251002_20251013 | North | 164 | ✓ |
| 20251024_20251104 | North | 127 | ✓ |
| 20251126_20251207 | North | 148 | ✓ |
| 20260211_20260222 | North | 192 | ✓ |
| 20260222_20260305 | North | 197 | ✓ |
| 20260304_20260315 | South | 82 | |
| 20260305_20260316 | North | 195 | ✓ |
| 20260326_20260406 | South | 89 | |
| 20260327_20260407 | North | 153 | ✓ |
| 20260406_20260417 | South | 51 | |
| 20260407_20260418 | North | 121 | ✓ |

**Candidates.** These seven scenes are in the workspace but **not** in the official set.
They are test scenes of the threshold-free generation-3 partitions that the label-quality
threshold (`nonz_num` must exceed 350 North / 200 South) removed. Relabelling may lift them
over it, but adding them to a test set is a separate, explicit partition decision; no tool
here does it.

| Scene | Frame | GT polygons | nonz_num / threshold |
|---|---|---|---|
| 20250121_20250201 | South | 13 | 42 / 200 |
| 20250224_20250307 | North | 62 | 340 / 350 |
| 20250512_20250523 | North | 69 | 346 / 350 |
| 20250603_20250614 | North | 70 | 343 / 350 |
| 20250706_20250717 | North | 77 | 339 / 350 |
| 20260315_20260326 | South | 13 | 70 / 200 |
| 20260316_20260327 | North | 30 | 116 / 350 |

> **20250512_20250523 is on the "poor recall" list but is not an official test scene.** Its
> low recall comes from generation-3 evaluations.

**Not in scope, but worth knowing.**

- Most other 2025–26 11-day interferograms have **no polygons at all** (`nonz_num = 0`). The
  labellers mapped roughly every other date.
- Four labelled scenes (20250122_20250202, 20251103_20251114, 20260210_20260221,
  20260221_20260304) are excluded because they lack a complete k=5 chain, not because of
  their labels.

`scenes.csv` lists every scope scene with its checks.

## 2. Which raster to label on, and why it is the right one

**Label on `rasters/<intf>_int_aoi.tif`.** It holds the exact pixels the model reads.

- prepare-patches reads the **root `tgeo_int_*.unw`** of `deadsea_sinkholes_data/` (via
  `scripts/data/link_scenes.py --variant int`).
- The `004/` (South) and `013/` (North) `tgeo_ccw_*.unw` copies have the same geometry but
  different values (0–1, not wrapped phase). **Do not label on those.**

What was verified for every scope scene:

- **Header.** The `.ers` header (origin, 2.777e-05° pixels, size, MSB byte order) equals
  `assets/intf_coord.json`.
- **Georeferencing.** GDAL reads the `.ers` with exactly the transform the pipeline uses:
  `from_origin(east, north, dx, dy)`, top-left corner, EPSG:4326, nodata 0.
- **Pixels.** A patch from `data_patches_…strpp2…` equals the raw `tgeo_int` pixels at
  `(row_off + i·100, col_off + j·50)` bit for bit, and differs from the `ccw` copy.
- **GT canvas.** The GT canvas rebuilt from `sub_20260701.shp` (raw-grid rasterisation →
  frame crop → 4500-column crop → tile accumulation) equals a saved `_gt.npy` bit for bit.

How the labelling GeoTIFFs relate to the raw scene:

- Each GeoTIFF is an **integer pixel window** of the raw scene: the pipeline canvas cut to
  the test AOI plus a 0.005° margin. Its transform is the raw transform shifted by whole
  pixels, with no resampling. It was read back and compared before being kept.
- Overviews use *nearest*, because averaging wrapped phase is meaningless.
- `rasters/<intf>_int_full.vrt` opens the whole raw scene (slow over SMB) if you need to look
  outside the AOI.

**A known sub-pixel quirk, unchanged by this work.** The nominal frame origins
(`geo.FRAME_ORIGINS`) are not on the scenes' pixel grid.

- North scenes sit −0.25 px in x and +0.12 px in y from the nominal origin; South scenes sit
  +0.25 px and −0.35 px.
- `crop_to_start_xy` rounds to the nearest pixel, consistently for every scene of a frame, so
  temporal stacks are aligned exactly.
- But the pipeline georeferences its *exported prediction polygons* (and the LiDAR gate)
  from the nominal origin, which shifts them by ≤ 0.35 px (≈ 1 m) relative to the data.
- The GT is unaffected: it is rasterised on the raw grid. The relabelling tools use the
  actual canvas origin.

## 3. Workspace layout

```
relabel_temporal_test_v2/
  README.md               this file
  provenance.json         source paths + SHA-256, partition, git commit, raster windows
  scenes.csv              per scene: role, frame, counts, raw file, chain, offsets, raster
  priority.csv            review order + diagnostics from existing evaluations
  manifest.csv            YOUR progress tracker (status, reviewer, dates, notes)
  gt/
    gt_test_original.gpkg   IMMUTABLE: original_gt, scenes (guarded by a content fingerprint)
    gt_test_working.gpkg    EDIT THIS: working_gt
    aux_context.gpkg        aoi_window, predictable_area, lidar2022_gate, scene_rasters_extent
  rasters/                <intf>_int_aoi.tif (label on these), <intf>_int_full.vrt
  predictions/<model>/    <intf>_conf.tif: a model's saved confidence, same grid as rasters/
                          (second pass only; OFF in the project), model.json
  qgis/                   relabel_test_v2.qgz, goto_scene.py
  history/                automatic snapshots of the working gpkg (every diff/export/review)
  changes/                change_registry.csv, gt_changes_v2.gpkg, change_list.csv,
                          change_summary.csv, validation_report.csv   (all generated)
  exports/                Output A: gt_test_corrected_v2.gpkg, shp/, history/
  change_review/          Output B: one dated package (+ .zip) per review round
  eval/                   corrected-label re-scoring, per model, + gt_canvases cache
```

### Schema

`original_gt` (immutable) and `working_gt` both carry:

- every source attribute, unchanged: `id, start_date, end_date, platform, flight_dir, decor,
  reporter, timestamp, notes, layer, path`
- `orig_fid`: the FID in `sub_20260701.shp`
- `orig_uid`: `<intf>_O<fid>`
- `intf_id`

`working_gt` adds:

| field | meaning | how it is set |
|---|---|---|
| `feat_uid` | identity of this polygon | `= orig_uid` for originals; `uuid()` for new ones (form default) |
| `edit_status` | `unchanged` / `added` / `modified` / `deleted` | automatic on edit (see below); you set `deleted` |
| `edit_reason` | `missing_sinkhole`, `not_subsidence`, `boundary_correction`, `split`, `merge`, `duplicate`, `wrong_scene`, `other` | drop-down; the form warns if empty on a change |
| `edit_notes` | free text | you |
| `edited_by`, `edit_timestamp` | who / when | automatic on every edit |

**Immutability of the original.** The SMB mount ignores `chmod`, and SQLite may rewrite pages
of a GeoPackage that is merely opened, so neither file permissions nor a file hash can guard
it. `init` records a SHA-256 over the features of `original_gt` (attributes + WKB) in
`provenance.json` as `original_content_sha256`. `diff`, `validate`, `export` and `review`
re-compute it and **abort if it changed**. QGIS opens the layer read-only. On WEXAC,
`chmod a-w gt/gt_test_original.gpkg` adds a real filesystem lock.

**File details.**

- GeoPackages are written as **v1.2**. GDAL ≥ 3.11 defaults to v1.4, which QGIS 3.40 opens
  only as "partially supported".
- Date fields are real **DATE** fields. A DATETIME field let QGIS store a new polygon's date
  as the previous day in UTC.
- Layers are typed **Polygon**, so QGIS offers its polygon tools on them.

**Deleted polygons stay in `working_gt`** with `edit_status = deleted`. That keeps the
deletion and its reason in the audit trail; the export leaves them out. Audit data never
reaches the evaluation GT: `export` writes only live polygons, in the source schema plus
four bookkeeping columns.

## 4. Labelling in QGIS

**Open.** Open `qgis/relabel_test_v2.qgz` in QGIS ≥ 3.34. Paths are relative, so it works
from the WEXAC path or the Mac mount.

**Switch scenes.** Open the Python console (Plugins → Python Console) and run:

```python
exec(open(QgsProject.instance().homePath() + '/goto_scene.py').read())
goto('20250329_20250409')   # filter labels, show its interferogram, zoom
nxt()                       # next scene in priority.csv order
where()                     # current scene
```

`goto()` sets `@relabel_intf`. New polygons inherit their `intf_id` and dates from it. It
refuses to switch scenes while there are unsaved edits. Without the console, set the layer
filter `"intf_id" = '…'` on both label layers, set the project variable `relabel_intf`
(Project → Properties → Variables), and tick the scene's raster.

**Layers.** The QGIS project stacks them in this order (top first):

- **working_gt** (editable), styled by `edit_status`:
  - yellow: unchanged
  - green: added
  - orange: modified
  - dashed red: deleted
- **original_gt** (read-only): dashed cyan outline, drawn alongside so original and corrected
  can be compared; toggle it to compare.
- **Context:**
  - the test AOI (yellow)
  - `predictable_area` (dashed lime): stride-4 tiles inside the AOI and the LiDAR2022 gate.
    GT outside it is **always a miss** for every model.
  - the LiDAR2022 footprint (grey)

**Conventions.**

- **Add** a missing sinkhole/subsidence. Draw it in `working_gt` (Add Polygon) and choose
  `edit_reason`. Status becomes `added` by itself.
- **Modify** an outline. Use the Vertex tool on the existing polygon; **don't redraw it**,
  because redrawing loses `orig_uid`. Status becomes `modified` by itself; choose a reason.
- **Delete** a false label. **Don't press Delete.** Set `edit_status = deleted` and a reason.
  A hard delete is still detected, but it carries no reason.
- **Split.** Use Split Features; the pieces keep `orig_uid` and each gets a fresh
  `feat_uid` (the project sets the field's split policy; the first session's project
  copied it). The diff keeps the largest
  piece as the modification and reports the others as additions "split from …". Reason:
  `split`.
- **Don't empty a polygon.** Delete Part on a polygon's last part leaves a row with no
  geometry. To remove a polygon, set `edit_status = deleted`. (This happened in the first
  session; `diff` now flags it.)
- **Merge.** Reshape one polygon to cover both, and mark the other `deleted` with reason
  `merge`.
- **Wrong scene.** Don't change `intf_id` by hand unless the polygon really belongs to
  another *workspace* scene. The diff reports it as a move, flagged for review.
- **Pixel rule.** A polygon burns the pixels whose **centre** lies inside it. Polygons under
  ~4 pixels (≈ 30 m²) barely exist in the mask, and `validate` warns about them.
- **Save often.** Each `diff` / `export` / `review` snapshots the working gpkg into
  `history/`.

**Flag hard or doubtful polygons, don't delete them.** Some polygons are real but hard for
the model, for example a large polygon over noisy, decorrelated phase. To keep such a polygon
in the ground truth and still have it listed, give it a **quality flag**. Select it, then in
the console run:

```python
flag('noisy_large', 'big polygon over decorrelated phase')   # note optional
unflag()        # remove the flag from the selected polygon(s)
flagged()       # list this scene's flagged polygons
```

You can also use the `qc_flag` drop-down in the attribute form. The flags are:

- `noisy_large`
- `low_coherence`
- `uncertain_boundary`
- `uncertain_existence`
- `other` (explain in `qc_note`)

A flag is **not** an edit: the polygon stays in the corrected GT and in every evaluation, and
`edit_status` stays as it was (the project only sets `modified` when the geometry differs from
the original). Flagged polygons get a purple dotted outline on top of their status colour.
Flags travel everywhere:

- `diff`: `changes/flagged_list.csv`, a `flagged` layer in `gt_changes_v2.gpkg`, and
  `flagged`/`flags` columns in `change_summary.csv`. Flag ids look like
  `<scene>_FLAG_<nnn>` and are stable.
- `review`: purple in every figure, listed separately as "kept in the corrected GT", with a
  `flag` column on the cover.
- `export`: `qc_flag`/`qc_note` columns in the corrected GT.
- `eval run --objects`: every corrected GT object overlapping a flagged polygon is marked
  `qc_flagged`. The per-scene summary splits recall into `recall_new_flagged` and
  `recall_new_unflagged`, plus `flagged_gt_area_share`. That measures how much of the miss
  rate comes from the hard polygons, without removing them.

The fields were added to an existing working file with `sinkholes relabel migrate` (QGIS
closed; additive; backup in `history/`).

`edit_status` is a declaration. **`relabel diff` decides what changed from the geometry**
and reports any declaration it disagrees with.

**Tracking.** Track progress in `manifest.csv` with these statuses:

- `not_reviewed`
- `in_progress`
- `reviewed`
- `approved`
- `sent_for_external_review`
- `externally_approved`
- `needs_revision`

Also fill in `reviewed_by`, `review_date`, `external_reviewer`, `external_review_date` and
`notes`. The count columns are filled in by `diff`.

**Order.** Follow `priority.csv`, which ranks scenes using the 8 generation-4 k=5 models at
RTh 0.25:

- **Tier 1:** your six flagged scenes, plus any scene in a model's bottom-5 recall for at
  least 75% of models. The five flagged official scenes are bottom-5 in 6–8 of 8 models.
- **Tier 2:** recall below the official mean.

## 4b. Two passes: blind, then against a model

**Pass 1 (blind).** Label without looking at any prediction. The project opens with every
`PREDICTIONS -- …` group **unchecked**. `goto()`/`nxt()` move those layers to the new scene
but never switch a group on, so a prediction cannot appear by accident. Finish and save each
scene, and mark it `reviewed` in `manifest.csv`, before pass 2.

**Pass 2 (disagreement review).** For a scene you have finished:

```python
predictions_on()          # tick the group: continuous confidence of the reference model
pred_threshold(0.25)      # + binary layer, confidence > 0.25 (the evaluation's own rule: strict >)
pred_threshold(0.5)       # any value: 0.125 0.25 0.5 0.7 0.9 ...
pred_bands()              # instead: nested classes > 0.125 / 0.25 / 0.5 / 0.7 / 0.9
pred_opacity(0.4)         # more interferogram, less prediction (default 0.6; thresholded 0.55)
predictions_off()         # (= blind()) hide everything again before the next blind scene
```

Look for strong confidence with no label (possible missing GT) and for labels with no
confidence (a miss, or a doubtful label). Every pass-2 edit is recorded like any other, but
**put "pass 2" in `edit_notes`**. That keeps blind and model-informed corrections separable
when the effect of relabelling is measured: changes seen in pass 2 are not independent of the
model.

**By hand.** Tick or untick the group in the Layers panel. To change the threshold by hand,
open the "> t" layer, then Properties → Symbology: the first class value is the threshold,
and pixels **above** it are coloured. Opacity is under Properties → Transparency.

**What the confidence is.** The reference model was evaluated with the RTh protocol
(stride 4, `--recon_average vote`), so a pixel's value is the **fraction of the 16 overlapping
tiles that called it positive**: 0, 1/16, …, 1. It is not a probability. The project's
operating points are RTh 0.125–0.25. The map is zero where no tile was predicted: outside
the AOI tiles and the LiDAR2022 gate (the dashed `predictable_area`). Colours are greens,
the one hue the phase colour scale never uses.

**Reference model:** `th350_tattn_ctx50_neg10`, from
`outputs/predictions/th350_tattn_ctx50_neg10_2026-09-10_01h16_lsf_816217/best/scenes_temporal_th350_rth_09_14_12h21`.

- It is the best generation-4 temporal k5 model in `docs/RESULTS.md` §3 (object F1 0.7905,
  P 0.869, R 0.725, strict tolerance).
- It ranks first on the area-weighted F1 recomputed from its saved metrics at RTh 0.25 and
  0.5.
- It has predictions for all 20 official scenes.
- The nominally equal k10 leader, `posw8_tattn_k10_plain_30e` (0.7907), covers only 17. It
  has nothing for 20260304, 20260326 or 20260406.

**Adding another model.** Pass any other saved eval directory; it becomes its own group, also
off:

```bash
sinkholes relabel predictions --workspace $WS --data_dir $DATA --eval_dir <eval dir> [--name tag]
scripts/relabel/qgis_python.sh scripts/relabel/build_qgis_project.py --workspace $WS
```

**How the rasters were made, and checked.**

- No inference is run; the evaluation directory is only read.
- Each `<intf>_pred.npy` canvas pixel (r, c) is raw pixel (row_off + r, col_off + c). The
  labelling raster records its raw window. The prediction GeoTIFF is that integer slice,
  written with **the labelling raster's own transform, CRS and size**, and is read back and
  compared before it is kept. Window rows the canvas does not reach (a few at the bottom) are
  nodata (−1).
- **Alignment is verified against the evaluation's own output.** The saved `_gt.npy`, cut the
  same way, must be positive on exactly the pixels where the original polygons rasterise on
  the labelling raster's grid. That holds on all 20 scenes (55k–298k labelled pixels each).
  Shifting the crop by one row or one column breaks the check, so a misalignment cannot pass.

**Rebuild with QGIS closed, or reopen without saving.** A QGIS session that still has the
old project open will write it back over a rebuilt one if you choose *Save* on exit. That
happened once, on 2026-10-05: the prediction layers vanished. Since then, `goto()` no longer
marks the project modified, and loading `goto_scene.py` warns when the open project is older
than `predictions/`. To pick up a rebuild while QGIS is open, use Project → Revert.

Rebuilding the QGIS project (`build_qgis_project.py`) first copies the previous `.qgz` into
`history/`. Layer styles you change by hand are reset by a rebuild; your labels are not, since
they live in the GeoPackage.

## 5. Commands

Run from the repo root, with the project environment:

```bash
WS=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/relabel_temporal_test_v2
DATA=/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data

sinkholes relabel validate --workspace $WS        # anytime; exit 1 on errors
sinkholes relabel diff --workspace $WS            # changes/, stable change ids, manifest counts
sinkholes relabel export --workspace $WS          # Output A (validates first; errors block)
sinkholes relabel review --workspace $WS          # Output B -> change_review/review_<date>/ + .zip
sinkholes relabel eval run --workspace $WS --data_dir $DATA --objects
sinkholes relabel eval compare --workspace $WS
```

Workspace setup, already done:

```bash
sinkholes relabel init --workspace $WS --data_dir $DATA
scripts/relabel/qgis_python.sh scripts/relabel/build_qgis_project.py --workspace $WS
```

### Validation (`validate`)

ERROR blocks export. WARNING and INFO are reported. Problems in untouched originals are
WARNING ("inherited"): one 2025 original is already an invalid geometry.

| level | check | when |
|---|---|---|
| ERROR | wrong CRS | layer not EPSG:4326 |
| ERROR | null / empty geometry | |
| ERROR | non-polygon geometry | |
| ERROR | `intf_id` | not a workspace scene |
| ERROR | `start_date`/`end_date` | inconsistent with `intf_id` (prepare-patches matches on exact dates) |
| ERROR | invalid geometry | on an edited polygon |
| ERROR | outside the raster | polygon outside the interferogram |
| ERROR | duplicate `feat_uid` | |
| ERROR | duplicate polygon | an edited polygon duplicates another |
| WARNING | edit bookkeeping | unmatched/ambiguous change, missing reason |
| WARNING | burn size | edited polygon burning < 4 pixels |
| WARNING | size | area > 0.5 km² |
| WARNING | overlap | near-duplicate overlap (IoU > 0.5) |
| WARNING | raster edge | polygon crossing the raster edge |
| INFO | position | outside the AOI (never scored); outside `predictable_area` (always missed) |
| INFO | bookkeeping | hard delete; declared status ≠ detected |

### Change detection and ids (`diff`)

Matching runs per interferogram in three passes. The first match wins.

1. **Lineage.** A working polygon with an original's `orig_uid` continues it:
   - identical geometry (Hausdorff ≤ 1e-8°) → unchanged;
   - otherwise → modified.

   Several polygons carrying one `orig_uid` are a split.
2. **Geometry, for orphans.** An original no longer carried by any polygon, and a new polygon
   with no lineage, are the same object when each is the other's **only** overlap and
   IoU ≥ 0.5. This is the "redrawn" case: a modification found by geometry, not FID. Any
   other overlap (one new polygon over two originals, two over one, a weak IoU) is **not
   matched**. It becomes additions plus deletions with `needs_review` set and the candidates
   named, so the change is flagged instead of silently mismatched.
3. **The rest.**
   - remaining originals → deleted (also any `edit_status = deleted`)
   - remaining new polygons → added

**Change ids.** Each change gets an id `<intf>_<ADD|DEL|MOD>_<nnn>`, numbered per scene.

- Ids are held in `changes/change_registry.csv`, keyed on (type, polygon): the original for
  DEL/MOD, the new polygon for ADD.
- Re-running the diff after more editing never renumbers an existing change.
- A change that disappears (you reverted it) is marked `withdrawn` and its number is not
  reused.

The same id appears in the PDF, the PNGs, `change_list.csv`, every layer of
`gt_changes_v2.gpkg` and the export's `change_id` column.

**`changes/gt_changes_v2.gpkg`** is for audit and communication, *not* GT. Its layers:

- `added`
- `deleted`
- `modified_original` (old outline)
- `modified_corrected` (new outline)
- `change_markers` (one labelled point per change)

Each feature carries:

- `change_id`, `intf_id`, `change_type`
- `orig_uid`, `feat_uid`, `start_date`, `end_date`, `frame`
- `edit_reason`, `edit_notes`, `edited_by`, `edit_timestamp`
- `declared_status`, `match_method`, `needs_review`, `review_note`, `related`
- `iou`, `area_before_m2`, `area_after_m2`, `centroid_shift_m`, `lon`, `lat`

### Output A: `export`

Writes `exports/gt_test_corrected_v2.gpkg` (layer `gt`) and
`exports/shp/sub_test_corrected_v2.shp`. Only live polygons are written, in the source
schema:

- dates re-derived from `intf_id`
- `reporter`/`timestamp` set to the editor and the edit date on changed polygons
- four extra columns: `intf_id`, `gt_uid`, `orig_uid`, `edit_stat`, plus `change_id` in the
  gpkg

Before publishing, each file is re-read and matched per scene *the way prepare-patches
matches*. `--full_merge --data_dir $DATA` also writes
`exports/full/sub_20260701_testcorrected_v2.shp`: the whole archive with the scope scenes
replaced, a drop-in for a future patch regeneration. Every export is copied into
`exports/history/` with a JSON of counts and SHA-256s.

### Output B: `review`

`change_review/review_<date>/` (and a `.zip` beside it) contains:

- `change_report.pdf`:
  - a cover page with a legend and the per-scene summary
  - then, for **each interferogram with changes**: an overview page with the whole AOI strip
    of the interferogram, every change circled, numbered zoom boxes, the counts
    (original / corrected / added / deleted / modified) and the change table with reasons,
    notes and coordinates
  - one page per zoom box, with the **original labels on the left and the corrected labels
    on the right** over the interferogram:
    - added: green
    - deleted: red, dashed and hatched
    - modified: old outline dashed magenta, new outline solid orange
    - unchanged: thin cyan
- `scenes/*.png`: the same figures.
- `gt_changes_v2.gpkg`, `change_list.csv`, `change_summary.csv`.
- `README.txt` (plain language).
- `reviewer_response.csv`: one row per change id for the reviewer's verdict.

To embed QGIS styles in the package's GeoPackage, so it opens styled:
`scripts/relabel/qgis_python.sh scripts/relabel/build_qgis_project.py --style_changes
<package>/gt_changes_v2.gpkg`.

## 6. Evaluating corrected labels

**Inference never reads the GT.** In `reconstruct_scene`, the GT canvas is accumulated
beside the prediction and the tiles are gated by LiDAR and the AOI only. The one exception
is `--positives_only`, which `relabel eval` refuses. So a saved `_pred.npy` is the model's
answer whatever the labels say, and **nothing is re-run**.

**Chosen design (Option B with a scoring hook).**

- `eval-outputs` gained `--gt_dir` / `--gt_map`. Defaults are unchanged; with either flag set
  it reads the predictions from `--path` read-only, requires `--out_json`, refuses
  `--save_figures`, and checks the GT canvas shape against the prediction's.
- `relabel eval run` then, per saved evaluation directory:
  1. reads its eval-scenes arguments from the log;
  2. rebuilds every scene's GT canvas from `sub_20260701.shp` and requires **bit-identity**
     with the saved `_gt.npy`. This is the proof that scene alignment, frame origin, pixel
     size, crop, canvas size and stride are reproduced, so the corrected canvas differs from
     the original in the labels only. A directory that fails is refused.
  3. builds the corrected canvas from `exports/gt_test_corrected_v2.gpkg` (cached per canvas
     shape under `eval/gt_canvases/`);
  4. scores the same `_pred.npy` twice with the unchanged scoring code and the directory's
     own stride, AOI and protocol (`--rth` for vote maps): `olm_original.json` and
     `olm_corrected_v2.json`;
  5. checks the re-scored original against the archived `olm_results_*.json`
     (`rescore_vs_archived` in `meta.json`).

**Per-object diagnostics.** `--objects` writes `objects_th0.25.gpkg` and a summary CSV.
Every predicted object is classified TP/FP under both labelings: `FP->TP` is an *apparent
false positive that was missing GT*. Every GT object is marked detected/missed under each
labeling. These use the exact `object_level_evaluate` rules (a test pins the equivalence),
georeferenced from the actual canvas origin.

**Comparison.** `relabel eval compare` writes `eval/v2/comparison/`:

- `aggregate_th0.25.csv`: per model and tolerance, weighted/mean recall, precision and F1,
  old → new
- `per_scene_th0.25.csv`, and `per_scene_mean_over_models_th0.25.csv`, which shows which
  scenes changed most and whether the flagged ones improved
- `model_ranking_th0.25.csv` + `comparison_th0.25.md`: rank old → new and Kendall τ,
  **within one scene list only** (k10 models are scored on 17 scenes, k5 on 20)

Default scope: the 12 generation-4 vote-protocol evaluations (8 on the k=5 list, 4 on the
k=10 list). Pass `--eval_dirs` for others; each must pass the bit-identity check.

Nothing is written to any `outputs/predictions/...` directory.

## 7. Found while building this

- **Wrong-scene original.** `20250706_20250717_O69042`, a polygon of the candidate scene
  20250706_20250717 (North frame, ascending, reporter Dafnash), lies at 30.99°N, entirely
  outside that interferogram's raster (31.25–31.81°N). It is probably a South-area polygon
  filed under the North pair's dates. `validate` reports it as an inherited `outside_raster`.
- **Pre-existing problems.** One original 2025–26 polygon outside the scope is an invalid
  geometry, and five scope originals burn fewer than 4 pixels. `validate` lists them.
- **The export guard works.** Writing dates through pyogrio would have produced String fields
  (`2025/03/29 00:00:00`) that prepare-patches' exact match never finds. The export's
  read-back check caught it, and the writer now uses fiona with a DATE schema.

## 8. Open questions to settle before labelling

1. **Labelling standard.** What counts as "subsidence" in an 11-day pair: one fringe? a
   minimum size? an outline at the fringe edge or the deformation edge? Decide with the
   Geological Survey collaborator **before** starting, and write it in this README. Today's
   labels come from two reporters with visibly different styles.
2. **Inside the AOI only?** Only polygons inside the test AOI are scored. Polygons inside
   the AOI but outside `predictable_area` (LiDAR2022 gate) are always misses. Fix those
   anyway, or treat the gate as part of the protocol?
3. **Independence.** Relabelling while looking at model predictions biases the labels toward
   the models. Predictions are hidden by default and shown only on request (§4b). Agree how
   pass-2 edits will be reported. They can be kept apart through the "pass 2" note, or
   evaluated as a separate label version (export once after pass 1 with `--tag v2_blind`,
   then again after pass 2).
4. **Candidates.** Relabelling the seven threshold-excluded scenes may make them eligible.
   Adding them changes the test set and makes old and new numbers incomparable. Should that
   be a new partition generation (5)?
5. **Training labels.** The same labelling problems presumably exist in the training years.
   Corrected *test* labels measure error better; they do not fix what the models learned.
6. **One labeller or two?** For an inter-annotator check, a second person could relabel a
   few scenes independently in a copy of the working gpkg. The diff tool compares any two.
