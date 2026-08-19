# Correlation maps — tested, dead end (2026-08-19)

The coherence rasters at `deadsea_sinkholes_data/{004,013}/` (`tgeo_ccw_*.unw`, 004 = South,
013 = North, 677 GB). **Tested as a model input and rejected. Do not re-audit the data — the
audit was clean; it is the signal that is absent.**

**The data is fine.** 436/437 interferograms covered (only `20201217_20201228` lacks a `.unw`),
every file byte-complete, values in [0, 1]. The rasters are a later reprocessing on a *different
origin and size* than the phase scenes, but after the usual `crop_to_start_xy` alignment they
land on the phase grid to **under 0.02 px** — verified on all 437, plus a ±3 px shift search
that peaked at exactly (0,0) on every scene tried.

**Two hypotheses, both null**, measured on 20,581 predicted polygons from
`geo_k5_convlstm_ring3` (18 test scenes), labelled TP/FP by the project's own object rule:

| hypothesis | statistic | AUC (0.5 = nothing) |
|---|---|---|
| errors sit in low-coherence ground *now* | current-frame correlation | **0.534** |
| errors follow decorrelated *history* (temporal gating) | worst of 5 predecessors | **0.512** |
| — | polygon **area** | **0.828** |

Correlation also does not mark sinkholes at all: inside GT polygons 0.578 vs 0.588 outside.
A post-hoc coherence filter peaks at precision 0.073 while destroying 11% of true detections.

**The one result worth keeping.** Area is what separates true from false detections — FP median
**51 px** vs TP median **392 px**. A minimum-area floor of 50 px removes **half** the false
positives for **2.5%** of recall; 100 px removes two thirds for 5%. There is no area filter in
the pipeline today: `polygons.mask_array_to_polygons` keeps every connected component at any
size. Adding a `min_area_px` argument there is the change. **Re-measure the floor on the
generation-3 AOI benchmark before hard-coding a value** — the ordering (area ≫ coherence) should
hold, the cutoff will move.

**If anyone reopens this**, the untested version is: evaluate twice, once normally and once with
`eval-scenes --replicate_input`, then split scenes by predecessor coherence. That distinguishes
"bad history does no harm" from "the ConvLSTM already discounts it by reading the phase" — the
one thing the null results above cannot separate. Needs no new code, costs two eval runs.

*Measurement scripts and CSVs deleted 2026-08-19 by request; regenerating them is ~40 min of
polygon/raster work if ever needed.*
