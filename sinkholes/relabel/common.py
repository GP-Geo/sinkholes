"""Shared pieces of the test-set relabelling workflow.

The relabelling workspace never touches the pipeline's inputs or outputs: the
original ground truth (``sub_20260701.shp``), the patch trees, the partitions
and every ``outputs/predictions`` directory are read, never written. Everything
here derives *from* them, and the two derivations that have to be exact -- the
labelling raster and the evaluation ground-truth canvas -- go through the same
functions the pipeline uses (``crop_to_start_xy``, ``geometry_mask`` on the raw
scene transform), so their equivalence is checkable bit for bit rather than
argued.

Identifiers used throughout:

``orig_uid``  ``<intf>_O<fid>`` -- one original polygon, keyed on its feature
              id in the source shapefile (the ``id`` attribute is 0/NaN on
              every 2025-26 row, so the FID is the only identifier there is).
``feat_uid``  one polygon of the working layer: equal to ``orig_uid`` for a
              polygon that came from the original, a UUID for one drawn during
              relabelling. Never edited by hand.
``change_id`` ``<intf>_<ADD|DEL|MOD>_<nnn>`` -- one detected change, stable
              across re-runs (see ``diff.py``).
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geo import FRAME_ORIGINS, X_CROP_COLS, X_CROP_OFFSET, crop_to_start_xy
from ..meta import intf_meta, parse_intf_id
from ..paths import REPO_ROOT

#: The ground truth every existing patch tree and evaluation was built from.
#: Verified, not assumed: ``sinkholes relabel eval`` re-derives each saved
#: ``_gt.npy`` from this file and refuses a directory where they differ.
ORIGINAL_GT_SHP = "sub_20260701.shp"

#: The official temporal test set: generation 4, k=5, the ``test`` split.
OFFICIAL_PARTITION = "assets/partition_temporal_k5_clean_th350x200.json"

#: Partitions whose test scenes the label-quality threshold removed. A scene in
#: one of these and not in the official list is a *candidate*: relabelling may
#: lift it over the threshold, but it never enters the official list here.
THRESHOLD_FREE_PARTITIONS = (
    "assets/partition_temporal_k5_clean.json",
    "assets/partition_temporal_k10_clean.json",
)

#: Scenes earlier evaluations singled out for poor recall across models.
USER_FLAGGED = (
    "20250329_20250409",
    "20250512_20250523",
    "20250819_20250830",
    "20250830_20250910",
    "20251002_20251013",
    "20251024_20251104",
)

#: Which processing variant prepare-patches reads (scripts/data/link_scenes.py
#: --variant int, from the data root). ``tgeo_ccw_*`` under 004/ and 013/ has
#: the same geometry but different pixel values (0..1, not wrapped phase).
SCENE_VARIANT = "int"

#: Workspace layout, relative to the workspace root.
LAYOUT = {
    "original_gpkg": "gt/gt_test_original.gpkg",
    "working_gpkg": "gt/gt_test_working.gpkg",
    "aux_gpkg": "gt/aux_context.gpkg",
    "rasters": "rasters",
    "qgis": "qgis",
    "history": "history",
    "changes": "changes",
    "exports": "exports",
    "review": "change_review",
    "eval": "eval",
    "manifest": "manifest.csv",
    "scenes_csv": "scenes.csv",
    "priority_csv": "priority.csv",
    "provenance": "provenance.json",
    "registry": "changes/change_registry.csv",
}

#: Vocabularies of the editable fields.
EDIT_STATUSES = ("unchanged", "added", "modified", "deleted")
EDIT_REASONS = (
    "missing_sinkhole",        # subsidence visible in the interferogram, not mapped
    "not_subsidence",          # mapped object is noise / atmosphere / unwrapping artefact
    "boundary_correction",     # outline too large, too small or offset
    "split",                   # one polygon covered two separate features
    "merge",                   # two polygons were one feature
    "duplicate",               # the same feature mapped twice
    "wrong_scene",             # belongs to another interferogram's dates
    "other",                   # explain in edit_notes
)
#: Quality flags: the polygon STAYS in the corrected GT and in every evaluation;
#: the flag only marks it as a known-hard or doubtful object, so reports can
#: list it and evaluations can score flagged and unflagged objects apart.
QC_FLAGS = (
    "noisy_large",             # large polygon over noisy / decorrelated phase
    "low_coherence",           # feature in a low-coherence area: shape hard to see
    "uncertain_boundary",      # real, but the outline is a guess
    "uncertain_existence",     # might not be subsidence at all
    "other",                   # explain in qc_note
)
QC_FIELDS = ("qc_flag", "qc_note")

MANIFEST_STATUSES = (
    "not_reviewed", "in_progress", "reviewed", "approved",
    "sent_for_external_review", "externally_approved", "needs_revision",
)

#: Attributes of the source shapefile, carried unchanged into every layer.
SOURCE_FIELDS = ("id", "start_date", "end_date", "platform", "flight_dir", "decor",
                 "reporter", "timestamp", "notes", "layer", "path")

#: GeoPackage creation options. GDAL >= 3.11 writes v1.4 by default, which
#: QGIS 3.40 (built on an older GDAL) opens only as "partially supported";
#: these files are edited in QGIS, so stay on the long-supported v1.2.
GPKG_OPTIONS = {"VERSION": "1.2"}

#: Projected CRS for areas and distances in reports (UTM 36N covers the Dead Sea).
METRIC_CRS = "EPSG:32636"


def ws_path(workspace, key: str) -> Path:
    return Path(workspace) / LAYOUT[key]


def date_strings(intf_id: str) -> Tuple[str, str]:
    """('YYYY-MM-DD', 'YYYY-MM-DD') exactly as prepare-patches matches them."""
    return (f"{intf_id[:4]}-{intf_id[4:6]}-{intf_id[6:8]}",
            f"{intf_id[9:13]}-{intf_id[13:15]}-{intf_id[15:17]}")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_of(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def repo_path(rel: str) -> str:
    return str(REPO_ROOT / rel)


def load_partition(rel: str) -> dict:
    with open(repo_path(rel)) as fh:
        return json.load(fh)


def find_scene_file(data_dir, intf_id: str, variant: str = SCENE_VARIANT) -> Optional[str]:
    """The raw .unw prepare-patches reads for ``intf_id`` (data root, one variant)."""
    s, e = intf_id.split("_")
    hits = sorted(p for p in os.listdir(data_dir)
                  if p.startswith(f"tgeo_{variant}_{s}T") and f"_{e}T" in p and p.endswith(".unw"))
    if len(hits) > 1:
        raise ValueError(f"{intf_id}: {len(hits)} '{variant}' scenes in {data_dir}: {hits}")
    return os.path.join(data_dir, hits[0]) if hits else None


# --------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class SceneGeometry:
    """How one interferogram's raw raster maps onto the pipeline's canvas.

    ``row_off``/``col_off`` are what ``crop_to_start_xy`` cuts off the raw
    raster to reach the frame origin. The canvas origin is the *actual*
    lon/lat of canvas pixel (0, 0) -- the raw pixel the crop lands on -- which
    differs from the nominal ``FRAME_ORIGINS`` by a fraction of a pixel
    because the nominal origin is not on the scenes' pixel grid.
    """

    intf_id: str
    frame: str
    east: float
    north: float
    dx: float
    dy: float
    nlines: int
    ncells: int
    row_off: int
    col_off: int
    canvas_x0: float
    canvas_y0: float
    nominal_x0: float
    nominal_y0: float

    @property
    def subpixel_offset(self) -> Tuple[float, float]:
        """(cols, rows) the nominal origin sits from the actual one."""
        return ((self.nominal_x0 - self.canvas_x0) / self.dx,
                (self.canvas_y0 - self.nominal_y0) / self.dy)

    def raw_transform(self):
        import rasterio

        return rasterio.transform.from_origin(self.east, self.north, self.dx, self.dy)


def scene_geometry(intf_id: str, intf_dict_path: Optional[str] = None) -> SceneGeometry:
    m = intf_meta(intf_id, intf_dict_path)
    x_star, y_star = FRAME_ORIGINS[m.frame]
    # The same rounding crop_to_start_xy applies (aligned_mask calls the real
    # function, so a disagreement here would surface as a shape mismatch).
    col_off = int(round((x_star - m.east) / m.dx))
    row_off = int(round((m.north - y_star) / m.dy))
    if not (0 <= col_off < m.ncells and 0 <= row_off < m.nlines):
        raise ValueError(f"{intf_id}: frame origin outside the raster")
    return SceneGeometry(
        intf_id=intf_id, frame=m.frame, east=m.east, north=m.north, dx=m.dx, dy=m.dy,
        nlines=m.nlines, ncells=m.ncells, row_off=row_off, col_off=col_off,
        canvas_x0=m.east + col_off * m.dx, canvas_y0=m.north - row_off * m.dy,
        nominal_x0=x_star, nominal_y0=y_star,
    )


def rasterise_like_prepare_patches(geoms: Sequence, geo: SceneGeometry) -> np.ndarray:
    """The full raw-grid mask, exactly as ``prepare_patches.main`` builds it.

    Same call (``geometry_mask(..., invert=True)``, default pixel-centre rule),
    same transform, same shape -- never a shifted window, so floating point has
    no room to move a boundary pixel.
    """
    from rasterio.features import geometry_mask

    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros((geo.nlines, geo.ncells), dtype=np.uint8)
    return geometry_mask(geoms, transform=geo.raw_transform(), invert=True,
                         out_shape=(geo.nlines, geo.ncells)).astype(np.uint8)


def aligned_mask(geoms: Sequence, geo: SceneGeometry,
                 offset_x: int = X_CROP_OFFSET, nx: int = X_CROP_COLS) -> np.ndarray:
    """The mask prepare-patches patchifies: raw mask -> frame crop -> column crop."""
    raw = rasterise_like_prepare_patches(geoms, geo)
    _, cropped, *_ = crop_to_start_xy(raw, raw, geo.east, geo.north,
                                      geo.nominal_x0, geo.nominal_y0, geo.dx, geo.dy)
    return cropped[:, offset_x: offset_x + nx]


def grid_from_canvas(shape: Tuple[int, int], patch_size: Tuple[int, int],
                     stride: int) -> Tuple[int, int]:
    """(ny, nx) of the patch grid an eval-scenes canvas of ``shape`` came from.

    Inverse of ``reconstruct.canvas_shape`` (out = n * step + patch*(1 - 1//stride)).
    """
    ph, pw = patch_size
    sy, sx = ph // stride, pw // stride
    tail_h, tail_w = ph * (1 - 1 // stride), pw * (1 - 1 // stride)
    H, W = shape
    if (H - tail_h) % sy or (W - tail_w) % sx:
        raise ValueError(f"canvas {shape} is not a stride-{stride} canvas of {patch_size} patches")
    return (H - tail_h) // sy, (W - tail_w) // sx


def gt_canvas(mask_aligned: np.ndarray, canvas: Tuple[int, int],
              patch_size: Tuple[int, int], stride: int) -> np.ndarray:
    """The ``_gt.npy`` eval-scenes would save for this aligned mask.

    Mirrors ``reconstruct_scene``: every tile (i, j) of the (ny, nx) grid adds
    its mask patch / stride**2 at (i*sy, j*sx), with no gating -- the GT canvas
    is accumulated before the LiDAR, AOI and positives-only gates. The tile
    loop is kept literal (rather than a closed form) so the float32 summation
    order, and therefore every bit, matches.
    """
    ph, pw = patch_size
    sy, sx = ph // stride, pw // stride
    ny, nx = grid_from_canvas(canvas, patch_size, stride)
    need_h, need_w = (ny - 1) * sy + ph, (nx - 1) * sx + pw
    if mask_aligned.shape[0] < need_h or mask_aligned.shape[1] < need_w:
        raise ValueError(f"aligned mask {mask_aligned.shape} smaller than the grid needs "
                         f"({need_h}, {need_w})")
    out = np.zeros(canvas, dtype=np.float32)
    m = mask_aligned.astype(np.float32)
    scale = float(stride ** 2)
    for i in range(ny):
        y0 = i * sy
        for j in range(nx):
            x0 = j * sx
            out[y0:y0 + ph, x0:x0 + pw] += m[y0:y0 + ph, x0:x0 + pw] / scale
    return out


# --------------------------------------------------------------------------- GT I/O


def read_original_gt(data_dir, intf_ids: Sequence[str]):
    """Original polygons of ``intf_ids`` with their source FID, exact-date matched."""
    import geopandas as gpd

    gdf = gpd.read_file(os.path.join(data_dir, ORIGINAL_GT_SHP), fid_as_index=True)
    sd = gdf["start_date"].dt.strftime("%Y%m%d")
    ed = gdf["end_date"].dt.strftime("%Y%m%d")
    gdf["intf_id"] = sd + "_" + ed
    gdf = gdf[gdf["intf_id"].isin(set(intf_ids))].copy()
    gdf["orig_fid"] = gdf.index.astype("int64")
    gdf["orig_uid"] = gdf["intf_id"] + "_O" + gdf["orig_fid"].astype(str)
    return gdf.reset_index(drop=True)


def read_active_gt(path, layer: Optional[str] = None):
    """Polygons of a corrected-GT file (gpkg or shp) with an ``intf_id`` column.

    Accepts both the export (``layer='gt'``) and the source shapefile, so the
    evaluation builds the original and the corrected canvases through one path.
    """
    import geopandas as gpd

    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if "intf_id" not in gdf.columns:
        sd = gdf["start_date"].astype("datetime64[ns]").dt.strftime("%Y%m%d")
        ed = gdf["end_date"].astype("datetime64[ns]").dt.strftime("%Y%m%d")
        gdf["intf_id"] = sd + "_" + ed
    return gdf


def scene_window_on_raw(geo: SceneGeometry, aoi: Sequence[float], margin_deg: float,
                        canvas_rows: Optional[int] = None,
                        nx: int = X_CROP_COLS) -> Tuple[int, int, int, int]:
    """(row0, row1, col0, col1) on the RAW raster: the canvas cut to the AOI + margin.

    Integer pixel window of the raw scene, so a raster written from it holds the
    pipeline's pixels unchanged with a transform that is a whole-pixel shift of
    the raw one.
    """
    lat_min, lat_max, lon_min, lon_max = aoi
    lat_min, lat_max = lat_min - margin_deg, lat_max + margin_deg
    lon_min, lon_max = lon_min - margin_deg, lon_max + margin_deg
    c_r0, c_c0 = geo.row_off, geo.col_off
    c_r1 = geo.nlines if canvas_rows is None else min(geo.nlines, c_r0 + canvas_rows)
    c_c1 = min(geo.ncells, c_c0 + nx)
    r0 = int(np.floor((geo.north - lat_max) / geo.dy))
    r1 = int(np.ceil((geo.north - lat_min) / geo.dy))
    c0 = int(np.floor((lon_min - geo.east) / geo.dx))
    c1 = int(np.ceil((lon_max - geo.east) / geo.dx))
    r0, r1 = max(r0, c_r0), min(r1, c_r1)
    c0, c1 = max(c0, c_c0), min(c1, c_c1)
    if r0 >= r1 or c0 >= c1:
        raise ValueError(f"{geo.intf_id}: AOI does not intersect the canvas")
    return r0, r1, c0, c1


def read_raw_window(path: str, geo: SceneGeometry, window: Tuple[int, int, int, int],
                    byte_order: str) -> np.ndarray:
    r0, r1, c0, c1 = window
    dtype = ">f4" if byte_order == "MSBFirst" else "<f4"
    mm = np.memmap(path, dtype=dtype, mode="r", shape=(geo.nlines, geo.ncells))
    return np.array(mm[r0:r1, c0:c1], dtype=np.float32)


def intf_dates(intf_id: str):
    s, e, _ = parse_intf_id(intf_id)
    return s.date(), e.date()


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def read_eval_args(eval_dir) -> Dict:
    """The ``args={...}`` eval-scenes logged for a saved evaluation directory."""
    import ast
    import glob

    logs = sorted(glob.glob(os.path.join(eval_dir, "*.log")))
    for log in logs:
        with open(log) as fh:
            for line in fh:
                if "eval-scenes:" in line and "args=" in line:
                    return ast.literal_eval(line.split("args=", 1)[1].strip())
    raise FileNotFoundError(f"no eval-scenes args line in a .log under {eval_dir}")


DATE_FIELDS = ("start_date", "end_date", "timestamp")


def write_gt(gdf, path, layer=None):
    """Write with REAL Date fields, like the source shapefile.

    pyogrio has no field-type override and writes datetime64 to a shapefile as
    a String ('2025/03/29 00:00:00'), which no exact-date match finds -- so
    fiona is used with an explicit schema. Without fiona the dates are written
    as 'YYYY-MM-DD' strings: prepare-patches still matches them, but
    scripts/data/verify_dataset.py (which uses .dt) would not.
    """
    import pandas as pd

    out = gdf.copy()
    for c in DATE_FIELDS:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    try:
        import fiona  # noqa: F401
    except ImportError:
        logging.warning("fiona not installed: writing dates as YYYY-MM-DD strings")
        for c in DATE_FIELDS:
            if c in out.columns:
                out[c] = out[c].dt.strftime("%Y-%m-%d")
        kw = {"layer": layer, "dataset_options": GPKG_OPTIONS} if layer else {}
        out.to_file(path, **kw)
        return
    props = {}
    for c in out.columns:
        if c == "geometry":
            continue
        if c in DATE_FIELDS:
            props[c] = "date"
            out[c] = [d.date() if not pd.isna(d) else None for d in out[c]]
        elif pd.api.types.is_integer_dtype(out[c]):
            props[c] = "int"
        elif pd.api.types.is_float_dtype(out[c]):
            props[c] = "float"
        elif pd.api.types.is_bool_dtype(out[c]):
            props[c] = "bool"
        else:
            props[c] = "str"
            out[c] = out[c].astype(object).where(out[c].notna(), None)
    is_gpkg = str(path).endswith(".gpkg")
    # A typed layer: QGIS only offers its polygon digitising tools on a
    # Polygon/MultiPolygon layer, not on a generic 'Geometry' one.
    types = set(out.geometry.geom_type.dropna())
    if types <= {"Polygon"}:
        gtype = "Polygon"
    else:
        from shapely.geometry import MultiPolygon

        gtype = "MultiPolygon"
        out = out.set_geometry([MultiPolygon([g]) if g is not None and g.geom_type == "Polygon" else g
                                for g in out.geometry], crs=out.crs)
    schema = {"geometry": gtype, "properties": props}
    kw = {"layer": layer, **GPKG_OPTIONS} if is_gpkg else {}
    out.to_file(path, engine="fiona", schema=schema, **kw)


def content_fingerprint(gdf) -> str:
    """SHA-256 of a layer's features (attributes + WKB), independent of the file bytes.

    The immutable original is guarded by this rather than by file permissions
    (the SMB mount ignores chmod) or by the file hash (SQLite may rewrite pages
    of a GeoPackage that is merely opened).
    """
    h = hashlib.sha256()
    cols = sorted(c for c in gdf.columns if c != "geometry")
    for row in gdf.sort_values("orig_uid").itertuples(index=False):
        d = row._asdict()
        h.update(repr([str(d[c]) for c in cols]).encode())
        g = d.get("geometry")
        h.update(g.wkb if g is not None else b"")
    return h.hexdigest()


def check_original_unchanged(workspace, orig) -> None:
    """Abort if original_gt differs from what init recorded (record it if absent)."""
    prov_path = Path(workspace) / LAYOUT["provenance"]
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    fp = content_fingerprint(orig)
    want = prov.get("original_content_sha256")
    if want is None:
        prov["original_content_sha256"] = fp
        write_json(prov_path, prov)
    elif want != fp:
        raise SystemExit(f"{LAYOUT['original_gpkg']} has been MODIFIED since init "
                         f"(content fingerprint {fp[:12]} != {want[:12]}). It must stay "
                         f"immutable: restore it from history or re-create the workspace.")


def list_eval_intfs(eval_dir) -> List[str]:
    from ..meta import INTF_ID_RE

    return sorted({m.group(0) for f in os.listdir(eval_dir)
                   if f.endswith("_pred.npy") and (m := INTF_ID_RE.search(f))})
