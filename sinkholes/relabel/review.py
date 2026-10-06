"""``sinkholes relabel review`` -- Output B, the change-review package.

A self-contained folder (and zip) for a reviewer who does not use the
pipeline: a PDF organised by interferogram, one PNG per overview and per
zoomed change area, the changes-only GeoPackage, and CSV tables -- all
carrying the same change ids (``20250329_20250409_ADD_001``) as
``changes/change_registry.csv``.

Every figure draws the interferogram itself underneath (the labelling raster:
the pipeline's own tgeo_int pixels), the original labels and the corrected
labels, with additions, deletions and both the old and new outline of a
modification styled apart.
"""

import csv
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from .common import ws_path

STYLE = {
    "unchanged": dict(edgecolor="#00e5ff", linewidth=0.7, facecolor="none"),
    "ADD": dict(edgecolor="#39ff14", linewidth=2.0, facecolor="#39ff1433"),
    "DEL": dict(edgecolor="#ff1744", linewidth=2.0, facecolor="#ff174422", linestyle="--", hatch="xx"),
    "MOD_old": dict(edgecolor="#ff00ff", linewidth=1.8, facecolor="none", linestyle="--"),
    "MOD_new": dict(edgecolor="#ff9100", linewidth=2.0, facecolor="none"),
    "FLAG": dict(edgecolor="#d500f9", linewidth=2.2, facecolor="#d500f91a", linestyle=":", hatch=".."),
}
COLOR = {"ADD": "#39ff14", "DEL": "#ff1744", "MOD": "#ff9100", "FLAG": "#d500f9"}
CMAP = "twilight"            # cyclic, and free of the overlay colours
ZOOM_PAD_PX = 80             # context around a change in a zoom panel
ZOOM_MIN_PX = 260            # smallest zoom window side (~720 m)


def add_arguments(p):
    p.add_argument("--workspace", required=True)
    p.add_argument("--tag", default=None, help="package name (default review_<date>)")
    p.add_argument("--scenes", nargs="*", default=None, help="only these interferograms")
    p.add_argument("--reviewer_note", default="", help="free text for the cover page")


# --------------------------------------------------------------------------- drawing helpers


def _patches(ax, geom, style, zorder=3):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    if geom is None or geom.is_empty:
        return
    polys = getattr(geom, "geoms", [geom])
    for g in polys:
        if g.geom_type != "Polygon":
            continue
        verts, codes = [], []
        for ring in [g.exterior, *g.interiors]:
            xy = np.asarray(ring.coords)
            verts.extend(xy)
            codes.extend([MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY])
        ax.add_patch(PathPatch(MPath(verts, codes), zorder=zorder, **style))


def _read_raster(tif, bounds=None, max_px=1600):
    """(array, extent) of the labelling raster, optionally windowed, decimated to max_px."""
    import rasterio
    import rasterio.errors
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    with rasterio.open(tif) as src:
        if bounds is None:
            win = rasterio.windows.Window(0, 0, src.width, src.height)
        else:
            win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            try:
                win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            except rasterio.errors.WindowError:
                # the change lies outside the labelling raster (outside the AOI crop)
                return None, (bounds[0], bounds[2], bounds[1], bounds[3])
        scale = max(1.0, max(win.height, win.width) / max_px)
        shape = (max(1, int(win.height / scale)), max(1, int(win.width / scale)))
        arr = src.read(1, window=win, out_shape=shape, resampling=Resampling.nearest)
        b = rasterio.windows.bounds(win, src.transform)
    arr = np.ma.masked_equal(arr, 0.0)
    return arr, (b[0], b[2], b[1], b[3])


def _geo_axes(ax, extent, lat):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.radians(lat)))
    ax.tick_params(labelsize=6)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("lon", fontsize=6)
    ax.set_ylabel("lat", fontsize=6)


def _scalebar(ax, extent, lat):
    span_m = (extent[1] - extent[0]) * 111_320 * np.cos(np.radians(lat))
    nice = [50, 100, 200, 250, 500, 1000, 2000, 5000]
    L = max([n for n in nice if n <= span_m / 3] or [nice[0]])
    dlon = L / (111_320 * np.cos(np.radians(lat)))
    x0 = extent[0] + 0.05 * (extent[1] - extent[0])
    y0 = extent[2] + 0.04 * (extent[3] - extent[2])
    ax.plot([x0, x0 + dlon], [y0, y0], color="white", lw=3, zorder=10)
    ax.plot([x0, x0 + dlon], [y0, y0], color="black", lw=1, zorder=11)
    ax.text(x0 + dlon / 2, y0, f"{L} m", color="white", fontsize=6, ha="center", va="bottom",
            zorder=12, bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))


def _label(ax, c, short=True, fontsize=6):
    txt = c["change_id"].split("_", 2)[-1] if short else c["change_id"]
    if c["change_type"] == "FLAG" and short:
        txt = f"{txt} {c.get('qc_flag', '')}"
    ax.annotate(txt, (c["lon"], c["lat"]), xytext=(6, 6), textcoords="offset points",
                fontsize=fontsize, color="black", zorder=20,
                bbox=dict(boxstyle="round,pad=0.2", fc=COLOR[c["change_type"]], ec="black", lw=0.5),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.5))


def _legend(ax):
    from matplotlib.patches import Patch

    items = [Patch(**{k: v for k, v in STYLE["unchanged"].items() if k != "linewidth"}, label="unchanged label"),
             Patch(edgecolor=COLOR["ADD"], facecolor="#39ff1433", lw=2, label="ADD: new polygon"),
             Patch(edgecolor=COLOR["DEL"], facecolor="#ff174422", lw=2, ls="--", hatch="xx",
                   label="DEL: removed polygon"),
             Patch(edgecolor="#ff00ff", facecolor="none", lw=2, ls="--", label="MOD: old outline"),
             Patch(edgecolor=COLOR["MOD"], facecolor="none", lw=2, label="MOD: new outline"),
             Patch(edgecolor=COLOR["FLAG"], facecolor="#d500f91a", lw=2, ls=":", hatch="..",
                   label="FLAG: hard/doubtful, KEPT in GT")]
    ax.legend(handles=items, loc="upper left", fontsize=6, framealpha=0.9)


# --------------------------------------------------------------------------- clustering


def clusters_of(changes, dx):
    """Groups of nearby changes, each with a lon/lat window for a zoom panel."""
    from shapely.geometry import box

    boxes = []
    for c in changes:
        geoms = [g for g in (c["old_geom"], c["new_geom"]) if g is not None]
        minx = min(g.bounds[0] for g in geoms)
        miny = min(g.bounds[1] for g in geoms)
        maxx = max(g.bounds[2] for g in geoms)
        maxy = max(g.bounds[3] for g in geoms)
        boxes.append(([c], box(minx, miny, maxx, maxy).buffer(ZOOM_PAD_PX * dx, join_style=2)))
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][1].intersects(boxes[j][1]):
                    cs = boxes[i][0] + boxes[j][0]
                    b = boxes[i][1].union(boxes[j][1]).envelope
                    boxes[j] = (cs, b)
                    boxes.pop(i)
                    merged = True
                    break
            if merged:
                break
    out = []
    for cs, b in sorted(boxes, key=lambda t: -t[1].centroid.y):
        minx, miny, maxx, maxy = b.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        half = max(maxx - minx, maxy - miny, ZOOM_MIN_PX * dx) / 2
        out.append((sorted(cs, key=lambda c: c["change_id"]), (cx - half, cy - half, cx + half, cy + half)))
    return out


# --------------------------------------------------------------------------- figures


def overview_figure(fig_or_none, scene, tif, orig_s, corr_s, cs, clusters, stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    lat = float(scene["geometry"].centroid.y)
    arr, ext = _read_raster(tif, max_px=2200)
    ax = fig.add_axes([0.05, 0.05, 0.36, 0.86])
    ax.imshow(arr, extent=ext, cmap=CMAP, vmin=-np.pi, vmax=np.pi, interpolation="nearest")
    for g in corr_s.geometry:
        _patches(ax, g, dict(STYLE["unchanged"], linewidth=0.4))
    for c in cs:
        ax.scatter([c["lon"]], [c["lat"]], s=60, facecolors="none",
                   edgecolors=COLOR[c["change_type"]], linewidths=1.5, zorder=15)
    for k, (_, b) in enumerate(clusters, 1):
        ax.add_patch(matplotlib.patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                                  fill=False, edgecolor="yellow", lw=1, zorder=16))
        ax.text(b[2], b[3], f"Z{k}", color="yellow", fontsize=6, zorder=17,
                bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"))
    _geo_axes(ax, ext, lat)
    _scalebar(ax, ext, lat)
    ax.set_title("Overview: circles = changes, Z boxes = zoom pages", fontsize=7)

    tx = fig.add_axes([0.45, 0.05, 0.52, 0.88])
    tx.axis("off")
    head = (f"Interferogram {scene['intf_id']}\n"
            f"{scene['start_date']} -> {scene['end_date']}   frame: {scene['frame']}"
            f"   ({'official test scene' if scene['role'] == 'official_test' else 'candidate scene, not in the official test set'})")
    tx.text(0, 1.0, head, fontsize=9, weight="bold", va="top")
    rows = [("Original polygons", stats["original_count"]),
            ("Corrected polygons", stats["corrected_count"]),
            ("Added", stats["added"]), ("Deleted", stats["deleted"]), ("Modified", stats["modified"]),
            ("Ambiguous changes (need review)", stats["needs_review"]),
            ("Flagged hard/doubtful (kept in GT)", stats.get("flagged", 0))]
    for k, (a, b) in enumerate(rows):
        tx.text(0, 0.93 - k * 0.022, f"{a}:", fontsize=8)
        tx.text(0.62, 0.93 - k * 0.022, str(b), fontsize=8, weight="bold")
    y = 0.76
    tx.text(0, y, "Changes", fontsize=8, weight="bold")
    y -= 0.022
    flag_head = False
    for c in cs:
        if c["change_type"] == "FLAG" and not flag_head:
            y -= 0.006
            tx.text(0, y, "Flagged polygons (kept in the corrected GT)", fontsize=8, weight="bold")
            y -= 0.022
            flag_head = True
        if y < 0.02:
            tx.text(0, y, "... continued in the change table", fontsize=6)
            break
        reason = (c.get("qc_flag") if c["change_type"] == "FLAG" else c.get("edit_reason")) or "-"
        z = next(k for k, (cc, _) in enumerate(clusters, 1) if c in cc)
        line = (f"{c['change_id']}  [Z{z}]  {reason}  "
                f"({c['lat']:.5f}N, {c['lon']:.5f}E)")
        tx.text(0, y, line, fontsize=5.8, color="black",
                bbox=dict(facecolor=COLOR[c["change_type"]], alpha=0.35, pad=0.5, edgecolor="none"))
        y -= 0.018
        note = c.get("qc_note") if c["change_type"] == "FLAG" else c.get("edit_notes")
        if note:
            tx.text(0.03, y, f"note: {str(note)[:110]}", fontsize=5.4, style="italic")
            y -= 0.016
        if c.get("needs_review"):
            tx.text(0.03, y, f"CHECK: {c['review_note']}", fontsize=5.4, color="#b00020")
            y -= 0.016
    return fig


def zoom_figure(scene, tif, orig_s, corr_s, cluster, k):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs, b = cluster
    lat = (b[1] + b[3]) / 2
    arr, ext = _read_raster(tif, bounds=b, max_px=900)
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 6.2))
    ids_del = {c["orig_uid"] for c in cs if c["change_type"] in ("DEL", "MOD")}
    ids_new = {c["feat_uid"] for c in cs if c["change_type"] in ("ADD", "MOD")}
    for ax, side in zip(axes, ("Original labels", "Corrected labels")):
        if arr is not None:
            ax.imshow(arr, extent=ext, cmap=CMAP, vmin=-np.pi, vmax=np.pi, interpolation="nearest")
        else:
            ax.set_facecolor("#dddddd")
            ax.text(0.5, 0.02, "outside the labelling raster (AOI crop)", transform=ax.transAxes,
                    ha="center", fontsize=7)
        if side.startswith("Original"):
            for r in orig_s.itertuples():
                if r.orig_uid not in ids_del:
                    _patches(ax, r.geometry, STYLE["unchanged"])
            for c in cs:
                if c["change_type"] == "DEL":
                    _patches(ax, c["old_geom"], STYLE["DEL"], 5)
                elif c["change_type"] == "MOD":
                    _patches(ax, c["old_geom"], STYLE["MOD_old"], 5)
        else:
            for r in corr_s.itertuples():
                if r.feat_uid not in ids_new:
                    _patches(ax, r.geometry, STYLE["unchanged"])
            for c in cs:
                if c["change_type"] == "ADD":
                    _patches(ax, c["new_geom"], STYLE["ADD"], 5)
                elif c["change_type"] == "MOD":
                    _patches(ax, c["old_geom"], dict(STYLE["MOD_old"], linewidth=1.0), 4)
                    _patches(ax, c["new_geom"], STYLE["MOD_new"], 5)
                elif c["change_type"] == "DEL":
                    _patches(ax, c["old_geom"], dict(STYLE["DEL"], facecolor="none", linewidth=0.8), 4)
        for c in cs:                                     # flags: same polygon on both sides
            if c["change_type"] == "FLAG":
                _patches(ax, c["new_geom"], STYLE["FLAG"], 6)
        for c in cs:
            _label(ax, c)
        _geo_axes(ax, ext, lat)
        _scalebar(ax, ext, lat)
        ax.set_title(side, fontsize=9)
    _legend(axes[0])
    fig.suptitle(f"{scene['intf_id']} -- zoom Z{k}: " + ", ".join(c["change_id"] for c in cs),
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def cover_figure(summary, ws, note, n_changes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.92])
    ax.axis("off")
    txt = [
        "Ground-truth relabelling: change review",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}   |   {n_changes} change(s)",
        "",
        "What this is: the subsidence polygons of the listed interferograms were re-inspected and",
        "corrected. This document shows every change against the interferogram it was drawn on",
        "(the same wrapped-phase raster the detection model reads; colour scale -pi..pi, cyclic).",
        "",
        "How to read it: one section per interferogram with changes. The first page shows the whole",
        "scene with every change circled and numbered zoom boxes (Z1, Z2, ...); following pages show",
        "each zoom with the ORIGINAL labels on the left and the CORRECTED labels on the right.",
        "",
        "Change ids, e.g. 20250329_20250409_ADD_001, are the same in this PDF, in change_list.csv",
        "and in gt_changes_v2.gpkg (open it in QGIS to see only the changed polygons).",
        "ADD = polygon added; DEL = polygon removed; MOD = outline changed (old dashed magenta,",
        "new solid orange). CHECK marks a change the automatic matching could not resolve uniquely.",
        "FLAG (purple, dotted) marks a polygon judged hard or doubtful (e.g. a large polygon over",
        "noisy phase). Flagged polygons are NOT removed: they stay in the corrected ground truth.",
    ]
    if note:
        txt += ["", f"Note: {note}"]
    for k, line in enumerate(txt):
        ax.text(0, 1 - k * 0.022, line, fontsize=11 if k == 0 else 8,
                weight="bold" if k == 0 else "normal", va="top")
    y = 1 - (len(txt) + 1) * 0.022
    ax.text(0, y, "Scene summary", fontsize=9, weight="bold")
    y -= 0.022
    cols = ["intf_id", "frame", "original_count", "corrected_count", "added", "deleted", "modified",
            "flagged"]
    hdr = ["interferogram", "frame", "orig", "corr", "add", "del", "mod", "flag"]
    xs = [0, 0.25, 0.37, 0.47, 0.57, 0.67, 0.77, 0.87]
    for x, h in zip(xs, hdr):
        ax.text(x, y, h, fontsize=7, weight="bold")
    for r in summary:
        y -= 0.017
        changed = r["added"] or r["deleted"] or r["modified"] or r.get("flagged")
        for x, c in zip(xs, cols):
            ax.text(x, y, str(r.get(c, 0)), fontsize=7, weight="bold" if changed else "normal",
                    color="black" if changed else "#777777")
    return fig


# --------------------------------------------------------------------------- main


def main(args):
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from . import diff as d

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ws = Path(args.workspace)
    changes, summary, detected, flags = d.run(ws, take_snapshot=True)
    orig, work, scenes = d.load_layers(ws)
    live = work[work["edit_status"].fillna("") != "deleted"]
    sc = {r["intf_id"]: r for r in scenes.to_dict("records")}
    stats = {r["intf_id"]: r for r in summary}
    tag = args.tag or f"review_{datetime.now():%Y%m%d}"
    out = ws_path(ws, "review") / tag
    if out.exists():
        raise SystemExit(f"{out} exists -- packages are never overwritten; pass another --tag")
    (out / "scenes").mkdir(parents=True)

    items = changes + list(flags)
    changed = [i for i in scenes["intf_id"] if any(c["intf_id"] == i for c in items)]
    if args.scenes:
        changed = [i for i in changed if i in set(args.scenes)]
    if not changed:
        logging.warning("no changes to review yet")
    with PdfPages(out / "change_report.pdf") as pdf:
        fig = cover_figure([stats[i] for i in scenes["intf_id"]], ws, args.reviewer_note, len(changes))
        pdf.savefig(fig)
        plt.close(fig)
        for intf in changed:
            s = sc[intf]
            tif = ws / s["raster"]
            cs = sorted([c for c in items if c["intf_id"] == intf],
                        key=lambda c: (c["change_type"] == "FLAG", c["change_id"]))
            dx = 2.777e-05
            clusters = clusters_of(cs, dx)
            o_s, c_s = orig[orig["intf_id"] == intf], live[live["intf_id"] == intf]
            fig = overview_figure(None, s, tif, o_s, c_s, cs, clusters, stats[intf])
            fig.savefig(out / "scenes" / f"{intf}_overview.png", dpi=170)
            pdf.savefig(fig)
            plt.close(fig)
            for k, cl in enumerate(clusters, 1):
                fig = zoom_figure(s, tif, o_s, c_s, cl, k)
                fig.savefig(out / "scenes" / f"{intf}_zoom_Z{k:02d}.png", dpi=170)
                pdf.savefig(fig)
                plt.close(fig)
            n_f = sum(c["change_type"] == "FLAG" for c in cs)
            logging.info(f"{intf}: {len(cs) - n_f} change(s), {n_f} flag(s), {len(clusters)} zoom(s)")

    src = ws_path(ws, "changes")
    for f in ("gt_changes_v2.gpkg", "change_list.csv", "change_summary.csv", "flagged_list.csv"):
        shutil.copy2(src / f, out / f)
    (out / "README.txt").write_text(
        "Change review package\n"
        "=====================\n\n"
        "change_report.pdf     one section per interferogram: overview + zoomed before/after views\n"
        "scenes/               the same figures as PNG images\n"
        "change_list.csv       one row per change (id, type, reason, notes, location, areas)\n"
        "change_summary.csv    one row per interferogram (original/corrected counts, added/deleted/modified)\n"
        "gt_changes_v2.gpkg    the changed polygons only, for QGIS: layers added, deleted,\n"
        "                      modified_original (old outlines), modified_corrected (new outlines),\n"
        "                      change_markers (one point per change, labelled by change_id),\n"
        "                      flagged (hard/doubtful polygons, KEPT in the ground truth)\n"
        "flagged_list.csv      one row per flagged polygon (flag id, flag, note, location)\n\n"
        "Change ids are identical across all of these files. Coordinates are WGS84 (EPSG:4326).\n"
        "Please reply per change id: agree / disagree / comment.\n")
    with open(out / "reviewer_response.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["change_id", "intf_id", "change_type", "verdict (agree/disagree/unsure)", "comment"])
        for c in sorted(items, key=lambda c: c["change_id"]):
            if c["intf_id"] in changed:
                w.writerow([c["change_id"], c["intf_id"], c["change_type"], "", ""])
    zpath = out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out.parent))
    logging.info(f"-> {out}\n-> {zpath}")
