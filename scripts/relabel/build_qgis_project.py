"""Build the relabelling QGIS project (and style a changes GeoPackage) with PyQGIS.

Runs under QGIS's own Python, not the project environment:

    scripts/relabel/qgis_python.sh scripts/relabel/build_qgis_project.py \
        --workspace <workspace>

    scripts/relabel/qgis_python.sh scripts/relabel/build_qgis_project.py \
        --style_changes <package>/gt_changes_v2.gpkg

The project stores RELATIVE paths, so the same .qgz opens from
/home/labs/rudich/... on WEXAC and from /Volumes/rudich/... on a Mac.

Editing conveniences configured on working_gt:
  * feat_uid   = uuid() on insert (identity of a new polygon; never typed)
  * intf_id    = @relabel_intf on insert (the scene goto() selected)
  * start/end  = derived from @relabel_intf on insert
  * edit_status: drop-down; 'added' on insert; set to 'modified' automatically
                 when an original polygon is edited (unless marked 'deleted')
  * edit_reason: drop-down; a soft constraint asks for it on any change
  * edited_by / edit_timestamp: refreshed on every edit
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

REASONS = ("missing_sinkhole", "not_subsidence", "boundary_correction", "split", "merge",
           "duplicate", "wrong_scene", "other")
STATUSES = ("unchanged", "added", "modified", "deleted")

# matplotlib 'twilight' sampled at 9 stops: cyclic, so -pi and +pi share a colour
TWILIGHT = ["#e2d9e2", "#9caac6", "#6276ba", "#5a3c9d", "#2f1436", "#6f2251", "#ad4d4a",
            "#c99784", "#e2d9e2"]


def qgis_init():
    from qgis.core import QgsApplication

    app = QgsApplication([], False)
    app.initQgis()
    return app


def phase_renderer(layer):
    from qgis.core import (QgsColorRampShader, QgsGradientColorRamp, QgsGradientStop,
                           QgsRasterShader, QgsSingleBandPseudoColorRenderer)
    from qgis.PyQt.QtGui import QColor

    n = len(TWILIGHT) - 1
    stops = [QgsGradientStop(k / n, QColor(c)) for k, c in enumerate(TWILIGHT[1:-1], 1)]
    ramp = QgsGradientColorRamp(QColor(TWILIGHT[0]), QColor(TWILIGHT[-1]), False, stops)
    shader_fn = QgsColorRampShader(-math.pi, math.pi, ramp)
    shader_fn.classifyColorRamp(64, -1)
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(shader_fn)
    r = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    r.setClassificationMin(-math.pi)
    r.setClassificationMax(math.pi)
    layer.setRenderer(r)


def fill_symbol(geom_type, outline, width, fill="transparent", dash=False):
    from qgis.core import QgsFillSymbol

    props = {"color": fill if fill != "transparent" else "0,0,0,0",
             "outline_color": outline, "outline_width": str(width), "outline_width_unit": "MM",
             "outline_style": "dash" if dash else "solid",
             "style": "solid" if fill != "transparent" else "no"}
    return QgsFillSymbol.createSimple(props)


def style_working(layer):
    from qgis.core import QgsCategorizedSymbolRenderer, QgsRendererCategory

    cats = [
        ("unchanged", fill_symbol(None, "255,235,59,255", 0.35), "unchanged"),
        ("added", fill_symbol(None, "57,255,20,255", 0.7, "57,255,20,70"), "added"),
        ("modified", fill_symbol(None, "255,145,0,255", 0.7, "255,145,0,50"), "modified"),
        ("deleted", fill_symbol(None, "255,23,68,255", 0.6, "255,23,68,60", dash=True),
         "deleted (kept for the audit trail, not exported)"),
    ]
    layer.setRenderer(QgsCategorizedSymbolRenderer(
        "edit_status", [QgsRendererCategory(v, s, lbl) for v, s, lbl in cats]))


def configure_form(layer):
    from qgis.core import (QgsDefaultValue, QgsEditorWidgetSetup, QgsFieldConstraints)

    f = layer.fields()
    idx = f.indexOf
    cfg = layer.editFormConfig()

    def ro(name):
        if idx(name) >= 0:
            cfg.setReadOnly(idx(name), True)

    for name in ("feat_uid", "orig_uid", "orig_fid", "id", "layer", "path"):
        ro(name)
    layer.setEditFormConfig(cfg)

    def default(name, expr, on_update=False):
        if idx(name) >= 0:
            layer.setDefaultValueDefinition(idx(name), QgsDefaultValue(expr, on_update))

    default("feat_uid", "uuid('WithoutBraces')")
    default("intf_id", "@relabel_intf")
    default("start_date", "to_date(left(@relabel_intf, 8), 'yyyyMMdd')")
    default("end_date", "to_date(right(@relabel_intf, 8), 'yyyyMMdd')")
    default("platform", "'TSX'")
    # Re-evaluated on every edit of a feature: a new polygon is 'added', an
    # original that is touched becomes 'modified', a soft delete stays deleted.
    # Only a declaration -- 'relabel diff' decides from the geometry.
    default("edit_status",
            "CASE WHEN \"orig_uid\" IS NULL THEN 'added' "
            "WHEN \"edit_status\" = 'deleted' THEN 'deleted' ELSE 'modified' END",
            on_update=True)
    default("edited_by", "coalesce(@user_full_name, @user_account_name)", on_update=True)
    default("edit_timestamp", "format_date(now(), 'yyyy-MM-ddTHH:mm:ss')", on_update=True)

    def valuemap(name, values):
        if idx(name) >= 0:
            layer.setEditorWidgetSetup(idx(name), QgsEditorWidgetSetup(
                "ValueMap", {"map": [{v: v} for v in values]}))

    valuemap("edit_status", STATUSES)
    valuemap("edit_reason", ("",) + REASONS)
    if idx("edit_reason") >= 0:
        layer.setConstraintExpression(
            idx("edit_reason"),
            "\"edit_status\" = 'unchanged' OR \"edit_reason\" IS NOT NULL AND \"edit_reason\" <> ''",
            "say why this polygon was added / modified / deleted")
        layer.setFieldConstraint(idx("edit_reason"), QgsFieldConstraints.ConstraintExpression,
                                 QgsFieldConstraints.ConstraintStrengthSoft)
    if idx("edit_notes") >= 0:
        layer.setEditorWidgetSetup(idx("edit_notes"), QgsEditorWidgetSetup(
            "TextEdit", {"IsMultiline": True}))


def build_project(ws):
    from qgis.core import (QgsCoordinateReferenceSystem, QgsExpressionContextUtils, QgsProject,
                           QgsRasterLayer, QgsVectorLayer)

    ws = Path(ws).resolve()
    qdir = ws / "qgis"
    qdir.mkdir(exist_ok=True)
    proj = QgsProject.instance()
    proj.clear()
    proj.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
    proj.writeEntryBool("Paths", "/Absolute", False)
    proj.setFileName(str(qdir / "relabel_test_v2.qgz"))
    with open(ws / "scenes.csv") as fh:
        scenes = list(csv.DictReader(fh))
    first = scenes[0]["intf_id"]
    QgsExpressionContextUtils.setProjectVariable(proj, "relabel_intf", first)
    root = proj.layerTreeRoot()

    g_edit = root.addGroup("1. EDIT -- working labels")
    g_ref = root.addGroup("2. Reference -- original labels (read-only)")
    g_ctx = root.addGroup("3. Context")
    g_ras = root.addGroup("4. Interferograms (tgeo_int wrapped phase, pipeline pixels)")
    g_full = root.addGroup("5. Full raw scenes (VRT, slow; outside the AOI crop)")

    work = QgsVectorLayer(f"{ws}/gt/gt_test_working.gpkg|layername=working_gt", "working_gt", "ogr")
    assert work.isValid(), "working gpkg"
    style_working(work)
    configure_form(work)
    work.setSubsetString(f"\"intf_id\" = '{first}'")
    proj.addMapLayer(work, False)
    g_edit.addLayer(work)

    orig = QgsVectorLayer(f"{ws}/gt/gt_test_original.gpkg|layername=original_gt",
                          "original_gt (immutable)", "ogr")
    assert orig.isValid(), "original gpkg"
    orig.setReadOnly(True)
    orig.setRenderer(fill_symbol_renderer("0,229,255,255", 0.45, dash=True))
    orig.setSubsetString(f"\"intf_id\" = '{first}'")
    proj.addMapLayer(orig, False)
    g_ref.addLayer(orig)

    aux = f"{ws}/gt/aux_context.gpkg"
    for name, style in (("aoi_window", ("255,255,0,255", 0.6, False)),
                        ("predictable_area", ("118,255,3,255", 0.4, True)),
                        ("lidar2022_gate", ("200,200,200,255", 0.4, False))):
        lyr = QgsVectorLayer(f"{aux}|layername={name}", name, "ogr")
        lyr.setRenderer(fill_symbol_renderer(*style[:2], dash=style[2]))
        lyr.setReadOnly(True)
        proj.addMapLayer(lyr, False)
        g_ctx.addLayer(lyr)

    for k, s in enumerate(scenes):
        tif = ws / s["raster"]
        if not tif.exists():
            continue
        tag = "" if s["role"] == "official_test" else "  [candidate]"
        rl = QgsRasterLayer(str(tif), f"{s['intf_id']}{tag}")
        phase_renderer(rl)
        proj.addMapLayer(rl, False)
        node = g_ras.addLayer(rl)
        node.setItemVisibilityChecked(s["intf_id"] == first)
        vrt = ws / "rasters" / f"{s['intf_id']}_int_full.vrt"
        if vrt.exists():
            fl = QgsRasterLayer(str(vrt), f"{s['intf_id']} (full)")
            phase_renderer(fl)
            proj.addMapLayer(fl, False)
            g_full.addLayer(fl).setItemVisibilityChecked(False)
    g_full.setExpanded(False)
    g_full.setItemVisibilityChecked(False)

    first_tif = ws / scenes[0]["raster"]
    if first_tif.exists():
        ext = QgsRasterLayer(str(first_tif), "tmp").extent()
        from qgis.core import QgsReferencedRectangle

        proj.viewSettings().setDefaultViewExtent(
            QgsReferencedRectangle(ext, QgsCoordinateReferenceSystem("EPSG:4326")))
    ok = proj.write()
    print(f"{'wrote' if ok else 'FAILED'} {proj.fileName()}")


def fill_symbol_renderer(outline, width, dash=False):
    from qgis.core import QgsSingleSymbolRenderer

    return QgsSingleSymbolRenderer(fill_symbol(None, outline, width, dash=dash))


def style_changes(gpkg):
    """Embed default styles in a changes GeoPackage (QGIS applies them on open)."""
    from qgis.core import (QgsPalLayerSettings, QgsTextFormat, QgsVectorLayer,
                           QgsVectorLayerSimpleLabeling, QgsMarkerSymbol, QgsSingleSymbolRenderer)
    from qgis.PyQt.QtGui import QColor

    styles = {
        "added": ("57,255,20,255", 0.8, "57,255,20,80", False),
        "deleted": ("255,23,68,255", 0.8, "255,23,68,60", True),
        "modified_original": ("255,0,255,255", 0.6, "transparent", True),
        "modified_corrected": ("255,145,0,255", 0.8, "transparent", False),
    }
    for name, (oc, w, fill, dash) in styles.items():
        lyr = QgsVectorLayer(f"{gpkg}|layername={name}", name, "ogr")
        if not lyr.isValid():
            continue
        lyr.setRenderer(QgsSingleSymbolRenderer(fill_symbol(None, oc, w, fill, dash)))
        lyr.saveStyleToDatabase(name, "relabel review style", True, "")
    lyr = QgsVectorLayer(f"{gpkg}|layername=change_markers", "change_markers", "ogr")
    if lyr.isValid():
        sym = QgsMarkerSymbol.createSimple({"name": "circle", "color": "0,0,0,0",
                                            "outline_color": "255,255,0,255",
                                            "outline_width": "0.6", "size": "6"})
        lyr.setRenderer(QgsSingleSymbolRenderer(sym))
        pal = QgsPalLayerSettings()
        pal.fieldName = "change_id"
        fmt = QgsTextFormat()
        fmt.setColor(QColor("black"))
        fmt.background().setEnabled(True)
        fmt.background().setFillColor(QColor(255, 255, 0, 220))
        pal.setFormat(fmt)
        lyr.setLabeling(QgsVectorLayerSimpleLabeling(pal))
        lyr.setLabelsEnabled(True)
        lyr.saveStyleToDatabase("change_markers", "relabel review style", True, "")
    print(f"styled {gpkg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace")
    p.add_argument("--style_changes")
    a = p.parse_args()
    app = qgis_init()
    try:
        if a.workspace:
            build_project(a.workspace)
        if a.style_changes:
            style_changes(a.style_changes)
    finally:
        app.exitQgis()


if __name__ == "__main__":
    main()
