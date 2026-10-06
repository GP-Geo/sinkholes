"""Scene switching and prediction toggling for the relabelling project -- run inside QGIS.

In the QGIS Python console (Plugins > Python Console) of relabel_test_v2.qgz:

    exec(open(QgsProject.instance().homePath() + '/goto_scene.py').read())
    goto('20250329_20250409')     # one scene: filter labels, show its rasters, zoom
    nxt()                         # next scene in priority order
    where()                       # which scene am I on?

    # second pass only -- the prediction groups are OFF in the project:
    predictions_on()              # show the reference model's confidence (continuous)
    pred_threshold(0.5)           # binary layer: confidence > 0.5, and turn it on
    pred_bands()                  # nested classes 0.125/0.25/0.5/0.7/0.9 instead
    pred_opacity(0.4)             # see more of the interferogram
    predictions_off()             # back to blind

    # quality flags -- the polygon STAYS in the GT; select it first (Select Features tool):
    flag('noisy_large', 'big polygon over decorrelated phase')
    flag('uncertain_boundary')    # flags: noisy_large low_coherence uncertain_boundary
                                  #        uncertain_existence other
    unflag()                      # remove the flag from the selected polygons
    flagged()                     # list the flagged polygons of this scene

goto() sets @relabel_intf (new polygons inherit it as their intf_id and dates),
filters working_gt and original_gt, and selects that scene's layer in every
per-scene group -- interferogram and every prediction layer alike. It only
selects *which* scene a group shows: a prediction group that is off stays off,
so the blind pass never sees a prediction by accident. It refuses to switch
while the working layer holds unsaved edits.
"""

import csv
import os

from qgis.core import (QgsColorRampShader, QgsExpressionContextUtils, QgsProject,
                       QgsRasterShader, QgsSingleBandPseudoColorRenderer)
from qgis.PyQt.QtGui import QColor
from qgis.utils import iface

_HOME = QgsProject.instance().homePath()
_WS = os.path.dirname(_HOME)
BANDS = (0.125, 0.25, 0.5, 0.7, 0.9)


def _order():
    with open(os.path.join(_WS, "priority.csv")) as fh:
        return [r["intf_id"] for r in csv.DictReader(fh)]


def _layer(prefix):
    for lyr in QgsProject.instance().mapLayers().values():
        if lyr.name().startswith(prefix):
            return lyr
    raise KeyError(prefix)


def _nodes(role=None, model=None):
    for node in QgsProject.instance().layerTreeRoot().findLayers():
        lyr = node.layer()
        if lyr is None or lyr.customProperty("relabel/intf") is None:
            continue
        if role and lyr.customProperty("relabel/role") != role:
            continue
        if model and lyr.customProperty("relabel/model") != model:
            continue
        yield node, lyr


def _pred_groups(model=None):
    for g in QgsProject.instance().layerTreeRoot().findGroups(True):
        name = g.customProperty("relabel/pred_group")
        if name and (model is None or name == model):
            yield g


def where():
    v = QgsExpressionContextUtils.projectScope(QgsProject.instance()).variable("relabel_intf")
    print(f"current scene: {v}")
    return v


def goto(intf):
    proj = QgsProject.instance()
    # Switching scenes changes a variable, two filters and layer visibility, all of
    # which mark the project modified -- and "Save" on exit then writes this
    # session's project over a rebuilt one (that is how the prediction layers were
    # lost once). Restore the flag, so only real edits trigger the prompt.
    was_dirty = proj.isDirty()
    work = _layer("working_gt")
    if work.isEditable() and work.isModified():
        raise RuntimeError("working_gt has unsaved edits: save (or discard) them first")
    QgsExpressionContextUtils.setProjectVariable(proj, "relabel_intf", intf)
    flt = f"\"intf_id\" = '{intf}'"
    was_editing = work.isEditable()
    if was_editing:
        work.commitChanges()
    work.setSubsetString(flt)
    _layer("original_gt").setSubsetString(flt)
    target, found = None, set()
    for node, lyr in _nodes():
        role = lyr.customProperty("relabel/role")
        if role == "full":
            continue                       # the slow full-scene VRTs stay manual
        on = lyr.customProperty("relabel/intf") == intf
        node.setItemVisibilityChecked(on)
        if on:
            found.add(role)
            if role == "interferogram":
                target = lyr
    if was_editing:
        work.startEditing()
    proj.setDirty(was_dirty)
    if target is not None and iface is not None:
        iface.mapCanvas().setExtent(target.extent())
        iface.mapCanvas().refresh()
        iface.setActiveLayer(work)
    shown = [g.name().split(" -- ")[1] for g in _pred_groups() if g.isVisible()]
    print(f"scene {intf}: {work.featureCount()} working polygon(s); "
          f"raster {'shown' if target else 'NOT FOUND'}; "
          f"predictions {'VISIBLE: ' + ', '.join(shown) if shown else 'hidden (blind)'}"
          + ("" if "pred_conf" in found or not list(_pred_groups()) else
             " -- no prediction for this scene"))


def nxt():
    order = _order()
    cur = where()
    i = order.index(cur) + 1 if cur in order else 0
    if i >= len(order):
        print("last scene reached")
        return
    goto(order[i])


# --------------------------------------------------------------------------- predictions


def predictions_on(model=None, thresholded=False):
    """Show a model's prediction group (default: every model's)."""
    for g in _pred_groups(model):
        g.setItemVisibilityChecked(True)
        for child in g.children():
            if "thresholded" in child.name():
                child.setItemVisibilityChecked(thresholded)
            else:
                child.setItemVisibilityChecked(True)
    print("predictions VISIBLE -- second pass")


def predictions_off():
    for g in _pred_groups():
        g.setItemVisibilityChecked(False)
    print("predictions hidden -- blind")


blind = predictions_off


def _shader(items, kind):
    fn = QgsColorRampShader(0.0, 1.0)
    fn.setColorRampType(kind)
    fn.setColorRampItemList(items)
    sh = QgsRasterShader()
    sh.setRasterShaderFunction(fn)
    return sh


def _set(role, make, opacity, model=None):
    for _, lyr in _nodes(role, model):
        r = QgsSingleBandPseudoColorRenderer(lyr.dataProvider(), 1, make())
        r.setClassificationMin(0.0)
        r.setClassificationMax(1.0)
        r.setOpacity(opacity)
        lyr.setRenderer(r)
        lyr.triggerRepaint()


def pred_threshold(t=0.25, model=None):
    """Binary layer: confidence > t (strict, as the evaluation counts it); turns it on."""
    I = QgsColorRampShader.ColorRampItem
    _set("pred_th", lambda: _shader([I(t, QColor(0, 0, 0, 0), f"<= {t:g}"),
                                     I(float("inf"), QColor(255, 0, 200), f"> {t:g}")],
                                    QgsColorRampShader.Discrete), 0.55, model)
    for g in _pred_groups(model):
        for child in g.children():
            if "thresholded" in child.name():
                child.setName(f"thresholded (> {t:g}; change with pred_threshold())")
                child.setItemVisibilityChecked(True)
    print(f"thresholded layer: confidence > {t:g}"
          + ("" if any(g.isVisible() for g in _pred_groups(model)) else
             "  (group still hidden: predictions_on())"))


def pred_bands(model=None):
    """Nested classes on the thresholded layer: > 0.125, 0.25, 0.5, 0.7, 0.9."""
    I = QgsColorRampShader.ColorRampItem
    cols = [(0, 0, 0, 0), (199, 233, 192), (116, 196, 118), (49, 163, 84), (0, 109, 44),
            (0, 50, 20)]
    edges = (0.0,) + BANDS
    items = [I(edges[k + 1] if k + 1 < len(edges) else float("inf"), QColor(*cols[k]),
               f"> {edges[k]:g}" if k else f"<= {BANDS[0]:g}") for k in range(len(cols))]
    _set("pred_th", lambda: _shader(items, QgsColorRampShader.Discrete), 0.6, model)
    for g in _pred_groups(model):
        for child in g.children():
            if "thresholded" in child.name():
                child.setName("thresholded (bands > 0.125/0.25/0.5/0.7/0.9)")
                child.setItemVisibilityChecked(True)
    print("thresholded layer: bands " + ", ".join(f"> {b:g}" for b in BANDS))


def pred_opacity(alpha=0.6, model=None):
    for role in ("pred_conf", "pred_th"):
        for _, lyr in _nodes(role, model):
            lyr.renderer().setOpacity(alpha)
            lyr.triggerRepaint()
    print(f"prediction opacity {alpha}")


# --------------------------------------------------------------------------- quality flags

QC_FLAGS = ("noisy_large", "low_coherence", "uncertain_boundary", "uncertain_existence", "other")


def _set_flag(value, note):
    work = _layer("working_gt")
    ids = work.selectedFeatureIds()
    if not ids:
        print("select the polygon(s) in working_gt first (Select Features tool)")
        return
    if work.fields().indexOf("qc_flag") < 0:
        raise RuntimeError("working_gt has no qc_flag field yet: run 'sinkholes relabel migrate'")
    started = not work.isEditable()
    if started:
        work.startEditing()
    fi, ni = work.fields().indexOf("qc_flag"), work.fields().indexOf("qc_note")
    for fid in ids:
        work.changeAttributeValue(fid, fi, value)
        if note is not None and ni >= 0:
            work.changeAttributeValue(fid, ni, note)
    if started:                               # we opened the edit session: save it too
        if not work.commitChanges():
            print("SAVE FAILED:", work.commitErrors())
            return
    work.triggerRepaint()
    print(f"{'flagged ' + value if value else 'unflagged'}: {len(ids)} polygon(s)"
          + ("" if started else " -- remember to save your edits"))


def flag(kind, note=None):
    """Mark the selected working_gt polygons as hard / doubtful. They stay in the GT."""
    if kind not in QC_FLAGS:
        raise ValueError(f"flag must be one of {QC_FLAGS}")
    _set_flag(kind, note)


def unflag():
    _set_flag(None, None)


def flagged():
    work = _layer("working_gt")
    rows = [f for f in work.getFeatures() if f["qc_flag"]] if work.fields().indexOf("qc_flag") >= 0 else []
    for f in rows:
        print(f"  {f['feat_uid']}  {str(f['qc_flag']):20s} {str(f['edit_status']):9s} "
              f"{f['qc_note'] if f['qc_note'] else ''}")
    print(f"{len(rows)} flagged polygon(s) in {where()}")


def _check_project_is_current():
    """Warn when the open project predates the prediction rasters in the workspace."""
    import glob
    import json

    models = set()
    for f in glob.glob(os.path.join(_WS, "predictions", "*", "model.json")):
        with open(f) as fh:
            models.add(json.load(fh)["name"])
    in_project = {g.customProperty("relabel/pred_group") for g in _pred_groups()}
    missing = sorted(models - in_project)
    if missing:
        print(f"WARNING: this project has no layers for {missing}, which exist in predictions/. "
              "It is an older copy of relabel_test_v2.qgz. Close QGIS WITHOUT saving and reopen "
              "the project (or rebuild it: scripts/relabel/build_qgis_project.py).")


_check_project_is_current()
