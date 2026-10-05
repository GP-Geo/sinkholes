"""Scene switching for the relabelling project -- run inside QGIS.

In the QGIS Python console (Plugins > Python Console) of relabel_test_v2.qgz:

    exec(open(QgsProject.instance().homePath() + '/goto_scene.py').read())
    goto('20250329_20250409')     # one scene: filter labels, show its raster, zoom
    nxt()                         # next scene in priority order
    where()                       # which scene am I on?

goto() sets the project variable @relabel_intf (new polygons inherit it as
their intf_id and dates), filters working_gt and original_gt to that scene, and
shows only that scene's interferogram. It refuses to switch while the working
layer holds unsaved edits, so a polygon can never be saved under a scene you
are no longer looking at.
"""

import csv
import os

from qgis.core import QgsExpressionContextUtils, QgsProject
from qgis.utils import iface

_HOME = QgsProject.instance().homePath()
_WS = os.path.dirname(_HOME)


def _order():
    path = os.path.join(_WS, "priority.csv")
    with open(path) as fh:
        return [r["intf_id"] for r in csv.DictReader(fh)]


def _layer(prefix):
    for lyr in QgsProject.instance().mapLayers().values():
        if lyr.name().startswith(prefix):
            return lyr
    raise KeyError(prefix)


def where():
    v = QgsExpressionContextUtils.projectScope(QgsProject.instance()).variable("relabel_intf")
    print(f"current scene: {v}")
    return v


def goto(intf):
    proj = QgsProject.instance()
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
    root = proj.layerTreeRoot()
    target = None
    for node in root.findLayers():
        name = node.layer().name()
        if len(name) >= 17 and name[:8].isdigit() and name[8] == "_" and name[9:17].isdigit():
            on = name.startswith(intf) and "(full)" not in name
            node.setItemVisibilityChecked(on)
            if on:
                target = node.layer()
    if was_editing:
        work.startEditing()
    if target is not None and iface is not None:
        iface.mapCanvas().setExtent(target.extent())
        iface.mapCanvas().refresh()
        iface.setActiveLayer(work)
    print(f"scene {intf}: {work.featureCount()} working polygon(s); "
          f"raster {'shown' if target else 'NOT FOUND'}")


def nxt():
    order = _order()
    cur = where()
    i = order.index(cur) + 1 if cur in order else 0
    if i >= len(order):
        print("last scene reached")
        return
    goto(order[i])
