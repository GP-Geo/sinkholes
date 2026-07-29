"""Per-interferogram metadata: the coordinate dictionary and 11-day chains.

An interferogram id is ``YYYYMMDD_YYYYMMDD`` (start_end). The coordinate
dictionary (assets/intf_coord.json, built from the .ers headers) maps each id
to its raster geometry, byte order, frame (North/South), LiDAR mask source and
positive-patch count (``nonz_num``, filled by ``sinkholes count-positives``;
``'none'`` means "no patches on disk" and excludes the id from training).
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .paths import asset

#: An interferogram id embedded in a longer string, e.g. a patch file name.
INTF_ID_RE = re.compile(r"\d{8}_\d{8}")


@dataclass(frozen=True)
class IntfMeta:
    """One interferogram's entry of the coordinate dictionary."""

    id: str
    east: float          # raw scene origin longitude (top-left)
    north: float         # raw scene origin latitude (top-left)
    dx: float
    dy: float
    ncells: int          # raster width
    nlines: int          # raster height
    byte_order: str      # 'MSBFirst' scenes need a byteswap after np.fromfile
    frame: str           # 'North' | 'South'
    lidar_mask: str      # source id in lidar_mask_polygs.shp, or 'no_mask'
    nonz_num: Any        # int, or 'none' when no patches exist for this id


def load_coord_dict(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """The raw coordinate dictionary; defaults to the committed asset."""
    with open(path or asset("intf_coord.json")) as fh:
        return json.load(fh)


@lru_cache(maxsize=4)
def _cached_coord_dict(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    return load_coord_dict(path)


def intf_meta(intf_id: str, path: Optional[str] = None) -> IntfMeta:
    """Typed metadata for one interferogram (dictionary parsed once per path)."""
    m = _cached_coord_dict(path)[intf_id]
    return IntfMeta(
        id=intf_id,
        east=float(m["east"]),
        north=float(m["north"]),
        dx=float(m["dx"]),
        dy=float(m["dy"]),
        ncells=int(m["ncells"]),
        nlines=int(m["nlines"]),
        byte_order=m["byte_order"],
        frame=m["frame"],
        lidar_mask=m.get("lidar_mask", "no_mask"),
        nonz_num=m.get("nonz_num", "none"),
    )


def intf_id_from_filename(name: str) -> str:
    """Interferogram id from a raw scene file name.

    Scene files embed the two acquisition timestamps at fixed positions:
    characters [9:17] are the start date and [24:33] is ``_YYYYMMDD`` for the
    end date (e.g. ``filt_topo_20190205T...``).
    """
    stem = name.split(".")[0]
    return stem[9:17] + stem[24:33]


def parse_intf_id(intf_id: str) -> Tuple[datetime, datetime, int]:
    """(start, end, duration_days) of an id."""
    s, e = intf_id.split("_")
    sd = datetime.strptime(s, "%Y%m%d")
    ed = datetime.strptime(e, "%Y%m%d")
    return sd, ed, (ed - sd).days


def find_11day_sequences(
    meta: Dict[str, Dict[str, Any]],
    k_prev: int = 2,
    step_days: int = 11,
    restrict_to: Optional[List[str]] = None,
    require_current_nonz_gt0: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Per-interferogram chains of the k previous same-frame acquisitions.

    A predecessor must exist exactly ``i * step_days`` earlier (i = 1..k), have
    the same frame, and itself span ``step_days``. Ids missing any predecessor
    are dropped from the returned valid list. ``prevs`` is ordered oldest ->
    newest — the temporal stack everywhere in this project is chronological
    with the current interferogram last.

    Returns ({id: {'prevs': [...], 'frame': ...}}, [valid ids]).
    """

    def make_key(sd, ed) -> str:
        return sd.strftime("%Y%m%d") + "_" + ed.strftime("%Y%m%d")

    daydiff = {k: parse_intf_id(k)[2] for k in meta}
    curr_keys = set(meta) if restrict_to is None else set(meta) & set(restrict_to)

    frame_groups: Dict[str, set] = {"North": set(), "South": set()}
    for k, info in meta.items():
        if info.get("frame") in frame_groups:
            frame_groups[info["frame"]].add(k)

    chains: Dict[str, Dict[str, Any]] = {}
    valid: List[str] = []
    for cur in curr_keys:
        info = meta.get(cur)
        if not info or info.get("frame") not in frame_groups:
            continue
        sd, ed, dd = parse_intf_id(cur)
        if dd != step_days:
            continue
        if require_current_nonz_gt0:
            try:
                nonz = int(info.get("nonz_num", 0))
            except (TypeError, ValueError):
                nonz = 0
            if nonz <= 0:
                continue

        group = frame_groups[info["frame"]]
        prevs: List[str] = []
        for i in range(k_prev, 0, -1):  # oldest first
            pk = make_key(sd - timedelta(days=i * step_days), ed - timedelta(days=i * step_days))
            if pk not in group or daydiff.get(pk) != step_days:
                prevs = []
                break
            prevs.append(pk)
        else:
            chains[cur] = {"prevs": prevs, "frame": info["frame"]}
            valid.append(cur)

    return chains, valid


def lidar_source_for(intf_id: str, mapping_path: Optional[str] = None) -> str:
    """LiDAR mask source for an id, from the fixed-column lidar_intf_mask.txt.

    Each line carries a start date at [8:16], an end date at [24:32] and the
    source id at [40:49]. A line matching both dates or just the start date
    assigns the source; later lines override earlier ones.
    """
    mask = "no_mask"
    with open(mapping_path or asset("lidar_intf_mask.txt")) as fh:
        for line in fh:
            if intf_id[:8] == line[8:16] and intf_id[9:17] == line[24:32]:
                mask = line[40:49]
            elif intf_id[:8] == line[8:16]:
                mask = line[40:49]
    return mask


#: .ers header fields -> coordinate dictionary keys.
_ERS_FIELDS = {
    "NrOfLines": ("nlines", int),
    "NrOfCellsPerLine": ("ncells", int),
    "Northings": ("north", float),
    "Eastings": ("east", float),
    "Ydimension": ("dy", float),
    "Xdimension": ("dx", float),
    "ByteOrder": ("byte_order", str),
}


def parse_ers_header(path: str) -> Dict[str, Any]:
    """Geometry fields of one .ers header file."""
    out: Dict[str, Any] = {}
    with open(path) as fh:
        for line in fh:
            for field, (key, cast) in _ERS_FIELDS.items():
                if field in line:
                    out[key] = cast(line.strip().split()[-1])
    missing = [k for k, _ in _ERS_FIELDS.values() if k not in out]
    if missing:
        raise ValueError(f"{path}: header is missing {missing}")
    return out
