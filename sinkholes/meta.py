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

#: The value ``lidar_source_for`` writes when the mapping covers an id.
NO_LIDAR_MASK = "no_mask"

#: LiDAR survey used for interferograms the mapping does not cover.
#:
#: assets/lidar_intf_mask.txt stops at start date 20240605 and
#: assets/lidar_mask_polygs.shp holds four surveys (2019, 2020, 2021, 2022), so
#: every interferogram from 20240615_20240626 onward -- 112 of 437, all of 2025
#: and 2026 -- has no mapping of its own and the dictionary records it as
#: ``no_mask``.
#:
#: ``no_mask`` used to reach :func:`rasterise_lidar_gates`, whose
#: fallback test only catches None/''/'none'/'null'. The literal string missed
#: it, matched nothing in the shapefile and tripped the "not in shapefile ->
#: using ALL polygons" branch, so those scenes were silently gated by the UNION
#: of all four footprints -- the widest area available, not one matched to the
#: scene's own date. Resolving to the LAST REAL SURVEY instead is both narrower
#: and explicit. Change this constant when a newer survey is added to the
#: shapefile, and add its rows to lidar_intf_mask.txt.
LIDAR_FALLBACK_SOURCE = "LiDAR2022"


def resolve_lidar_mask(value: Optional[str]) -> str:
    """The LiDAR source id a consumer should gate on.

    Maps the dictionary's ``no_mask`` (and the empty/None spellings) onto
    :data:`LIDAR_FALLBACK_SOURCE`. Applied at read time in :func:`intf_meta`
    rather than written into assets/intf_coord.json, so the dictionary keeps
    recording what the mapping file actually says and
    ``sinkholes prepare-metadata`` still warns about ids it does not cover.
    """
    if value is None or str(value).strip().lower() in ("", "none", "null", NO_LIDAR_MASK):
        return LIDAR_FALLBACK_SOURCE
    return value


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
    lidar_mask: str      # source id in lidar_mask_polygs.shp; ids the mapping
                         # does not cover resolve to LIDAR_FALLBACK_SOURCE,
                         # never to 'no_mask' (see resolve_lidar_mask)
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
        lidar_mask=resolve_lidar_mask(m.get("lidar_mask")),
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


#: A dense lookback: every 11-day slot from 1 to ``lookback``. The spelling
#: ``find_11day_sequences`` implies, and the schedule a probe should use when
#: the question is *where* attention lands rather than how cheaply to get there.
DENSE_SCHEDULE = "dense"


def parse_history_schedule(spec: str, lookback: int) -> List[int]:
    """Candidate offsets, in 11-day slots, newest first (never including 0).

    ``spec`` is ``"dense"`` — every slot out to ``lookback`` — or a comma-listed
    ``step:until`` grammar, e.g. ``"1:6,2:12,4:40"``: every slot out to 6, then
    every second out to 12, then every fourth out to 40. Thinning the old end
    is what lets a lookback reach back a year without paying for a frame at
    every slot; the shared encoder costs are linear in the number of frames, so
    ``"1:6,2:12,4:40"`` is 16 frames where dense-40 is 40.

    This is only meaningful because the positional encoding takes real offsets
    (``models/temporal_attention.py``). Once it does, a deliberately skipped
    slot and a missing acquisition are the same thing to the model, which is
    the whole reason one mechanism can serve both.
    """
    if spec == DENSE_SCHEDULE:
        return list(range(1, int(lookback) + 1))

    offsets: List[int] = []
    cursor = 1
    for part in spec.split(","):
        try:
            step_s, until_s = part.split(":")
            step, until = int(step_s), int(until_s)
        except ValueError:
            raise ValueError(
                f"bad history schedule segment {part!r} in {spec!r}; expected "
                f"'step:until' (e.g. '1:6,2:12,4:40') or '{DENSE_SCHEDULE}'."
            ) from None
        if step < 1:
            raise ValueError(f"history schedule step must be >= 1; got {step} in {spec!r}.")
        until = min(until, int(lookback))
        while cursor <= until:
            offsets.append(cursor)
            cursor += step
    if not offsets:
        raise ValueError(f"history schedule {spec!r} selects no offsets at lookback {lookback}.")
    return offsets


def frame_groups_of(meta: Dict[str, Dict[str, Any]]) -> Dict[str, set]:
    """{'North': {ids...}, 'South': {ids...}} — built once, reused per lookup."""
    groups: Dict[str, set] = {"North": set(), "South": set()}
    for key, info in meta.items():
        if info.get("frame") in groups:
            groups[info["frame"]].add(key)
    return groups


def select_history(
    intf_id: str,
    meta: Dict[str, Dict[str, Any]],
    *,
    lookback: int = 10,
    schedule: str = DENSE_SCHEDULE,
    step_days: int = 11,
    groups: Optional[Dict[str, set]] = None,
) -> Tuple[List[str], List[int]]:
    """The available history of one interferogram, holes permitted.

    Where :func:`find_11day_sequences` demands an unbroken run of ``k``
    predecessors and drops the interferogram otherwise, this keeps whatever the
    archive actually holds and reports each frame's real age alongside it. That
    difference is the whole cost of strictness: the archive is missing 12
    acquisition slots on the North frame and 33 on South, so at a 40-slot
    lookback the strict rule leaves 20 usable interferograms out of 273, while
    this leaves all 273 with a mean of 31 frames each.

    Returns ``(ids, offsets)`` oldest first, with the current interferogram last
    at offset 0 — the same chronological convention as ``seq_dict[id]['prevs']
    + [id]``, so a caller that already builds stacks that way only has to carry
    the offsets alongside. Offsets are in slot units, not days.
    """
    info = meta.get(intf_id)
    if info is None or info.get("frame") not in ("North", "South"):
        raise KeyError(f"{intf_id} has no usable frame in the coordinate dictionary")
    group = (groups or frame_groups_of(meta))[info["frame"]]

    start, end, _ = parse_intf_id(intf_id)
    ids: List[str] = []
    offsets: List[int] = []
    # Oldest first, so walk the candidate offsets from the far end inwards.
    for offset in sorted(parse_history_schedule(schedule, lookback), reverse=True):
        shift = timedelta(days=offset * step_days)
        key = (start - shift).strftime("%Y%m%d") + "_" + (end - shift).strftime("%Y%m%d")
        # Same frame, and itself an 11-day interferogram — the two conditions
        # find_11day_sequences applies. Only the "no gaps" rule is dropped.
        if key in group and parse_intf_id(key)[2] == step_days:
            ids.append(key)
            offsets.append(offset)
    ids.append(intf_id)
    offsets.append(0)
    return ids, offsets


def lidar_source_for(intf_id: str, mapping_path: Optional[str] = None) -> str:
    """LiDAR mask source for an id, from the fixed-column lidar_intf_mask.txt.

    Each line carries a start date at [8:16], an end date at [24:32] and the
    source id at [40:49]. A line matching both dates or just the start date
    assigns the source; later lines override earlier ones.

    Reports what the mapping file says, so an uncovered id comes back as
    ``no_mask`` and ``sinkholes prepare-metadata`` can warn about it. Consumers
    that need a survey to gate on call :func:`resolve_lidar_mask` instead --
    :func:`intf_meta` already does.
    """
    mask = NO_LIDAR_MASK
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
