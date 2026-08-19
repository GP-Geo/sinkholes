#!/usr/bin/env python3
"""
Browser-based interferogram viewer.

Runs on the remote server next to the data; you open it in your local
browser through an SSH port-forward:

    (on the cluster)  python intf_viewer.py --polyg_path <predicted_polygs.shp or dir> [--port 8050]
    (on your laptop)  ssh -L 8050:localhost:8050 <cluster>
    then browse to    http://localhost:8050

Features:
  - full-resolution interferogram display with smooth zoom/pan (Leaflet)
  - predicted polygons overlaid (combined shapefile with intf_key/start_date
    columns, or a directory of per-intf *_predicted_polygs.shp)
  - step back/forth in time with arrow keys, North and South frames as
    separate chronological sequences (switching frames jumps to the
    nearest date)
  - the current zoom/center is kept while stepping, so you can stay locked
    on one area across time
  - neighbor interferograms are pre-rendered in the background for fast
    navigation
"""
import argparse
import io
import json
import logging
import os
import re
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import matplotlib
matplotlib.use('Agg')
from PIL import Image

try:
    from matplotlib import colormaps as _colormaps   # matplotlib >= 3.6
    def get_cmap(name):
        return _colormaps[name]
except ImportError:                                    # older matplotlib
    from matplotlib import cm as _cm
    def get_cmap(name):
        return _cm.get_cmap(name)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

Image.MAX_IMAGE_PIXELS = None  # interferograms are large; we trust our own files


# ----------------------------------------------------------------- args
def get_args():
    p = argparse.ArgumentParser(description='Browser-based interferogram viewer')
    p.add_argument('--input_dir', type=str,
                   default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/',
                   help='directory with the tgeo_int_*.unw files')
    p.add_argument('--intf_dict_path', type=str, default='./intf_coord.json',
                   help='per-interferogram coordinate dictionary')
    p.add_argument('--polyg_path', type=str,
                   default='/home/labs/rudich/Rudich_Collaboration/sinkholes/pred_outputs2/'
                           'job_train_temporal_2019_2021_negrings04_2026-04-28_17h59checkpoint_epoch55/'
                           'job_full_2019_2023_hann_07_13_17h07/'
                           'job_train_temporal_2019_2021_negrings04_2026-04-28_17h59checkpoint_epoch55_2019_2023_combined.shp',
                   help='predicted polygons: a combined shapefile, or a directory '
                        'of per-intf *_predicted_polygs.shp')
    p.add_argument('--days_diff', type=int, default=11,
                   help='only show interferograms with this duration in days')
    p.add_argument('--port', type=int, default=8050)
    p.add_argument('--cache_dir', type=str, default='./intf_viewer_cache',
                   help='where rendered images are cached')
    p.add_argument('--fmt', type=str, default='jpg', choices=['jpg', 'png'],
                   help='rendered image format (jpg = smaller/faster, png = lossless)')
    p.add_argument('--clip_pct', nargs=2, type=float, default=[2.0, 98.0],
                   help='percentile clip for display normalization')
    p.add_argument('--legacy_polyg_correction', action='store_true',
                   help='apply the legacy x4000-origin lon correction; only for '
                        'shapefiles written before the polygonization coord fix '
                        '(e.g. 2020_2023_pred_11days.shp). Newer runs such as the '
                        'hann full-range 2019-2023 combined shp are already '
                        'correctly georeferenced.')
    p.add_argument('--gt_polyg_path', type=str,
                   default='/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/sub_20260701.shp',
                   help='mapped subsidence GT polygons shapefile')
    p.add_argument('--lidar_mask_path', type=str, default='lidar_mask_polygs.shp',
                   help='LiDAR valid-coverage polygons shapefile (has a "source" '
                        'column matching intf_coord.json\'s lidar_mask field, e.g. LiDAR2021)')
    return p.parse_args()


ARGS = get_args()

# ----------------------------------------------------------------- index
with open(ARGS.intf_dict_path) as f:
    COORD = json.load(f)


def parse_key(key):
    s, e = key.split('_')
    return datetime.strptime(s, '%Y%m%d'), datetime.strptime(e, '%Y%m%d')


def build_sequences():
    seqs = {'North': [], 'South': []}
    for key, info in COORD.items():
        frame = info.get('frame')
        if frame not in seqs:
            continue
        try:
            sd, ed = parse_key(key)
        except ValueError:
            continue
        if (ed - sd).days != ARGS.days_diff:
            continue
        seqs[frame].append(key)
    for frame in seqs:
        seqs[frame].sort()
    return seqs


SEQS = build_sequences()
logging.info('11-day interferograms: %d North, %d South',
             len(SEQS['North']), len(SEQS['South']))

# map intf key -> .unw filename (filename carries times, key carries dates only)
UNW_BY_KEY = {}
for fn in os.listdir(ARGS.input_dir):
    if not fn.endswith('.unw'):
        continue
    m = re.match(r'tgeo_int_(\d{8})T\d+_(\d{8})T\d+\.unw$', fn)
    if m:
        UNW_BY_KEY[f'{m.group(1)}_{m.group(2)}'] = fn
logging.info('found %d .unw files in %s', len(UNW_BY_KEY), ARGS.input_dir)

# map intf key -> pre-rendered .unw.jpg filename, when one exists alongside
# the .unw (not all intfs have one). Used as a fast path for the default
# 'jet' colormap only, since we don't know the clip/colormap the source jpg
# was rendered with; other colormaps still go through the .unw pipeline.
JPG_BY_KEY = {}
for fn in os.listdir(ARGS.input_dir):
    if not fn.endswith('.unw.jpg'):
        continue
    m = re.match(r'tgeo_int_(\d{8})T\d+_(\d{8})T\d+\.unw\.jpg$', fn)
    if m:
        JPG_BY_KEY[f'{m.group(1)}_{m.group(2)}'] = fn
logging.info('found %d pre-rendered .unw.jpg files in %s (fast path for jet)',
             len(JPG_BY_KEY), ARGS.input_dir)

# ----------------------------------------------------------------- polygons
POLYG_GDF = None       # combined-shapefile mode
POLYG_DIR = None       # per-intf-directory mode
def _to_lonlat(gdf):
    """Match view_intf_quick.py: display coordinates are EPSG:4326 lon/lat."""
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs('EPSG:4326')
    # datetime64 columns (e.g. the GT shapefile's start_date/end_date/timestamp)
    # break GeoDataFrame.to_json(), which can't serialize pandas Timestamps
    import pandas as pd
    datetime_cols = [c for c in gdf.columns if c != gdf.geometry.name
                     and pd.api.types.is_datetime64_any_dtype(gdf[c])]
    if datetime_cols:
        gdf = gdf.copy()
        for c in datetime_cols:
            gdf[c] = gdf[c].astype(str)
    return gdf


if ARGS.polyg_path:
    import geopandas as gpd
    if os.path.isdir(ARGS.polyg_path):
        POLYG_DIR = ARGS.polyg_path
        logging.info('polygons: per-intf directory %s', POLYG_DIR)
    else:
        POLYG_GDF = _to_lonlat(gpd.read_file(ARGS.polyg_path))
        logging.info('polygons: %d features from %s (crs: %s, columns: %s)',
                     len(POLYG_GDF), ARGS.polyg_path, POLYG_GDF.crs,
                     list(POLYG_GDF.columns))


GT_GDF = None
if ARGS.gt_polyg_path and os.path.exists(ARGS.gt_polyg_path):
    import geopandas as gpd
    GT_GDF = _to_lonlat(gpd.read_file(ARGS.gt_polyg_path))
    logging.info('GT polygons: %d features from %s', len(GT_GDF), ARGS.gt_polyg_path)
else:
    logging.warning('GT polygons file not found: %s', ARGS.gt_polyg_path)


def gt_geojson(key):
    """GT subsidence polygons matching this intf's date pair (as in patch prep)."""
    empty = '{"type":"FeatureCollection","features":[]}'
    if GT_GDF is None:
        return empty
    sd, ed = key.split('_')
    g = GT_GDF
    norm = lambda col: g[col].astype(str).str.replace('-', '', regex=False)
    sel = g[(norm('start_date') == sd) & (norm('end_date') == ed)]
    return sel.to_json() if len(sel) else empty


LIDAR_GDF = None
if ARGS.lidar_mask_path and os.path.exists(ARGS.lidar_mask_path):
    import geopandas as gpd
    LIDAR_GDF = _to_lonlat(gpd.read_file(ARGS.lidar_mask_path))
    logging.info('LiDAR mask polygons: %d features from %s (sources: %s)',
                 len(LIDAR_GDF), ARGS.lidar_mask_path,
                 sorted(LIDAR_GDF['source'].dropna().unique()) if 'source' in LIDAR_GDF.columns else 'n/a')
else:
    logging.warning('LiDAR mask file not found: %s', ARGS.lidar_mask_path)


def lidar_mask_geojson(key):
    """LiDAR valid-coverage polygon(s) for this intf's source survey, matching
    the case-insensitive 'source' lookup in test_full_intf.py."""
    empty = '{"type":"FeatureCollection","features":[]}'
    if LIDAR_GDF is None or 'source' not in LIDAR_GDF.columns:
        return empty
    src = COORD.get(key, {}).get('lidar_mask')
    if src is None or str(src).strip().lower() in ('', 'none', 'null'):
        sel = LIDAR_GDF
    else:
        col = LIDAR_GDF['source'].astype(str).str.strip().str.lower()
        sel = LIDAR_GDF[col == str(src).strip().lower()]
        if sel.empty:
            sel = LIDAR_GDF
    return sel.to_json() if len(sel) else empty


def polyg_lon_correction(key):
    """Legacy shapefiles were written with plg_indx2longlat's x_start=x4000
    fallback, displaced a further fixed 480 px. Verified against the GT
    polygons: dlon = frame_origin - (x0 + 4000*dx) + 0.01333, dlat = 0
    (residual < 0.000125 deg over all 142 intfs of both frames)."""
    info = COORD.get(key)
    if not info:
        return 0.0
    fixed = 35.37 if info.get('frame') == 'North' else 35.32
    x4000 = info['east'] + 4000 * info['dx']
    return fixed - x4000 + 0.01333


def polygons_geojson(key):
    """GeoJSON FeatureCollection of the predicted polygons for one intf."""
    empty = '{"type":"FeatureCollection","features":[]}'
    sel = None
    if POLYG_GDF is not None:
        g = POLYG_GDF
        if 'intf_key' in g.columns:
            sel = g[g['intf_key'] == key]
        elif 'start_date' in g.columns and 'end_date' in g.columns:
            sd, ed = key.split('_')
            norm = lambda col: g[col].astype(str).str.replace('-', '', regex=False)
            sel = g[(norm('start_date') == sd) & (norm('end_date') == ed)]
        else:
            sel = g  # no date columns: show everything
    elif POLYG_DIR is not None:
        import geopandas as gpd
        hits = sorted(Path(POLYG_DIR).glob(f'*{key}*.shp'))
        if hits:
            sel = _to_lonlat(gpd.read_file(hits[0]))
    if sel is None or not len(sel):
        return empty
    if ARGS.legacy_polyg_correction:
        sel = sel.copy()
        sel.geometry = sel.geometry.translate(xoff=polyg_lon_correction(key))
    return sel.to_json()


# ----------------------------------------------------------------- rendering
os.makedirs(ARGS.cache_dir, exist_ok=True)
_render_lock = threading.Lock()
_rendering = set()


def intf_extent(key):
    info = COORD[key]
    x0, y0 = info['east'], info['north']
    dx, dy = info['dx'], info['dy']
    nlines, ncells = info['nlines'], info['ncells']
    # [south, west, north, east] for Leaflet bounds
    return [y0 - dy * nlines, x0, y0, x0 + dx * ncells]


def render_image(key, cmap_name):
    """Render one .unw to a cached image file; return its path."""
    out = os.path.join(ARGS.cache_dir, f'{key}_{cmap_name}.{ARGS.fmt}')
    if os.path.exists(out):
        return out
    with _render_lock:
        if key + cmap_name in _rendering:
            # someone else is rendering it; wait by polling
            pass
        _rendering.add(key + cmap_name)
    try:
        if os.path.exists(out):
            return out

        # fast path: a pre-rendered jpg avoids the .unw read + percentile
        # clip + colorize (the expensive part, ~8s for a full-size scene).
        # Only used for 'jet' since that's the source jpgs' assumed colormap;
        # other colormaps need the actual .unw data.
        jpg_fn = JPG_BY_KEY.get(key)
        if cmap_name == 'jet' and jpg_fn is not None:
            src = Image.open(os.path.join(ARGS.input_dir, jpg_fn))
            if src.mode != 'RGB':
                src = src.convert('RGB')
            tmp = out + '.tmp'
            if ARGS.fmt == 'jpg':
                src.save(tmp, format='JPEG', quality=87)
            else:
                src.save(tmp, format='PNG')
            os.replace(tmp, out)
            logging.info('used source jpg for %s (%s)', key, cmap_name)
            return out

        info = COORD[key]
        fn = UNW_BY_KEY.get(key)
        if fn is None:
            raise FileNotFoundError(f'no .unw file for {key}')
        data = np.fromfile(os.path.join(ARGS.input_dir, fn),
                           dtype=np.float32).reshape(info['nlines'], info['ncells'])
        if info.get('byte_order') == 'MSBFirst':
            data = data.byteswap().view(data.dtype.newbyteorder('<'))

        valid = data != 0
        if valid.any():
            lo, hi = np.percentile(data[valid], ARGS.clip_pct)
        else:
            lo, hi = 0.0, 1.0
        # quantize to 8-bit indices, then colorize through a 256-entry LUT —
        # avoids building huge float RGBA intermediates for large rasters
        idx = np.clip((data - lo) / max(hi - lo, 1e-12), 0, 1)
        idx = (idx * 255).astype(np.uint8)
        if cmap_name == 'gray':
            gray = np.where(valid, idx, 0).astype(np.uint8)
            img = Image.fromarray(gray, mode='L')
        else:
            lut = (get_cmap(cmap_name)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            rgb = lut[idx]
            rgb[~valid] = 0  # no-data black
            img = Image.fromarray(rgb, mode='RGB')
        tmp = out + '.tmp'
        if ARGS.fmt == 'jpg':
            img.save(tmp, format='JPEG', quality=87)
        else:
            img.save(tmp, format='PNG')
        os.replace(tmp, out)
        logging.info('rendered %s (%s)', key, cmap_name)
        return out
    finally:
        with _render_lock:
            _rendering.discard(key + cmap_name)


def prerender_neighbors(key, cmap_name):
    """Warm the cache for the previous/next intf of the same frame."""
    for frame, keys in SEQS.items():
        if key in keys:
            i = keys.index(key)
            for j in (i - 1, i + 1):
                if 0 <= j < len(keys):
                    k = keys[j]
                    threading.Thread(target=render_image, args=(k, cmap_name),
                                     daemon=True).start()
            return


# ----------------------------------------------------------------- web page
PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"><title>Interferogram Viewer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { margin:0; height:100%; }
  body { display:flex; flex-direction:column; }
  #map { flex:1 1 0; min-height:0; background:#111; }
  #bar { flex:0 0 auto; background:#1e1e1e;
         color:#ddd; font:13px/22px sans-serif; padding:6px 12px; box-sizing:border-box;
         display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; z-index:1000; }
  #bar b { color:#fff; }
  #bar button, #bar select { background:#333; color:#ddd; border:1px solid #555;
         border-radius:4px; padding:3px 10px; cursor:pointer; }
  #bar label { cursor:pointer; }
  .pix { image-rendering: pixelated; }
  #loading { color:#f90; display:none; }
  .gridlbl { color:#0ff; font:bold 16px monospace; text-shadow:0 0 4px #000, 0 0 4px #000, 0 0 4px #000;
             white-space:nowrap; pointer-events:none; }
</style>
</head>
<body>
<div id="bar">
  <span>
    <label><input type="radio" name="frame" value="North" checked> North</label>
    <label><input type="radio" name="frame" value="South"> South</label>
  </span>
  <button id="prev">&#8592; prev</button>
  <button id="next">next &#8594;</button>
  <select id="intfsel" style="max-width:230px"></select>
  <b id="title">-</b>
  <span id="pos"></span>
  <select id="cmap">
    <option value="jet" selected>jet</option>
    <option value="gray">gray</option>
    <option value="twilight">twilight</option>
    <option value="RdBu">RdBu</option>
  </select>
  <label><input type="checkbox" id="showpoly" checked> polygons (p)</label>
  <span id="npoly" style="color:#fff"></span>
  <label><input type="checkbox" id="showgt"> GT (t)</label>
  <span id="ngt" style="color:#fd0"></span>
  <label><input type="checkbox" id="showlidar"> LiDAR mask (m)</label>
  <label><input type="checkbox" id="showgrid" checked> grid (g)</label>
  <span id="cursorpos" style="color:#9cf; min-width:150px"></span>
  <button id="boxzoom">box zoom (z)</button>
  <button id="fit">fit (f)</button>
  <span id="loading">loading&#8230;</span>
  <span style="margin-left:auto;color:#888">&#8592;/&#8594; time &nbsp; n/s frame</span>
</div>
<div id="map"></div>
<script>
const map = L.map('map', {crs: L.CRS.Simple, minZoom: 6, maxZoom: 19,
                          zoomSnap: 0.25, zoomDelta: 0.5});
let seqs = null, frame = 'North', idx = 0;
let overlay = null, polyLayer = null, gtLayer = null, lidarLayer = null, firstLoad = true;

function key() { return seqs[frame][idx]; }

function fmtDate(d) { return d.slice(0,4)+'-'+d.slice(4,6)+'-'+d.slice(6,8); }

function fillIntfSel() {
  const sel = document.getElementById('intfsel');
  sel.innerHTML = '';
  seqs[frame].forEach((k, i) => {
    const o = document.createElement('option');
    o.value = i;
    o.textContent = fmtDate(k.slice(0,8)) + ' \\u2192 ' + fmtDate(k.slice(9));
    sel.appendChild(o);
  });
  sel.value = idx;
}

async function load(fit) {
  const k = key();
  document.getElementById('loading').style.display = 'inline';
  const meta = await (await fetch('/api/meta/' + k)).json();
  const b = [[meta.extent[0], meta.extent[1]], [meta.extent[2], meta.extent[3]]];
  const cmap = document.getElementById('cmap').value;
  const url = '/api/image/' + k + '?cmap=' + cmap;

  const img = new window.Image();
  img.onload = () => {
    if (overlay) map.removeLayer(overlay);
    overlay = L.imageOverlay(url, b, {className:'pix'}).addTo(overlay ? map : map);
    if (fit || firstLoad) { map.fitBounds(b); firstLoad = false; }
    document.getElementById('loading').style.display = 'none';
  };
  img.src = url;

  const [s, e] = k.split('_');
  document.getElementById('title').textContent =
      fmtDate(s) + ' \\u2192 ' + fmtDate(e) + '  (' + frame + ')';
  document.getElementById('pos').textContent =
      (idx+1) + '/' + seqs[frame].length;
  document.getElementById('intfsel').value = idx;

  if (polyLayer) { map.removeLayer(polyLayer); polyLayer = null; }
  const gj = await (await fetch('/api/polygs/' + k)).json();
  polyLayer = L.geoJSON(gj, {style: {color:'#ffffff', weight:2,
                                     fillColor:'#ffffff', fillOpacity:0.25}});
  const n = gj.features ? gj.features.length : 0;
  document.getElementById('npoly').textContent = n ? n + ' polygs' : 'no polygs';
  if (document.getElementById('showpoly').checked) polyLayer.addTo(map);

  if (gtLayer) { map.removeLayer(gtLayer); gtLayer = null; }
  const gt = await (await fetch('/api/gt/' + k)).json();
  gtLayer = L.geoJSON(gt, {style: {color:'#ffdd00', weight:2,
                                   fillColor:'#ffdd00', fillOpacity:0.15}});
  const ng = gt.features ? gt.features.length : 0;
  document.getElementById('ngt').textContent = ng ? ng + ' GT' : 'no GT';
  if (document.getElementById('showgt').checked) gtLayer.addTo(map);

  if (lidarLayer) { map.removeLayer(lidarLayer); lidarLayer = null; }
  const lm = await (await fetch('/api/lidarmask/' + k)).json();
  lidarLayer = L.geoJSON(lm, {style: {color:'#ffff00', weight:5, opacity:1,
                                      dashArray:'10 6', fill:false, lineCap:'butt'}});
  if (document.getElementById('showlidar').checked) lidarLayer.addTo(map);
}

function step(d) {
  idx = Math.min(Math.max(idx + d, 0), seqs[frame].length - 1);
  load(false);
}

function switchFrame(f) {
  if (f === frame || !seqs[f].length) return;
  const cur = key().slice(0, 8);
  frame = f;
  // jump to the nearest start date in the other frame
  let best = 0, bestDiff = Infinity;
  seqs[frame].forEach((k, i) => {
    const diff = Math.abs(parseInt(k.slice(0,8)) - parseInt(cur));
    if (diff < bestDiff) { bestDiff = diff; best = i; }
  });
  idx = best;
  document.querySelector('input[name=frame][value='+f+']').checked = true;
  fillIntfSel();
  load(true);
}

document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
document.getElementById('fit').onclick = () => load(true);
document.getElementById('intfsel').onchange = (e) => {
  idx = parseInt(e.target.value); load(false); e.target.blur();
};

// --- box zoom: click the button (or press z), then drag a rectangle ---
let boxMode = false, boxStart = null, boxRect = null;
const bzBtn = document.getElementById('boxzoom');
function setBoxMode(on) {
  boxMode = on;
  bzBtn.style.background = on ? '#c60' : '#333';
  document.getElementById('map').style.cursor = on ? 'crosshair' : '';
  on ? map.dragging.disable() : map.dragging.enable();
}
bzBtn.onclick = () => setBoxMode(!boxMode);
map.on('mousedown', (e) => {
  if (!boxMode) return;
  boxStart = e.latlng;
  boxRect = L.rectangle([boxStart, boxStart],
                        {color:'#ffa500', weight:1.5, fill:false, dashArray:'4'}).addTo(map);
});
map.on('mousemove', (e) => {
  if (boxMode && boxStart && boxRect) boxRect.setBounds(L.latLngBounds(boxStart, e.latlng));
});
map.on('mouseup', (e) => {
  if (!boxMode || !boxStart) return;
  const b = L.latLngBounds(boxStart, e.latlng);
  map.removeLayer(boxRect); boxRect = null; boxStart = null;
  setBoxMode(false);
  if (b.isValid()) map.fitBounds(b);
});
document.getElementById('cmap').onchange = () => load(false);
document.getElementById('showpoly').onchange = (e) => {
  if (!polyLayer) return;
  e.target.checked ? polyLayer.addTo(map) : map.removeLayer(polyLayer);
};
document.getElementById('showgt').onchange = (e) => {
  if (!gtLayer) return;
  e.target.checked ? gtLayer.addTo(map) : map.removeLayer(gtLayer);
};
document.getElementById('showlidar').onchange = (e) => {
  if (!lidarLayer) return;
  e.target.checked ? lidarLayer.addTo(map) : map.removeLayer(lidarLayer);
};
document.querySelectorAll('input[name=frame]').forEach(r =>
  r.onchange = () => switchFrame(r.value));
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') step(-1);
  else if (e.key === 'ArrowRight') step(1);
  else if (e.key === 'n' || e.key === 'N') switchFrame('North');
  else if (e.key === 's' || e.key === 'S') switchFrame('South');
  else if (e.key === 'f' || e.key === 'F') load(true);
  else if (e.key === 'z' || e.key === 'Z') setBoxMode(!boxMode);
  else if (e.key === 'g' || e.key === 'G') {
    const c = document.getElementById('showgrid');
    c.checked = !c.checked; drawGrid();
  }
  else if (e.key === 't' || e.key === 'T') {
    const c = document.getElementById('showgt');
    c.checked = !c.checked; c.dispatchEvent(new Event('change'));
  }
  else if (e.key === 'm' || e.key === 'M') {
    const c = document.getElementById('showlidar');
    c.checked = !c.checked; c.dispatchEvent(new Event('change'));
  }
  else if (e.key === 'p' || e.key === 'P') {
    const c = document.getElementById('showpoly');
    c.checked = !c.checked; c.dispatchEvent(new Event('change'));
  }
});

// --- lon/lat graticule -------------------------------------------------
const gridLayer = L.layerGroup().addTo(map);
function niceStep(span) {
  const target = span / 12;
  const p = Math.pow(10, Math.floor(Math.log10(target)));
  for (const m of [1, 2, 5, 10]) { if (m * p >= target) return m * p; }
  return 10 * p;
}
function drawGrid() {
  gridLayer.clearLayers();
  if (!document.getElementById('showgrid').checked) return;
  const b = map.getBounds();
  const step = niceStep(Math.max(b.getNorth() - b.getSouth(), b.getEast() - b.getWest()));
  const dec = Math.max(0, Math.ceil(-Math.log10(step)) + 1);
  const style = {color:'#00e5ff', weight:0.8, opacity:0.55, interactive:false};
  for (let x = Math.ceil(b.getWest() / step) * step; x <= b.getEast(); x += step) {
    L.polyline([[b.getSouth(), x], [b.getNorth(), x]], style).addTo(gridLayer);
    L.marker([b.getSouth(), x], {interactive:false, icon: L.divIcon({className:'gridlbl',
      html: x.toFixed(dec), iconAnchor: [-4, 22]})}).addTo(gridLayer);
  }
  for (let y = Math.ceil(b.getSouth() / step) * step; y <= b.getNorth(); y += step) {
    L.polyline([[y, b.getWest()], [y, b.getEast()]], style).addTo(gridLayer);
    L.marker([y, b.getWest()], {interactive:false, icon: L.divIcon({className:'gridlbl',
      html: y.toFixed(dec), iconAnchor: [-5, 20]})}).addTo(gridLayer);
  }
}
map.on('moveend zoomend', drawGrid);
document.getElementById('showgrid').onchange = drawGrid;
map.on('mousemove', (e) => {
  document.getElementById('cursorpos').textContent =
      e.latlng.lng.toFixed(5) + ', ' + e.latlng.lat.toFixed(5);
});
window.addEventListener('resize', () => map.invalidateSize());

fetch('/api/intfs').then(r => r.json()).then(d => {
  seqs = d;
  if (!seqs[frame].length) frame = 'South';
  fillIntfSel();
  load(true);
});
</script>
</body>
</html>
"""


# ----------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            url = urlparse(self.path)
            parts = url.path.strip('/').split('/')
            if url.path == '/':
                self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
            elif url.path == '/api/intfs':
                self._send(200, json.dumps(SEQS).encode(), 'application/json')
            elif parts[:2] == ['api', 'meta'] and len(parts) == 3:
                body = json.dumps({'extent': intf_extent(parts[2])}).encode()
                self._send(200, body, 'application/json')
            elif parts[:2] == ['api', 'polygs'] and len(parts) == 3:
                self._send(200, polygons_geojson(parts[2]).encode(), 'application/json')
            elif parts[:2] == ['api', 'gt'] and len(parts) == 3:
                self._send(200, gt_geojson(parts[2]).encode(), 'application/json')
            elif parts[:2] == ['api', 'lidarmask'] and len(parts) == 3:
                self._send(200, lidar_mask_geojson(parts[2]).encode(), 'application/json')
            elif parts[:2] == ['api', 'image'] and len(parts) == 3:
                cmap = parse_qs(url.query).get('cmap', ['gray'])[0]
                path = render_image(parts[2], cmap)
                prerender_neighbors(parts[2], cmap)
                st = os.stat(path)
                etag = f'"{st.st_mtime_ns}-{st.st_size}"'
                if self.headers.get('If-None-Match') == etag:
                    self.send_response(304)
                    self.send_header('ETag', etag)
                    self.end_headers()
                    return
                with open(path, 'rb') as f:
                    data = f.read()
                ctype = 'image/jpeg' if ARGS.fmt == 'jpg' else 'image/png'
                self._send(200, data, ctype,
                           {'ETag': etag, 'Cache-Control': 'public, max-age=86400'})
            else:
                self._send(404, b'not found', 'text/plain')
        except BrokenPipeError:
            pass
        except Exception as e:
            logging.exception('error serving %s', self.path)
            self._send(500, str(e).encode(), 'text/plain')

    def log_message(self, fmt, *args):
        pass  # keep the console quiet; we log renders ourselves


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', ARGS.port), Handler)
    logging.info('viewer running: http://localhost:%d', ARGS.port)
    logging.info('from your laptop:  ssh -L %d:localhost:%d <cluster-host>',
                 ARGS.port, ARGS.port)
    server.serve_forever()
