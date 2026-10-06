"""``sinkholes relabel migrate`` -- add newer fields to an existing working GeoPackage.

Additive only: missing columns are appended (``ALTER TABLE ... ADD COLUMN``),
no row and no geometry is touched, and the file is copied into ``history/``
first. Run it with QGIS CLOSED -- a schema change under an open editing session
is how SQLite files on a network share get damaged.
"""

import logging
import shutil
import sqlite3
import subprocess
from datetime import datetime

from .common import QC_FIELDS, ws_path

#: column -> SQLite type, in the order they were introduced
FIELDS = {name: "TEXT" for name in QC_FIELDS}


def add_arguments(p):
    p.add_argument("--workspace", required=True)


def qgis_running() -> bool:
    try:
        return subprocess.run(["pgrep", "-f", "MacOS/QGIS$"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = ws_path(args.workspace, "working_gpkg")
    if qgis_running():
        raise SystemExit("QGIS is running: save your edits, close QGIS, then run migrate again")
    for side in (".gpkg-journal", ".gpkg-wal"):
        if path.with_name(path.name.replace(".gpkg", side)).exists():
            logging.warning(f"{side} file present next to {path.name}; SQLite will settle it on open")
    con = sqlite3.connect(path)
    have = {r[1] for r in con.execute("PRAGMA table_info(working_gt)")}
    missing = [f for f in FIELDS if f not in have]
    if not missing:
        logging.info("nothing to add: all fields present")
        return
    n_before = con.execute("SELECT count(*) FROM working_gt").fetchone()[0]
    con.close()
    backup = ws_path(args.workspace, "history") / f"gt_test_working_before_migrate_{datetime.now():%Y%m%d_%H%M%S}.gpkg"
    shutil.copy2(path, backup)
    con = sqlite3.connect(path)
    with con:
        for f in missing:
            con.execute(f'ALTER TABLE working_gt ADD COLUMN "{f}" {FIELDS[f]}')
        con.execute("UPDATE gpkg_contents SET last_change = strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                    "WHERE table_name = 'working_gt'")
    n_after = con.execute("SELECT count(*) FROM working_gt").fetchone()[0]
    ok = con.execute("PRAGMA integrity_check").fetchone()[0]
    con.close()
    if n_after != n_before or ok != "ok":
        raise SystemExit(f"migration check failed ({n_before} -> {n_after} rows, integrity {ok}); "
                         f"restore {backup}")
    logging.info(f"added {missing} to working_gt ({n_after} rows untouched); backup {backup.name}")
