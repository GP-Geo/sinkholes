#!/bin/bash
# Run a script under the Python bundled with QGIS (PyQGIS available).
# Override QGIS_APP for a non-default install; on Linux set QGIS_PYTHON instead.
QGIS_APP="${QGIS_APP:-/Applications/QGIS-LTR.app}"
if [ -n "$QGIS_PYTHON" ]; then exec "$QGIS_PYTHON" "$@"; fi
R="$QGIS_APP/Contents/Resources"
export PYTHONPATH="$R/python:$R/python/plugins${PYTHONPATH:+:$PYTHONPATH}"
export PROJ_DATA="$R/proj" PROJ_LIB="$R/proj" GDAL_DATA="$R/gdal"
export QGIS_PREFIX_PATH="$QGIS_APP/Contents/MacOS"
export QT_QPA_PLATFORM=offscreen
exec "$QGIS_APP/Contents/MacOS/bin/python3" "$@"
