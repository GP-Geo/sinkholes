#!/usr/bin/env bash
#BSUB -J consolidate_pinkas
#BSUB -o /home/labs/rudich/pinkas/sinkholes/logs/consolidate_pinkas_%J.out
#BSUB -e /home/labs/rudich/pinkas/sinkholes/logs/consolidate_pinkas_%J.err
#BSUB -q short
#BSUB -R rusage[mem=2GB]
#BSUB -W 1:00
#
# One-off (2026-10-15): bring every sinkholes directory under pinkas/ into
# pinkas/sinkholes, so the project is one directory and one git worktree.
#
#   sinkholes_studies/k10t2026_hpo_v1             -> sinkholes/studies/
#   sinkholes_studies/engine_check_20261014_0942  -> sinkholes/studies/
#   scenes_11day, scenes_all (scene symlink farms) -> sinkholes/data/
#   python_envs/sinkholes                         deleted: the July venv, superseded by
#                                                 .conda/envs/sinkholes and referenced nowhere
#
# Run on WEXAC, not from the Mac: the scene folders hold server-side symlinks the
# Mac's SMB client cannot see, and Finder handles block directory renames there.
# Re-runnable: anything already moved is skipped, nothing is overwritten, and a
# move that leaves more broken symlinks than it found is undone.
#
# Submit (repo root):  bsub < scripts/consolidate_pinkas.sh
# The repo already points at the new paths (scripts/data/*, assets/hpo/*.json);
# the study ledger is rewritten afterwards, from the Mac.
set -euo pipefail
P=/home/labs/rudich/pinkas
R=$P/sinkholes

broken() { find -L "$1" -type l 2>/dev/null | wc -l; }   # -L: only dangling links stay type l

move() {  # move SRC into DEST_DIR
  local src=$1 dst="$2/$(basename "$1")"
  if [ ! -e "$src" ]; then echo "skip    $src (absent)"; return; fi
  if [ -e "$dst" ]; then echo "REFUSE  $src: $dst already exists" >&2; exit 1; fi
  local n links before after
  n=$(find "$src" | wc -l); links=$(find "$src" -type l | wc -l); before=$(broken "$src")
  mkdir -p "$2"
  mv "$src" "$dst"
  after=$(broken "$dst")
  if [ "$after" -ne "$before" ]; then
    mv "$dst" "$src"
    echo "UNDONE  $src: dangling symlinks $before -> $after after the move" >&2; exit 1
  fi
  echo "moved   $src -> $dst  ($n entries, $links symlinks, $before dangling before and after)"
}

move "$P/sinkholes_studies/k10t2026_hpo_v1" "$R/studies"
move "$P/sinkholes_studies/engine_check_20261014_0942" "$R/studies"
move "$P/scenes_11day" "$R/data"
move "$P/scenes_all" "$R/data"

if [ -d "$P/python_envs/sinkholes" ]; then
  rm -rf "$P/python_envs/sinkholes"
  echo "deleted $P/python_envs/sinkholes"
fi
for d in "$P/python_envs" "$P/sinkholes_studies"; do
  [ -d "$d" ] || continue
  rmdir "$d" 2>/dev/null && echo "removed $d (empty)" || { echo "kept    $d (not empty):"; ls -la "$d"; }
done

echo; echo "pinkas/ now:"; ls -l "$P"
echo; echo "sinkholes/studies and sinkholes/data:"; ls -la "$R/studies" "$R/data"
