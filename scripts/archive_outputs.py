"""Move everything except the focus models into outputs/archive/. Dry run unless --apply.

Pure renames on one volume: instant, nothing deleted, reversible by moving back.
The tree under archive/ mirrors outputs/ (archive/<date>/, archive/predictions/, ...),
so a model and its evaluation still match by name.

RUN IT ON WEXAC, NOT THE MAC. Since clean_outputs.py dedup, every <intf>_gt.npy in an
evaluation is a RELATIVE symlink into predictions/_gt/. Moving an eval dir one level
deeper (archive/predictions/) breaks that relative path, so each moved link is re-pointed
here. The Mac's SMB client cannot do that: the server follows the links and the Mac sees
plain files (and a symlink made from the Mac is not one the server understands).
"""
import os, sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"
ARC = OUT / "archive"
STORE = OUT / "predictions" / "_gt"

FOCUS = {   # run name -> why it stays; outputs/README.md "Focus models"
    "temporal_k10_th350_fill_tattn_posw8_30e": "best model (gen 5, k10)",
    "temporal_k5_th350_fill_tattn_posw8_40e":  "best k5 (gen 5)",
    "temporal_k10_th350_tattn_posw8_30e":      "reference model; gen-4 twin of the k10 leader",
    "temporal_k5_th350_tattn_posw8_40e":       "gen-4 twin of the k5 leader",
    "temporal_k10_th350_single_posw8_30e":     "single-frame baseline",
}
KEEP_TOP = {"README.md", "archive", "predictions", "aoi_preview",
            "deck_2026-09-17", "deck_data_2026-09-16.json", "attention_probe"}
KEEP_PROBES = {"diagnostics", "ctx50_tattn_2026-09-16", "longgrid_tattn_plain_vs_ctx50_2026-09-16",
               "k10_tattn_plain_vs_ctx50_2026-09-17"}
RENAME_TOP = {"archive_2019_2026_noisy_data": "2019_2026_noisy_data"}

if not OUT.is_dir():
    sys.exit(f"{OUT} not found")
apply = "--apply" in sys.argv
if apply and sys.platform != "linux":
    sys.exit("--apply only on WEXAC: gt symlinks must be re-pointed server-side (see header)")


def relink(root: Path) -> int:
    """Re-point every gt symlink under root at the shared store; return how many."""
    n = 0
    for p in root.rglob("*_gt.npy"):
        if not p.is_symlink():
            continue
        target = STORE / Path(os.readlink(p)).name
        if not target.exists():
            sys.exit(f"{p}: store file {target} missing -- stopping")
        new = os.path.relpath(target, p.parent)
        if os.readlink(p) != new:
            p.unlink(); os.symlink(new, p); n += 1
    return n

moves = []

for d in sorted(OUT.iterdir()):
    if d.name in KEEP_TOP or d.name.startswith("."):
        continue
    if d.name[:4] == "2026" and d.is_dir():            # dated training folder
        for run in sorted(d.iterdir()):
            if run.name not in FOCUS:
                moves.append((run, ARC / d.name / run.name))
    elif "_lsf_" in d.name:
        print(f"  SKIP unfiled run {d.name} -- file it first (tidy_outputs.sh)")
    else:                                               # screens/, predictions_positives/, misc
        moves.append((d, ARC / RENAME_TOP.get(d.name, d.name)))

for d in sorted((OUT / "predictions").iterdir()):
    if d.is_dir() and d.name not in FOCUS and not d.name.startswith((".", "_")):   # _gt store, README stay
        moves.append((d, ARC / "predictions" / d.name))

for d in sorted((OUT / "attention_probe").iterdir()):
    if d.name not in KEEP_PROBES:
        moves.append((d, ARC / "attention_probe" / d.name))

missing = [f for f in FOCUS if not any((OUT / x / f).exists() for x in os.listdir(OUT) if x[:4] == "2026")]
if missing:
    sys.exit(f"focus run(s) not found: {missing}")
for s, t in moves:
    if t.exists():
        sys.exit(f"target exists: {t}")

for s, t in moves:
    print(f"{'mv' if apply else 'would mv'}  {s.relative_to(OUT)}  ->  {t.relative_to(OUT)}")
    if apply:
        t.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(s, t)
        except OSError as e:
            print(f"   FAILED ({e}) -- `lsof +D {s}` names the holder"); continue
        if (k := relink(t)):
            print(f"   re-pointed {k} gt links")
if apply:   # dated folders left empty
    for d in OUT.iterdir():
        if d.name[:4] == "2026" and d.is_dir() and not any(d.iterdir()):
            d.rmdir()
print(f"{len(moves)} moves")
if apply:
    bad = [p for p in OUT.rglob("*_gt.npy") if p.is_symlink() and not p.exists()]
    print(f"broken gt links after the move: {len(bad)}")
    for p in bad[:10]:
        print("   ", p)
