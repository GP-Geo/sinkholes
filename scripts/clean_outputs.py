"""outputs/ clean-up of 2026-10-07. Dry run unless --apply.

Run from anywhere:  python3 scripts/clean_outputs.py <phases...> [--apply]

phases:
  prune   delete _image.npy/_pred_th.npy in every eval dir that has olm_results*.json
  ckpt    delete resume.pt/last.pt/interrupted.pt next to a best.pt in finished runs
  drop    delete _pred.npy/_gt.npy in *_valneg and *_pre2023_* evaluations (metrics stay)
  stub    delete the never-started fill_tattn_k5_ctx50_neg10 run
  hash    sha256 every remaining _gt.npy into a manifest (slow; resumable)
  dedup   move one copy per hash into predictions/_gt/, replace all copies by relative symlinks
"""
import argparse, hashlib, json, os, re, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Resolved from this file, so the same script works on the Mac mount
# (/Volumes/rudich/...) and on WEXAC (/home/labs/rudich/...).
OUT = Path(__file__).resolve().parents[1] / "outputs"
PRED = OUT / "predictions"
STORE = PRED / "_gt"
MANIFEST = PRED / "_gt_hashes.json"     # keys are paths relative to PRED
if not PRED.is_dir():
    sys.exit(f"{PRED} not found -- refusing to run (an empty walk would silently do nothing)")
KEEP_RESUME = {"geo_k5_single_ring3_valpos"}   # unfinished on purpose, README 2026-08-20

ap = argparse.ArgumentParser()
ap.add_argument("phases", nargs="+", choices="prune ckpt drop stub hash dedup".split())
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()
tot = {"n": 0, "b": 0}


def rm(p: Path, why=""):
    tot["n"] += 1; tot["b"] += p.lstat().st_size
    if args.apply:
        p.unlink()


def eval_dirs():
    for d, _, files in os.walk(PRED):
        if Path(d).is_relative_to(STORE):
            continue
        if any(f.endswith("_pred.npy") or f.endswith("_gt.npy") or f.startswith("olm_results") for f in files):
            yield Path(d), files


def finished(files):
    return any(f.startswith("olm_results") and f.endswith(".json") for f in files)


def report(label):
    print(f"[{label}] {'deleted' if args.apply else 'would delete'} {tot['n']} files, {tot['b']/1e9:.1f} GB")
    tot.update(n=0, b=0)


if "prune" in args.phases:
    for d, files in eval_dirs():
        if not finished(files):
            if any(f.endswith(("_image.npy", "_pred_th.npy")) for f in files):
                print(f"  SKIP unfinished: {d.relative_to(OUT)}")
            continue
        for f in files:
            if f.endswith(("_image.npy", "_pred_th.npy")):
                rm(d / f)
    report("prune")

if "ckpt" in args.phases:
    for ck in OUT.rglob("checkpoints"):
        if PRED in ck.parents or not (ck / "best.pt").exists():
            continue
        run = ck.parent
        top = run.name
        m = re.match(r"^(.+)_\d{4}-\d{2}-\d{2}_\d{2}h\d{2}_lsf_\d+$", top)
        if m and not (run / "logs/reporter.log").exists():
            continue
        if m and not re.search(r"Training (complete|interrupted)", (run / "logs/reporter.log").read_text(errors="replace")):
            print(f"  SKIP unfinished: {run.relative_to(OUT)}"); continue
        for f in ("resume.pt", "last.pt", "interrupted.pt"):
            if f == "resume.pt" and top in KEEP_RESUME:
                continue
            if (ck / f).exists():
                rm(ck / f)
    report("ckpt")

if "drop" in args.phases:
    for d, files in eval_dirs():
        top = d.relative_to(PRED).parts[0]
        if not (top.endswith("_valneg") or "_pre2023_" in top):
            continue
        if not finished(files):
            print(f"  SKIP unfinished: {d.relative_to(OUT)}"); continue
        for f in files:
            if f.endswith(("_pred.npy", "_gt.npy", "_image.npy", "_pred_th.npy")):
                rm(d / f)
    report("drop")

if "stub" in args.phases:
    for d in OUT.glob("fill_tattn_k5_ctx50_neg10_*_lsf_261015"):
        if (d / "checkpoints").exists() or (d / "results.csv").exists():
            sys.exit(f"{d} has results -- refusing")
        print(f"  {'removing' if args.apply else 'would remove'} {d.name}")
        if args.apply:
            shutil.rmtree(d)

if "hash" in args.phases:
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    todo = [str((d / f).relative_to(PRED)) for d, files in eval_dirs() for f in files
            if f.endswith("_gt.npy") and not (d / f).is_symlink()
            and str((d / f).relative_to(PRED)) not in man]
    print(f"[hash] {len(man)} already hashed, {len(todo)} to go")

    def h(p):
        s = hashlib.sha256()
        with open(PRED / p, "rb") as fh:
            while b := fh.read(16 << 20):
                s.update(b)
        return p, s.hexdigest()

    MANIFEST.write_text(json.dumps(man, indent=0))
    with ThreadPoolExecutor(4) as ex:
        for i, (p, dig) in enumerate(ex.map(h, todo), 1):
            man[p] = dig
            if i % 20 == 0 or i == len(todo):
                MANIFEST.write_text(json.dumps(man, indent=0)); print(f"  {i}/{len(todo)}", flush=True)

if "dedup" in args.phases:
    if args.apply and sys.platform != "linux":
        sys.exit("dedup --apply only on WEXAC: a symlink made over SMB from the Mac is not a server-side link")
    if not MANIFEST.exists():
        sys.exit(f"{MANIFEST} missing -- run the hash phase first")
    man = json.loads(MANIFEST.read_text())
    groups = {}
    for p, dig in man.items():
        p = PRED / p
        if p.exists() and not p.is_symlink():
            groups.setdefault(dig, []).append(p)
    if args.apply:
        STORE.mkdir(exist_ok=True)
    saved = links = 0
    for dig, paths in groups.items():
        intf = paths[0].name[: -len("_gt.npy")]
        target = STORE / f"{intf}_{dig[:12]}_gt.npy"
        size = paths[0].stat().st_size
        if any(p.stat().st_size != size for p in paths):
            sys.exit(f"hash collision with differing sizes: {paths}")
        if args.apply:
            if not target.exists():
                shutil.move(str(paths[0]), target)       # same volume: a rename
            else:
                paths[0].unlink()
            os.symlink(os.path.relpath(target, paths[0].parent), paths[0])
            for p in paths[1:]:
                p.unlink()
                os.symlink(os.path.relpath(target, p.parent), p)
        saved += size * (len(paths) - 1); links += len(paths)
    print(f"[dedup] {len(groups)} distinct GT arrays from {links} copies; "
          f"{'freed' if args.apply else 'would free'} {saved/1e9:.1f} GB")
