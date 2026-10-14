"""Real-data engine check: legacy eval-scenes + eval-outputs vs eval-snapshots.

Scores the same validation scenes, over their whole AOI, with one existing
checkpoint through both engines and compares every per-scene recall and
precision at every RTh, and the confidence map itself. eval-snapshots runs at
the study's tile batch and at tile batch 1 (eval-scenes' exact path), so the
report separates "the engine reproduces the legacy pipeline" from "batching on
this GPU moves a vote".

Run by scripts/hpo/engine_check.sh on a WEXAC GPU node. Touches no test scene,
selects nothing: the model is an already-trained one and only the engines are
compared. Writes <out>/report.json and prints a verdict.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sinkholes.inference import outputs, scenes  # noqa: E402
from sinkholes.inference import snapshot_eval as SE  # noqa: E402
from sinkholes.meta import find_11day_sequences, load_coord_dict  # noqa: E402


def pick_scenes(partition, n):
    """The gappiest validation chains first (they exercise the frame ages), then the rest."""
    coord = load_coord_dict(None)
    val = json.loads(Path(partition).read_text())["val"]
    chains, ok = find_11day_sequences(coord, k_prev=10, restrict_to=val,
                                      require_current_nonz_gt0=False)
    ranked = sorted(ok, key=lambda s: (-chains[s]["offsets"][0], s))
    return ranked[:n], {s: chains[s]["offsets"] for s in ranked[:n]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--partition", default=str(REPO / "assets/partition_temporal_k10_test2026_fill_th350x200.json"))
    p.add_argument("--patches_dir", default="/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches")
    p.add_argument("--n_scenes", type=int, default=2)
    p.add_argument("--tile_batch", type=int, default=32)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chosen, offsets = pick_scenes(a.partition, a.n_scenes)
    print("scenes", chosen, "offsets", offsets, flush=True)
    timing = {}

    t0 = time.time()
    sp = argparse.ArgumentParser(); scenes.add_arguments(sp)
    scenes.main(sp.parse_args([
        "--model", a.model, "--input_patch_dir", a.patches_dir, "--patch_size", "200", "100",
        "--data_stride", "4", "--intf_source", "intf_list", "--intf_list", ",".join(chosen),
        "--k_prevs", "10", "--tattn_unet", "--recon_average", "vote", "--vote_threshold", "0.5",
        "--aoi_from_partition", a.partition, "--aoi_split", "val", "--min_positives", "150",
        "--save_confidence", "--output_dir", str(out / "legacy")]))
    eval_dir = next(q.parent for q in (out / "legacy").rglob("*_pred.npy"))
    op = argparse.ArgumentParser(); outputs.add_arguments(op)
    outputs.main(op.parse_args([
        "--path", str(eval_dir), "--patch_size", "200", "100", "--data_stride", "4",
        "--aoi_from_partition", a.partition, "--aoi_split", "val", "--rth",
        "--out_json", str(out / "olm.json")]))
    timing["legacy_s"] = round(time.time() - t0)

    for tb in sorted({a.tile_batch, 1}, reverse=True):
        t0 = time.time()
        ep = argparse.ArgumentParser(); SE.add_arguments(ep)
        SE.main(ep.parse_args([
            "--checkpoint", f"m={a.model}", "--partition", a.partition,
            "--patches_dir", a.patches_dir, "--k_prevs", "10", "--arch", "tattn_unet",
            "--intfs", ",".join(chosen), "--tile_batch", str(tb),
            "--out_dir", str(out / f"new_tb{tb}"), "--cache_dir", str(out / "cache"),
            "--save_maps"]))
        timing[f"new_tb{tb}_s"] = round(time.time() - t0)

    legacy = json.loads((out / "olm.json").read_text())["per_intf"]
    report = {"model": a.model, "scenes": chosen, "offsets": offsets, "timing": timing}
    verdict = []
    for tb in sorted({a.tile_batch, 1}, reverse=True):
        new = json.loads((out / f"new_tb{tb}" / "m.json").read_text())["per_scene"]
        rec = {}
        for intf in chosen:
            z = np.load(out / f"new_tb{tb}" / "maps" / "m" / f"{intf}.npz")
            Y0, Y1, X0, X1 = z["crop"]
            conf = SE.confidence_from_votes(z["votes"], z["counts"])
            lc = np.load(eval_dir / f"{intf}_pred.npy")[Y0:Y1, X0:X1]
            diffs = {t: {"recall": new[intf][t]["recall"] - legacy[intf][t]["recall"],
                         "precision": new[intf][t]["precision"] - legacy[intf][t]["precision"]}
                     for t in legacy[intf]}
            rec[intf] = {"confidence_identical": bool(np.array_equal(conf, lc)),
                         "pixels_differing": int((conf != lc).sum()),
                         "pixels_scored": int(conf.size), "tiles": int((z["counts"] > 0).sum()),
                         "max_abs_metric_diff": max(abs(v) for d in diffs.values() for v in d.values()),
                         "diffs": diffs}
        report[f"tile_batch_{tb}"] = rec
        same = all(r["confidence_identical"] for r in rec.values())
        worst = max(r["max_abs_metric_diff"] for r in rec.values())
        verdict.append(f"tile_batch {tb}: confidence {'IDENTICAL' if same else 'DIFFERS'}; "
                       f"largest |metric difference| {worst:.3g}")
    report["verdict"] = verdict
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("\n".join(verdict))
    print(f"timing: {timing}")
    print(f"report -> {out / 'report.json'}")


if __name__ == "__main__":
    main()
