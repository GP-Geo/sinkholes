"""Resource and storage estimates for every phase, from measured numbers.

MEASURED (sources in the comments) versus EXTRAPOLATED is stated per line; the
first real jobs replace the extrapolations (their LSF summaries print host
memory, the train log prints peak VRAM and s/epoch, eval-snapshots logs
seconds per scene and checkpoint).
"""

from __future__ import annotations

from typing import Dict, List

from ..training.schedules import parse_epoch_spec
from .spec import Spec

# -- training -------------------------------------------------------------------------------
#: plain k10 tattn on the filled partition, 32x4, 82,010 samples (LSF 261016,
#: temporal_k10_th350_fill_tattn_posw8_30e): 12m40s/epoch, dataset build 41 min,
#: 104.8 GB host peak, 11.5 GiB VRAM reserved.               MEASURED
PLAIN_EPOCH_MIN = 12.67
PLAIN_BUILD_MIN = 41
PLAIN_HOST_GB = 104.8
PLAIN_VRAM_GIB = 11.5
#: ctx50 k10 tattn, same architecture and optimiser, 32x4, generation 4
#: (54,328 samples; LSF 254183): 11m53s/epoch, build 46 min, 206.3 GB host,
#: 33.5 GiB VRAM reserved.                                    MEASURED
CTX_EPOCH_MIN_GEN4, CTX_SAMPLES_GEN4 = 11.88, 54328
CTX_BUILD_MIN_GEN4 = 46
CTX_HOST_GB_GEN4 = 206.3
CTX_VRAM_GIB = 33.5
FILL_SAMPLES = 82010
#: The study's partition (generation 5-t2026, k10): 43,944 train positives in
#: the window, ring negatives 1:1 -> ~87,888 samples (+7% on fill); 131 + 11
#: target scenes. Plain scales from the fill run, ctx50 from the gen-4 one;
#: host memory from resident pixels (samples + ~8 GB val + the gen-4 non-data
#: overhead).                                                    EXTRAPOLATED
STUDY_SAMPLES = 87888
PLAIN_EPOCH_MIN = PLAIN_EPOCH_MIN * STUDY_SAMPLES / FILL_SAMPLES          # ~13.6
PLAIN_BUILD_MIN = PLAIN_BUILD_MIN * 142 / 132                              # targets vs fill's
PLAIN_HOST_GB = round(PLAIN_HOST_GB * STUDY_SAMPLES / FILL_SAMPLES, 1)     # ~112
CTX_EPOCH_MIN = CTX_EPOCH_MIN_GEN4 * STUDY_SAMPLES / CTX_SAMPLES_GEN4     # ~19.2
CTX_BUILD_MIN = CTX_BUILD_MIN_GEN4 * 142 / 84                              # targets 131+11 vs 74+10
CTX_HOST_GB = STUDY_SAMPLES * 11 * 300 * 200 * 4 / 1e9 + 8.3 + (CTX_HOST_GB_GEN4 - 143.4 - 7.8)
#: the training-fit probe, 2048 samples at batch 1, ~ +5% of an epoch.  ESTIMATE
FIT_OVERHEAD = 1.05

# -- scene evaluation (eval-snapshots) ------------------------------------------------------
#: legacy eval-scenes, plain k10, 24 test scenes: 12,916 s, 75.1 GB (LSF 364705);
#: ctx50 k10, 17 scenes: 15,336 s, 224.3 GB (LSF 303197).        MEASURED (legacy engine)
LEGACY_PLAIN_S_PER_SCENE = 12916 / 24
LEGACY_CTX_S_PER_SCENE = 15336 / 17
#: the new engine reads only AOI tiles inside LiDAR (~40% of the bytes) once per
#: batch, and runs 16-32 tiles per forward pass instead of 1.     ESTIMATE, to be measured
EVAL_LOAD_S = {"plain": 90, "ctx50": 260}          # per scene, per job
EVAL_PREDICT_S = {"plain": 110, "ctx50": 330}      # per scene, per checkpoint (incl. metrics)
EVAL_HOST_GB = {"plain": 45, "ctx50": 125}         # largest (North) scene, gated tiles only
N_VAL_SCENES = 11

# -- storage --------------------------------------------------------------------------------
WEIGHTS_GB = 0.128          # best.pt of a k10 tattn run (128,462,557 bytes)   MEASURED
RESUME_GB = 0.39            # weights + AdamW moments + scheduler/RNG            ESTIMATE
MAPS_GB_PER_CKPT = 0.035    # 11 scenes of uint8 votes+counts, compressed        ESTIMATE


def train_hours(geometry: str, epochs: int, chunks: int = 1) -> float:
    per = (PLAIN_EPOCH_MIN if geometry == "plain" else CTX_EPOCH_MIN) * FIT_OVERHEAD
    build = PLAIN_BUILD_MIN if geometry == "plain" else CTX_BUILD_MIN
    return (epochs * per + chunks * build) / 60


def eval_hours(geometry: str, n_ckpt: int, n_jobs: int = 1, scenes: int = N_VAL_SCENES) -> float:
    return (n_jobs * scenes * EVAL_LOAD_S[geometry] + n_ckpt * scenes * EVAL_PREDICT_S[geometry]) / 3600


def chunks_for(spec: Spec, geometry: str, epochs: int) -> int:
    budget = float(spec.resources("train", geometry)["time_budget_hours"])
    per = (PLAIN_EPOCH_MIN if geometry == "plain" else CTX_EPOCH_MIN) * FIT_OVERHEAD / 60
    build = (PLAIN_BUILD_MIN if geometry == "plain" else CTX_BUILD_MIN) / 60
    per_chunk = max(1, int((budget - build - 0.1) / (per * 1.15)))
    return -(-epochs // per_chunk)


def estimate(spec: Spec) -> Dict:
    b, ev, L = spec.budget, spec.evaluation, spec.long
    n_init = len(parse_epoch_spec(ev["initial_epochs"]))
    n_prom = len(parse_epoch_spec(ev["promoted_epochs"]))
    n_seed = len(parse_epoch_spec(ev["seed_epochs"]))
    n_long = len(parse_epoch_spec(L["eval_epochs"]))
    e0, e1, eL = b["initial_epochs"], b["promoted_epochs"], L["epochs"]
    n_shared, n_tpe = len(spec.shared), int(spec.tpe["n_trials"])
    rows = []
    for g in spec.geometries:
        def row(phase, n_runs, epochs, n_ckpt, n_eval_jobs, note=""):
            c = chunks_for(spec, g, epochs)
            rows.append({
                "geometry": g, "phase": phase, "runs": n_runs, "epochs_each": epochs,
                "chunks_each": c,
                "train_gpu_h": round(n_runs * train_hours(g, epochs, c), 1),
                "eval_gpu_h": round(n_runs * eval_hours(g, n_ckpt, n_eval_jobs), 1),
                "note": note})
        row("baseline", 1, e0, n_init, 1)
        row("shared", n_shared - 1, e0, n_init, 1)
        row("tpe", n_tpe, e0, n_init, 1)
        row("promote", int(b["promote_top"]), e1 - e0, n_prom, 1, "continuation 30->60")
        row("seeds", len(b["repeat_seeds"]), e1, n_seed, 2)
        row("long (seed 42)", len(L["first_seeds"]), eL, n_long, -(-n_long // 3))
        row("long seeds", len(L["later_seeds"]), eL, n_long, -(-n_long // 3))
    tot = {"train_gpu_h": round(sum(r["train_gpu_h"] for r in rows), 1),
           "eval_gpu_h": round(sum(r["eval_gpu_h"] for r in rows), 1)}
    hpo_runs = (n_shared + n_tpe) * len(spec.geometries)
    storage = {
        "hpo_checkpoints_gb": round(hpo_runs * (3 * WEIGHTS_GB + 2 * RESUME_GB + n_init * WEIGHTS_GB), 1),
        "hpo_after_retention_gb": round(hpo_runs * (4 * WEIGHTS_GB + RESUME_GB), 1),
        "long_run_gb_each": round((n_long + 3) * WEIGHTS_GB + 2 * RESUME_GB, 1),
        "maps_gb_all_hpo": round(MAPS_GB_PER_CKPT * (hpo_runs * n_init + 4 * n_prom + 4 * n_seed), 1),
        "ctx50_stride4_tree_to_build_gb": "see `hpo coverage`",
    }
    memory = {
        "train_host_gb": {"plain": PLAIN_HOST_GB, "ctx50": round(CTX_HOST_GB)},
        "train_vram_gib": {"plain": PLAIN_VRAM_GIB, "ctx50": CTX_VRAM_GIB},
        "eval_host_gb": EVAL_HOST_GB,
        "train_epoch_min": {"plain": round(PLAIN_EPOCH_MIN * FIT_OVERHEAD, 1),
                            "ctx50": round(CTX_EPOCH_MIN * FIT_OVERHEAD, 1)},
        "legacy_eval_s_per_scene": {"plain": round(LEGACY_PLAIN_S_PER_SCENE),
                                    "ctx50": round(LEGACY_CTX_S_PER_SCENE)},
    }
    return {"rows": rows, "totals": tot, "storage": storage, "memory": memory}


def format_estimate(est: Dict) -> str:
    from .controller import format_table

    lines = [format_table(est["rows"]), "",
             f"totals: training {est['totals']['train_gpu_h']} GPU-h, "
             f"scene evaluation {est['totals']['eval_gpu_h']} GPU-h", ""]
    lines += [f"memory: {k} = {v}" for k, v in est["memory"].items()]
    lines += [f"storage: {k} = {v}" for k, v in est["storage"].items()]
    return "\n".join(lines)
