"""
Portable CLI training reporter.

Combines the things that make a training run readable from the terminal:
  - colored, level-based console logging (plus a rotating file log)
  - an Ultralytics-style fixed-width per-epoch metrics table
  - a results.csv mirror for plotting after the fact
  - best-metric tracking with early-stopping patience
  - run artefacts: a curves.png learning-curve figure and per-epoch
    input/GT/prediction sample grids (EDIT 2026-07-27, CHANGELOG.md #5)

Standard library only, apart from an optional tqdm import used so that log
records emitted mid-epoch go through ``tqdm.write()`` instead of smearing the
progress bar, and an optional matplotlib import confined to the plotting
helpers in section 5. Nothing here imports torch, so the module stays reusable.

Regenerate the figure for a finished run without retraining::

    python train_reporter.py outputs/<run_dir>

See CHANGELOG.md #3 and #5.
"""

from __future__ import annotations

# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---

import csv
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Mapping, Optional

try:  # tqdm is already a dependency of the training script
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - keeps the module importable standalone
    _tqdm = None


# ---------------------------------------------------------------------------
# 1. Colored console + rotating file logger
# ---------------------------------------------------------------------------

class _ColoredFormatter(logging.Formatter):
    """Level-colored console format; plain timestamped format for files."""

    COLORS = {
        logging.DEBUG: "\x1b[38;5;244m",
        logging.INFO: "\x1b[38;5;39m",
        logging.WARNING: "\x1b[38;5;226m",
        logging.ERROR: "\x1b[38;5;196m",
        logging.CRITICAL: "\x1b[31;1m",
    }
    RESET = "\x1b[0m"

    def __init__(self, color: bool = True):
        super().__init__(datefmt="%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        if self.color:
            c = self.COLORS.get(record.levelno, "")
            fmt = f"{c}%(levelname)-8s{self.RESET} %(message)s"
        else:
            fmt = "%(asctime)s %(levelname)-8s %(message)s"
        return logging.Formatter(fmt, datefmt=self.datefmt).format(record)


class _TqdmStreamHandler(logging.StreamHandler):
    """Write through tqdm so a live progress bar is not smeared.

    tqdm draws on stderr and the logger writes to stdout; interleaving the two
    while a bar is open corrupts both. ``tqdm.write()`` parks the bar, emits the
    line, and redraws.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if _tqdm is None:
            return super().emit(record)
        try:
            _tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:  # pragma: no cover - never let logging kill training
            self.handleError(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Console handler at `level`, optional file handler at `file_level`.

    Idempotent: calling twice with the same name returns the configured logger
    instead of stacking (or silently dropping) handlers.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_reporter_configured", False):
        return logger

    logger.setLevel(logging.DEBUG)  # handlers do the filtering
    logger.propagate = False

    console = _TqdmStreamHandler(sys.stdout)
    console.setLevel(level)
    # Only emit ANSI codes to a real terminal, so piping to a file stays clean.
    console.setFormatter(_ColoredFormatter(color=sys.stdout.isatty()))
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setLevel(file_level)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    logger._reporter_configured = True
    return logger


def quiet_root_console(level: int = logging.WARNING) -> None:
    """Raise the level of the root logger's console handlers only.

    The training script configures the root logger with ``basicConfig`` plus a
    ``FileHandler``, so every ``logging.info`` in the repo lands on both. The
    per-step loss line makes that unreadable next to a metrics table. Raising
    only the stream handlers keeps the run's ``.log`` file byte-identical while
    handing the console to the reporter.
    """
    for handler in logging.getLogger().handlers:
        # FileHandler subclasses StreamHandler, so check it is *not* one first.
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(level)


def banner(logger: logging.Logger, title: str, width: int = 72, **fields) -> None:
    """Rule, title, rule, then aligned key/value lines for the resolved config."""
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)
    if fields:
        pad = max(len(k) for k in fields) + 2
        for key, value in fields.items():
            logger.info(f"  {key.replace('_', ' ').capitalize():<{pad}} {value}")


def human_time(seconds: float) -> str:
    """Compact duration: 42.1s, 3m12s, 1h04m."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


# ---------------------------------------------------------------------------
# 2. Per-epoch metrics table
# ---------------------------------------------------------------------------

class EpochTable:
    """Right-aligned fixed-width table, reprinting its header periodically.

    Columns stay put as the run scrolls, which is the whole reason the
    Ultralytics output is readable at a glance.
    """

    def __init__(
        self,
        columns: Iterable[str],
        logger: logging.Logger,
        width: int = 11,
        header_every: int = 25,
    ):
        self.columns = list(columns)
        self.logger = logger
        self.width = max(width, max(len(c) for c in self.columns) + 2)
        self.header_every = header_every
        self._rows = 0

    @staticmethod
    def _fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    def header(self) -> None:
        self.logger.info("".join(c.rjust(self.width) for c in self.columns))

    def row(self, values: Mapping) -> None:
        if self.header_every and self._rows % self.header_every == 0:
            self.header()
        self.logger.info(
            "".join(self._fmt(values.get(c, "")).rjust(self.width) for c in self.columns)
        )
        self._rows += 1


# ---------------------------------------------------------------------------
# 3. results.csv mirror
# ---------------------------------------------------------------------------

class ResultsCSV:
    """Append-per-epoch CSV, flushed each row so a killed run keeps its history."""

    def __init__(self, path: Path, columns: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = list(columns)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.columns)

    def append(self, values: Mapping) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([values.get(c, "") for c in self.columns])


# ---------------------------------------------------------------------------
# 4. Best-metric tracking + early stopping
# ---------------------------------------------------------------------------

class BestTracker:
    """Track the best value of one metric and count epochs since it improved."""

    def __init__(self, metric: str, mode: str = "max", patience: int = 0):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.best = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self._since = 0

    def update(self, epoch: int, values: Mapping) -> bool:
        """Return True when this epoch set a new best."""
        current = values[self.metric]
        improved = current > self.best if self.mode == "max" else current < self.best
        if improved:
            self.best, self.best_epoch, self._since = current, epoch, 0
        else:
            self._since += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return bool(self.patience) and self._since >= self.patience


# ---------------------------------------------------------------------------
# 5. Run artefacts — learning-curve figure and prediction sample grids
#    EDIT 2026-07-27: new section. CHANGELOG.md #5
# ---------------------------------------------------------------------------

def _figure(width: float, height: float):
    """A matplotlib Figure backed by the Agg canvas, bypassing pyplot entirely.

    ``evaluate.py:6`` sets ``plt.rcParams['backend'] = 'Qt5Agg'`` at import time
    (see CHANGELOG.md "Considered and rejected"), so anything that goes through
    pyplot afterwards tries to open a GUI window and dies on a machine without
    PyQt5 — every WEXAC compute node, and this laptop. Constructing ``Figure``
    and attaching ``FigureCanvasAgg`` by hand never consults the global backend,
    which makes these helpers safe to call from inside a training run.

    Raises ImportError when matplotlib is missing; callers treat plots as
    optional and keep training.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), dpi=120)
    FigureCanvasAgg(fig)
    return fig


def read_results_csv(csv_path: Path) -> dict:
    """results.csv → ``{column: [values]}``, with ``epoch`` as ints.

    ``epoch`` is written as ``"7/30"`` and ``time`` as ``"3m19s"``; the epoch
    number is parsed out and every other column is coerced to float where it
    can be, leaving non-numeric columns as strings. Missing cells and the
    literal ``nan`` become None so gaps break the line instead of plotting as 0.
    """
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    out: dict = {c: [] for c in rows[0]}
    for r in rows:
        for col, raw in r.items():
            if col == "epoch":
                out[col].append(int(str(raw).split("/")[0]))
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                out[col].append(raw or None)
                continue
            out[col].append(None if value != value else value)  # drop NaN
    return out


def plot_curves(
    csv_path: Path,
    out_png: Optional[Path] = None,
    best_epoch: Optional[int] = None,
    title: Optional[str] = None,
) -> Optional[Path]:
    """Render results.csv as a two-panel learning curve; return the PNG path.

    Left panel: train/val loss, with the learning rate as a dashed step on a
    log-scaled twin axis — that is where the ReduceLROnPlateau drops become
    visible against the loss they were meant to fix. Right panel: every
    ``val/*`` metric except the loss. The best epoch is marked in both.

    Returns None (without raising) when there is nothing to draw or matplotlib
    is unavailable, so a plotting problem can never kill a finished run.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    data = read_results_csv(csv_path)
    epochs = data.get("epoch")
    if not epochs:
        return None
    out_png = Path(out_png) if out_png else csv_path.with_name("curves.png")

    def series(col):
        """Epoch/value pairs for one column, skipping gaps."""
        if col not in data:
            return [], []
        pairs = [(e, v) for e, v in zip(epochs, data[col])
                 if isinstance(v, (int, float))]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    try:
        fig = _figure(13, 4.8)
    except ImportError:
        return None
    ax_loss, ax_met = fig.subplots(1, 2)

    for col, color in (("train/loss", "#1f77b4"), ("val/loss", "#d62728")):
        x, y = series(col)
        if x:
            ax_loss.plot(x, y, marker="o", ms=3, lw=1.6, color=color, label=col)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("loss")
    ax_loss.grid(alpha=0.3)

    x_lr, y_lr = series("lr")
    if x_lr:
        ax_lr = ax_loss.twinx()
        ax_lr.step(x_lr, y_lr, where="post", ls="--", lw=1.1,
                   color="#7f7f7f", alpha=0.8, label="lr")
        ax_lr.set_yscale("log")
        ax_lr.set_ylabel("lr", color="#7f7f7f")
        ax_lr.tick_params(axis="y", colors="#7f7f7f")

    metric_cols = [c for c in data
                   if c.startswith("val/") and c != "val/loss" and series(c)[0]]
    for col in metric_cols:
        x, y = series(col)
        emphasis = col == "val/dice"
        ax_met.plot(x, y, marker="o", ms=3,
                    lw=2.2 if emphasis else 1.2,
                    alpha=1.0 if emphasis else 0.75,
                    zorder=3 if emphasis else 2, label=col)
    ax_met.set_xlabel("epoch")
    ax_met.set_ylabel("score")
    ax_met.set_ylim(0, 1)
    ax_met.set_title("validation metrics")
    ax_met.grid(alpha=0.3)

    if best_epoch is None:
        _, dice = series("val/dice")
        x_dice, _ = series("val/dice")
        if dice:
            best_epoch = x_dice[dice.index(max(dice))]
    if best_epoch is not None:
        for ax in (ax_loss, ax_met):
            ax.axvline(best_epoch, color="#2ca02c", ls=":", lw=1.4,
                       zorder=1, label=f"best epoch {best_epoch}")

    for ax in (ax_loss, ax_met):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8, loc="best", framealpha=0.85)

    fig.suptitle(title or csv_path.parent.name, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    return out_png


def save_prediction_grid(
    samples: Mapping,
    out_png: Path,
    epoch: Optional[int] = None,
    prob_th: float = 0.5,
) -> Optional[Path]:
    """Save an input / ground-truth / prediction grid for a few val patches.

    ``samples`` carries plain numpy arrays — ``image``, ``gt`` and ``prob``,
    each ``(N, H, W)`` — as produced by ``evaluate(..., samples_out=...)``.
    One column per patch, three rows: the input interferogram, the ground-truth
    mask, and the predicted probability with the thresholded contour drawn over
    it, so a boundary that is nearly right reads differently from one that is
    absent.

    Returns None rather than raising when there is nothing to draw, matplotlib
    is missing, or contouring fails — a picture is never worth losing an epoch.
    """
    image, gt, prob = (samples.get(k) for k in ("image", "gt", "prob"))
    if image is None or gt is None or prob is None or len(image) == 0:
        return None

    n = len(image)
    h, w = image[0].shape[-2:]
    # Patches are tall and narrow (200x100), so scale the cells by their aspect
    # ratio instead of forcing squares, which would letterbox every panel.
    cell_w = max(1.6, 3.2 * (w / max(h, 1)))
    try:
        fig = _figure(cell_w * n + 1.0, 8.4)
    except ImportError:
        return None
    axes = fig.subplots(3, n, squeeze=False)

    for j in range(n):
        img_j, gt_j, prob_j = image[j], gt[j], prob[j]

        axes[0][j].imshow(img_j, cmap="gray")
        axes[1][j].imshow(gt_j, cmap="gray", vmin=0, vmax=1)
        im = axes[2][j].imshow(prob_j, cmap="magma", vmin=0, vmax=1)
        try:
            axes[2][j].contour(prob_j, levels=[prob_th], colors="#00ff9f",
                               linewidths=1.0)
        except (ValueError, TypeError):
            pass  # a constant probability map has no contour to draw

        axes[0][j].set_title(f"px+ {int(gt_j.sum())}", fontsize=8)
        for i in range(3):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

    for i, name in enumerate(("input", "ground truth", f"pred (th {prob_th})")):
        axes[i][0].set_ylabel(name, fontsize=9)

    fig.colorbar(im, ax=axes[2], fraction=0.035, pad=0.02)
    fig.suptitle(f"validation samples — epoch {epoch}" if epoch is not None
                 else "validation samples", fontsize=11)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    return out_png


# ---------------------------------------------------------------------------
# Usage — a plain PyTorch-shaped loop, but nothing here is torch-specific
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    import random

    # EDIT 2026-07-27: with a run directory argument, regenerate that run's
    # curves.png from its results.csv instead of running the self-test. Lets a
    # run that finished before this change (or one whose plot was lost) get its
    # figure without retraining. CHANGELOG.md #5
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
        png = plot_curves(run_dir / "results.csv", run_dir / "curves.png")
        print(f"wrote {png}" if png
              else f"nothing to plot: no usable {run_dir / 'results.csv'}")
        sys.exit(0 if png else 1)

    RUN_DIR = Path("runs/demo")
    logger = setup_logger(__name__, log_file=RUN_DIR / "logs" / "training.log")

    EPOCHS = 12
    banner(
        logger,
        "Reporter self-test",
        run_name="demo",
        device="mps",
        epochs=EPOCHS,
        batch_size=8,
        learning_rate=0.003,
    )

    # EDIT 2026-07-27: val/-prefixed metric names so the self-test also exercises
    # plot_curves()'s metric panel, which keys off that prefix. CHANGELOG.md #5
    COLUMNS = ["epoch", "time", "train/loss", "val/loss", "val/P", "val/R", "val/dice", "lr"]
    table = EpochTable(COLUMNS, logger, header_every=10)
    results = ResultsCSV(RUN_DIR / "results.csv", COLUMNS)
    tracker = BestTracker("val/dice", mode="max", patience=5)

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        decay = math.exp(-epoch / 4)
        row = {
            "epoch": f"{epoch}/{EPOCHS}",
            "time": round(time.time() - start, 1),
            "train/loss": 3.0 * decay + random.uniform(0, 0.1),
            "val/loss": 3.2 * decay + random.uniform(0, 0.2),
            "val/P": 1 - decay + random.uniform(-0.05, 0.05),
            "val/R": 1 - decay + random.uniform(-0.05, 0.05),
            "val/dice": 1 - decay + random.uniform(-0.05, 0.05),
            "lr": 1e-3 * (0.1 ** (epoch // 5)),
        }

        table.row(row)
        results.append(row)

        if tracker.update(epoch, row):
            logger.info(f"  new best val/dice={tracker.best:.4f} — saving best.pt")

        if tracker.should_stop:
            logger.warning(
                f"Early stopping: no improvement in {tracker.patience} epochs "
                f"(best val/dice={tracker.best:.4f} @ epoch {tracker.best_epoch})"
            )
            break

    curves = plot_curves(RUN_DIR / "results.csv", RUN_DIR / "curves.png",
                         best_epoch=tracker.best_epoch)  # EDIT 2026-07-27, CHANGELOG.md #5
    logger.info("=" * 72)
    logger.info(f"Done in {human_time(time.time() - start)} — results in {RUN_DIR}")
    if curves:
        logger.info(f"Curves    {curves}")
    logger.info("=" * 72)
