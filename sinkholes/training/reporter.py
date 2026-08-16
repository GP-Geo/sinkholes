"""CLI training reporter: colored logging, a fixed-width per-epoch table, a
results.csv mirror, best-metric tracking, and run figures.

Standard library only, apart from an optional tqdm import (so log lines emitted
mid-epoch go through ``tqdm.write()`` instead of smearing the progress bar) and
an optional matplotlib import confined to the plotting helpers. Nothing here
imports torch.

Regenerate the learning-curve figure for a finished run without retraining::

    sinkholes curves outputs/<run_dir>
"""

from __future__ import annotations

import csv
import logging
import shutil
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Mapping, Optional

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


# -- logging ----------------------------------------------------------------------------

class _ColoredFormatter(logging.Formatter):
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
    """Write through tqdm.write() so an open progress bar is not smeared."""

    def emit(self, record: logging.LogRecord) -> None:
        if _tqdm is None:
            return super().emit(record)
        try:
            _tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:  # pragma: no cover — logging must never kill training
            self.handleError(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Console handler at `level`, optional rotating file handler. Idempotent."""
    logger = logging.getLogger(name)
    if getattr(logger, "_reporter_configured", False):
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = _TqdmStreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_ColoredFormatter(color=sys.stdout.isatty()))
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    logger._reporter_configured = True
    return logger


def quiet_root_console(level: int = logging.WARNING) -> None:
    """Raise the root logger's *console* handlers only; file handlers keep INFO,
    so the run's .log file stays complete while the console shows the table."""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
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


# -- per-epoch table + CSV --------------------------------------------------------------

class EpochTable:
    """Right-aligned fixed-width table, reprinting its header periodically."""

    def __init__(self, columns: Iterable[str], logger: logging.Logger,
                 width: int = 11, header_every: int = 25):
        self.columns = list(columns)
        self.logger = logger
        self.width = max(width, max(len(c) for c in self.columns) + 2)
        self.header_every = header_every
        self._rows = 0

    @staticmethod
    def _fmt(value) -> str:
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    def header(self) -> None:
        self.logger.info("".join(c.rjust(self.width) for c in self.columns))

    def row(self, values: Mapping) -> None:
        if self.header_every and self._rows % self.header_every == 0:
            self.header()
        self.logger.info("".join(self._fmt(values.get(c, "")).rjust(self.width) for c in self.columns))
        self._rows += 1


def _epoch_number(cell) -> Optional[int]:
    """The epoch an 'N' or 'N/total' cell refers to, or None if unreadable."""
    try:
        return int(str(cell).split("/")[0].strip())
    except (TypeError, ValueError):
        return None


class ResultsCSV:
    """Append-per-epoch CSV, flushed each row so a killed run keeps its history.

    ``keep_through`` makes it resumable: instead of truncating, the file keeps
    the rows for epochs 1..N and the run appends N+1 onwards. Rows beyond N —
    an epoch that was logged but whose checkpoint never landed, so it is about
    to be repeated — are dropped rather than duplicated, and the original is
    copied aside first so nothing is destroyed silently.
    """

    def __init__(self, path: Path, columns: Iterable[str],
                 keep_through: Optional[int] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = list(columns)
        self.kept = 0
        self.backup: Optional[Path] = None
        if keep_through is None:
            self._write_header()
        else:
            self._truncate_to(int(keep_through))

    def _write_header(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.columns)

    def _truncate_to(self, epoch: int) -> None:
        if not self.path.exists():
            self._write_header()
            return
        with self.path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header = rows[0] if rows else []
        body = rows[1:]
        if "epoch" in header:
            index = header.index("epoch")
            kept = [r for r in body
                    if len(r) > index and (_epoch_number(r[index]) or 0) <= epoch
                    and _epoch_number(r[index]) is not None]
        else:  # an unreadable file: keep nothing, but never delete it
            kept = []

        if header != self.columns or len(kept) != len(body):
            self.backup = self.path.with_name(
                f"{self.path.stem}.bak-{time.strftime('%Y%m%d-%H%M%S')}{self.path.suffix}")
            shutil.copy2(self.path, self.backup)

        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
            for row in kept:
                writer.writerow(self._realign(row, header))
        self.kept = len(kept)

    def _realign(self, row, header) -> list:
        """Re-order a row read under `header` into this writer's columns."""
        if header == self.columns:
            return row
        as_map = dict(zip(header, row))
        return [as_map.get(c, "") for c in self.columns]

    def append(self, values: Mapping) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([values.get(c, "") for c in self.columns])


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

    def state_dict(self) -> dict:
        return {
            "metric": self.metric,
            "mode": self.mode,
            "patience": self.patience,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "since": self._since,
        }

    def load_state_dict(self, state: Mapping) -> "BestTracker":
        """Restore the best value and the early-stopping counter.

        ``patience`` deliberately stays as configured now rather than as saved:
        it is a policy knob, and raising it on a resume should take effect.
        """
        self.best = float(state.get("best", self.best))
        self.best_epoch = int(state.get("best_epoch", self.best_epoch))
        self._since = int(state.get("since", self._since))
        return self


# -- run figures ------------------------------------------------------------------------

def _figure(width: float, height: float):
    """A matplotlib Figure on the Agg canvas, bypassing pyplot and any global
    backend entirely — safe on headless nodes and inside training runs."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), dpi=120)
    FigureCanvasAgg(fig)
    return fig


def read_results_csv(csv_path: Path) -> dict:
    """results.csv -> {column: [values]}, with epoch as ints and NaN as None."""
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
            out[col].append(None if value != value else value)
    return out


def plot_curves(
    csv_path: Path,
    out_png: Optional[Path] = None,
    best_epoch: Optional[int] = None,
    title: Optional[str] = None,
) -> Optional[Path]:
    """Render results.csv as a two-panel learning curve; return the PNG path.

    Left: train/val loss with the learning rate as a dashed step on a log twin
    axis. Right: every ``val/*`` metric. Returns None (without raising) when
    there is nothing to draw or matplotlib is unavailable.
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
        if col not in data:
            return [], []
        pairs = [(e, v) for e, v in zip(epochs, data[col]) if isinstance(v, (int, float))]
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
        ax_lr.step(x_lr, y_lr, where="post", ls="--", lw=1.1, color="#7f7f7f", alpha=0.8, label="lr")
        ax_lr.set_yscale("log")
        ax_lr.set_ylabel("lr", color="#7f7f7f")
        ax_lr.tick_params(axis="y", colors="#7f7f7f")

    metric_cols = [c for c in data if c.startswith("val/") and c != "val/loss" and series(c)[0]]
    for col in metric_cols:
        x, y = series(col)
        emphasis = col == "val/dice"
        ax_met.plot(x, y, marker="o", ms=3, lw=2.2 if emphasis else 1.2,
                    alpha=1.0 if emphasis else 0.75, zorder=3 if emphasis else 2, label=col)
    ax_met.set_xlabel("epoch")
    ax_met.set_ylabel("score")
    ax_met.set_ylim(0, 1)
    ax_met.set_title("validation metrics")
    ax_met.grid(alpha=0.3)

    if best_epoch is None:
        x_dice, dice = series("val/dice")
        if dice:
            best_epoch = x_dice[dice.index(max(dice))]
    if best_epoch is not None:
        for ax in (ax_loss, ax_met):
            ax.axvline(best_epoch, color="#2ca02c", ls=":", lw=1.4, zorder=1,
                       label=f"best epoch {best_epoch}")

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
    """Input / ground truth / predicted-probability grid for a few val patches.

    ``samples`` carries plain numpy arrays ``image``, ``gt`` and ``prob``, each
    (N, H, W), as produced by ``evaluate(..., samples_out=...)``. The
    probability map with the threshold contour makes a nearly-right boundary
    read differently from an absent one. Returns None rather than raising.
    """
    image, gt, prob = (samples.get(k) for k in ("image", "gt", "prob"))
    if image is None or gt is None or prob is None or len(image) == 0:
        return None

    n = len(image)
    h, w = image[0].shape[-2:]
    # Patches are tall and narrow (200x100): scale cells by aspect ratio.
    cell_w = max(1.6, 3.2 * (w / max(h, 1)))
    try:
        fig = _figure(cell_w * n + 1.0, 8.4)
    except ImportError:
        return None
    axes = fig.subplots(3, n, squeeze=False)

    for j in range(n):
        axes[0][j].imshow(image[j], cmap="gray")
        axes[1][j].imshow(gt[j], cmap="gray", vmin=0, vmax=1)
        im = axes[2][j].imshow(prob[j], cmap="magma", vmin=0, vmax=1)
        try:
            axes[2][j].contour(prob[j], levels=[prob_th], colors="#00ff9f", linewidths=1.0)
        except (ValueError, TypeError):
            pass  # a constant probability map has no contour
        axes[0][j].set_title(f"px+ {int(gt[j].sum())}", fontsize=8)
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
