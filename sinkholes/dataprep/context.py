"""Context-window geometry: one place that decides what a large-context patch is.

A context tree is the **same grid** as its 200x100 parent, with a margin of
context grown around every cell. Cell (i, j) of a `ctx50` tree is cell (i, j) of
the plain tree padded by 50 px on all four sides, so:

* grid coordinates, ``nonz_indices.json``, ``count-positives`` and every
  partition keep their meaning exactly -- nothing is re-derived;
* the **mask** tree is not regenerated at all. Targets stay 200x100, so sample
  selection by "is the centre positive" is true by construction rather than by
  a cropping convention someone has to remember;
* scene reconstruction stamps the 200x100 prediction at the cell's original
  location, so the output canvas is bit-comparable with a plain evaluation.

The margin is what the network sees and never what it is scored on. Training
and inference must derive it from the same function or a checkpoint will be
evaluated on geometry it was not trained for -- the same doctrine as
``resolve_patch_dirs``.
"""

from typing import Optional, Tuple

import numpy as np

#: Value written into padded ground beyond the scene edge. Raw patches are
#: normalised later and 0 is the project's no-data code, which
#: ``normalise_phase`` maps to 0.5 and ``--treat_nodata_regions`` then marks
#: invalid -- so padding costs nothing extra and is already handled.
PAD_VALUE = 0.0


def context_size(predict_size: Tuple[int, int], margin: Tuple[int, int]) -> Tuple[int, int]:
    """(H, W) the network sees for a ``predict_size`` target with ``margin``."""
    (ph, pw), (my, mx) = predict_size, margin
    return ph + 2 * my, pw + 2 * mx


def context_margin(predict_size: Tuple[int, int], ctx: Tuple[int, int]) -> Tuple[int, int]:
    """Margin implied by a context size, validated to be symmetric.

    Asymmetric margins are rejected rather than rounded: a half-pixel offset
    between training and inference is invisible in every shape assertion and
    silently scores the model on ground it did not predict.
    """
    (ph, pw), (ch, cw) = predict_size, ctx
    dy, dx = ch - ph, cw - pw
    if dy < 0 or dx < 0:
        raise ValueError(f"context {ctx} is smaller than the prediction {predict_size}")
    if dy % 2 or dx % 2:
        raise ValueError(
            f"context {ctx} minus prediction {predict_size} = ({dy}, {dx}); both must be "
            "even so the target sits exactly in the centre"
        )
    return dy // 2, dx // 2


def centre_slices(ctx: Tuple[int, int], predict_size: Tuple[int, int]):
    """Slices cutting the centred ``predict_size`` region out of ``ctx``."""
    my, mx = context_margin(predict_size, ctx)
    return slice(my, my + predict_size[0]), slice(mx, mx + predict_size[1])


def context_window(
    scene: np.ndarray, i: int, j: int, predict_size: Tuple[int, int],
    stride: Tuple[int, int], margin: Tuple[int, int],
    *, offset: int = 0, pad_value: float = PAD_VALUE,
) -> np.ndarray:
    """The context window of grid cell (i, j), padded where it leaves the scene.

    Cell (i, j) targets ``rows [i*Sy, i*Sy+ph), cols [offset+j*Sx, offset+j*Sx+pw)``
    -- the same rectangle ``patchify`` cuts without a margin. The window is that
    rectangle grown by ``margin``, so its centre *is* the plain patch.
    """
    ph, pw = predict_size
    sy, sx = stride
    my, mx = margin
    r0, c0 = i * sy - my, offset + j * sx - mx
    r1, c1 = r0 + ph + 2 * my, c0 + pw + 2 * mx

    rr0, cc0 = max(r0, 0), max(c0, 0)
    rr1, cc1 = min(r1, scene.shape[0]), min(c1, scene.shape[1])
    out = np.full((r1 - r0, c1 - c0), pad_value, dtype=scene.dtype)
    if rr0 < rr1 and cc0 < cc1:
        out[rr0 - r0: rr1 - r0, cc0 - c0: cc1 - c0] = scene[rr0:rr1, cc0:cc1]
    return out


def assert_centre_matches(window: np.ndarray, plain: np.ndarray,
                          predict_size: Tuple[int, int], where: str = "") -> None:
    """Fail loudly unless the centre of ``window`` is exactly ``plain``.

    The invariant the whole design rests on. Cheap enough to keep on in
    ``prepare-patches``; the test suite runs it over real grids.
    """
    rs, cs = centre_slices(window.shape[-2:], predict_size)
    centre = window[..., rs, cs]
    if centre.shape != plain.shape:
        raise AssertionError(f"{where}: centre {centre.shape} != plain patch {plain.shape}")
    if not np.array_equal(np.nan_to_num(centre, nan=-9e9), np.nan_to_num(plain, nan=-9e9)):
        bad = int((centre != plain).sum())
        raise AssertionError(f"{where}: centre of the context window differs from the plain "
                             f"patch in {bad} of {centre.size} pixels")


def io_geometry(context: Optional[Tuple[int, int]], predict: Tuple[int, int]) -> dict:
    """The checkpoint record. Absent/equal sizes mean a plain 200x100 model."""
    return {"context": list(context or predict), "predict": list(predict)}
