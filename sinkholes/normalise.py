"""Phase normalisation and the no-data convention.

One rule everywhere: data already in [0, 1] passes through with exact ``0``
remapped to ``0.5``, otherwise the values are wrapped-phase radians and map
through (phi + pi) / (2 pi). The ``0.5`` is the no-data code — after
normalisation, validity is recovered as ``|x - 0.5| > tol``.

Note the branch is decided per call, over whatever array the caller hands in:
training normalises per patch, full-scene inference per scene. That asymmetry
is inherited from the original pipeline and only matters for data that mixes
sub-[0,1] and radian values within one scene, which this dataset does not.
"""

import numpy as np

#: Tolerance for the "already in [0, 1]" test on the full-scene path.
SCENE_RANGE_TOL = 1e-3

#: Same test on the per-patch training path (looser, as in the original code).
PATCH_RANGE_TOL = 1e-1

#: A normalised pixel is no-data iff it equals 0.5 to within this tolerance.
VALIDITY_TOL = 1e-9


def normalise_phase(channel: np.ndarray, *, range_tol: float) -> np.ndarray:
    """Normalise one channel to [0, 1] with 0.5 as the no-data code.

    Returns a new array; the input is not modified.
    """
    mn = float(np.nanmin(channel))
    mx = float(np.nanmax(channel))
    if (mn < -range_tol) or (mx > 1.0 + range_tol):
        return (channel + np.pi) / (2 * np.pi)
    out = channel.copy()
    zeros = out == 0.0
    if zeros.any():
        out[zeros] = 0.5
    return out


def normalise_channels(stack: np.ndarray, *, range_tol: float, n_channels: int | None = None) -> np.ndarray:
    """Normalise the leading `n_channels` channels of `stack` (all when None).

    Channels at or after `n_channels` are validity maps and must stay strictly
    {0, 1} — running them through the 0 -> 0.5 remap would make the masked loss
    see no invalid pixels at all.
    """
    out = stack.copy()
    stop = out.shape[0] if n_channels is None else int(n_channels)
    for c in range(stop):
        out[c] = normalise_phase(out[c], range_tol=range_tol)
    return out


def validity_from_normalised(x: np.ndarray, tol: float = VALIDITY_TOL) -> np.ndarray:
    """1 where a normalised pixel carries data, 0 where it is the 0.5 no-data code."""
    return (np.abs(x - 0.5) > tol).astype(np.float32)
