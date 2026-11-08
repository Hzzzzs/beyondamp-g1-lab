"""Offline, height-inferred foot targets for flat walking; these are not measured contacts."""

import numpy as np


def walking_foot_targets(ankles, fps):
    """Freeze stance XY and bridge swing XY with zero velocity/acceleration at full contacts.

    Inputs and outputs are world metres. At clipped swing endpoints preserve source
    velocity, since contact outside the recorded clip is unknown. Ground clearance
    is tapered at handovers rather than abruptly reverting to the human target.
    Root pose and timing are untouched. Assumes flat walking without flight.
    """
    ankles = np.asarray(ankles, dtype=float)
    if (ankles.ndim != 3 or ankles.shape[1:] != (2, 3) or len(ankles) < 2
            or not np.isfinite(ankles).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Expected finite [frames,2,3] ankles and positive fps")
    count = len(ankles)
    labels = np.empty(count, dtype=int)
    support = int(np.argmin(ankles[0, :, 2]))
    for t, pair in enumerate(ankles):
        candidate = int(np.argmin(pair[:, 2]))
        if pair[support, 2] > pair[candidate, 2] + 0.01:
            support = candidate
        labels[t] = support
    xy = ankles[:, :, :2].copy()
    clearance = np.maximum(0, ankles[:, :, 2] - ankles[:, :, 2].min(axis=1, keepdims=True))
    for foot in range(2):
        stance = labels == foot
        if not stance.any():
            continue  # No known contact on this foot: preserve the source trajectory.
        starts = np.flatnonzero(np.diff(np.r_[False, stance].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[stance, False].astype(int)) == -1)
        for start, end in zip(starts, ends):
            xy[start:end+1, foot] = ankles[start, foot, :2]
            clearance[start:end+1, foot] = 0
        # Swing runs, including fragments cut by clip boundaries.
        starts = np.flatnonzero(np.diff(np.r_[False, ~stance].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[~stance, False].astype(int)) == -1)
        for start, end in zip(starts, ends):
            left, right = max(0, start - 1), min(count - 1, end + 1)
            u = np.linspace(0, 1, right - left + 1)
            first, last = xy[left, foot].copy(), xy[right, foot].copy()
            if start > 0 and end < count - 1:
                # Quintic rest-to-rest bridge, with C2 continuity at both stance boundaries.
                blend = u**3 * (10 + u * (-15 + 6 * u))
                xy[left:right+1, foot] = first + blend[:, None] * (last - first)
            else:
                # A clipped endpoint is not assumed to be at rest.
                duration = (right - left) / fps
                v0 = (ankles[1, foot, :2] - ankles[0, foot, :2]) * fps if start == 0 else np.zeros(2)
                v1 = ((ankles[-1, foot, :2] - ankles[-2, foot, :2]) * fps
                      if end == count - 1 else np.zeros(2))
                xy[left:right+1, foot] = ((2*u**3 - 3*u**2 + 1)[:, None] * first
                                        + (u**3 - 2*u**2 + u)[:, None] * duration * v0
                                        + (-2*u**3 + 3*u**2)[:, None] * last
                                        + (u**3 - u**2)[:, None] * duration * v1)
            # Smooth the first/last 80ms of a full swing; keep its source clearance at the apex.
            envelope = np.ones(len(u))
            if start > 0:
                edge = np.clip(np.arange(len(u)) / max(1, 0.08 * fps), 0, 1)
                envelope *= edge**2 * (3 - 2 * edge)
            if end < count - 1:
                edge = np.clip(np.arange(len(u))[::-1] / max(1, 0.08 * fps), 0, 1)
                envelope *= edge**2 * (3 - 2 * edge)
            clearance[left:right+1, foot] *= envelope
    return xy, clearance, labels
