"""Explicit, offline joint filtering experiment with complete motion provenance."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .motion import sha256


def smooth_qpos(source, output, *, sigma_seconds=0.04):
    """Positive Gaussian weights preserve joint ranges; root poses are copied exactly.

    This is an offline noncausal filter, not a contact or dynamics projection.
    Recompute FK/velocities and audit the result before using it for training.
    """
    source, output = Path(source), Path(output)
    if output.exists() or output.with_suffix(".retarget.json").exists():
        raise FileExistsError(output)
    if not np.isfinite(sigma_seconds) or sigma_seconds <= 0:
        raise ValueError("sigma_seconds must be finite and positive")
    with np.load(source, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    q, fps = motion["joint_pos"], float(motion["fps"])
    if (q.ndim != 2 or len(q) < 2 or not np.isfinite(q).all()
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Expected finite joint positions and positive fps")
    original_report = source.with_suffix(".retarget.json")
    if not original_report.is_file():
        raise ValueError("Source .retarget.json is required to preserve data provenance")
    parent = json.loads(original_report.read_text())
    if parent.get("qpos_sha256") != sha256(source):
        raise ValueError("Source retarget provenance mismatch")
    motion["joint_pos"] = gaussian_filter1d(q.astype(float), sigma_seconds * fps, axis=0, mode="nearest")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **motion)
    report = dict(parent, method="offline_gaussian_joint_filter_v1",
                  status="candidate_requires_visual_and_contact_review", parent_retarget=parent,
                  input_qpos_sha256=sha256(source), qpos_sha256=sha256(output),
                  smoothing={"sigma_seconds": sigma_seconds, "boundary_mode": "nearest",
                             "noncausal": True, "root_trajectory_modified": False,
                             "maximum_joint_change_rad": float(np.abs(motion["joint_pos"] - q).max())},
                  contact_consistency_validated=False, dynamics_validated=False)
    output.with_suffix(".retarget.json").write_text(json.dumps(report, indent=2))
    return report
