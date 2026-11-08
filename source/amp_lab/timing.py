"""Change named qpos playback speed before recomputing control-rate FK and velocities."""

import json
from pathlib import Path

import numpy as np

from .motion import sha256


def retime_qpos(source, output, *, speed):
    """A speed below one lengthens the motion; every source pose stays unchanged."""
    source, output = Path(source), Path(output)
    if output.exists() or output.with_suffix(".retarget.json").exists():
        raise FileExistsError(output)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    parent = json.loads(source.with_suffix(".retarget.json").read_text())
    if parent.get("qpos_sha256") != sha256(source):
        raise ValueError("Source retarget provenance mismatch")
    with np.load(source, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    fps = float(motion["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Source fps must be finite and positive")
    motion["fps"] = np.float64(fps * speed)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **motion)
    report = dict(parent, method="uniform_time_scaling_v1", parent_retarget=parent,
                  status="candidate_requires_visual_and_contact_review",
                  input_qpos_sha256=sha256(source), qpos_sha256=sha256(output), fps=fps * speed,
                  timing={"speed": speed, "source_fps": fps, "poses_modified": False},
                  contact_consistency_validated=False, dynamics_validated=False)
    output.with_suffix(".retarget.json").write_text(json.dumps(report, indent=2))
    return report
