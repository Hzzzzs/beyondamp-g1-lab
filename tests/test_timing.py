import json

import numpy as np
import pytest
from amp_lab.motion import load_motion, qpos_to_motion, sha256
from amp_lab.timing import retime_qpos


def test_retiming_recomputes_slower_velocities_and_preserves_provenance(tmp_path, tiny_robot):
    time = np.arange(51) / 50
    root = np.column_stack([time, time * 0, time * 0 + .8])
    joints = np.column_stack([time * .1, time * .2])
    source, output = tmp_path / "qpos.npz", tmp_path / "slower.npz"
    np.savez(source, fps=50, root_pos=root, joint_pos=joints, joint_names=["a", "b"],
             root_quat_wxyz=np.tile([1, 0, 0, 0], (51, 1)))
    source.with_suffix(".retarget.json").write_text(json.dumps({
        "qpos_sha256": sha256(source), "source": "fixture", "status": "fixture",
        "license_note": "fixture", "dynamics_validated": True,
    }))
    report = retime_qpos(source, output, speed=.5)
    with np.load(output) as slowed:
        np.testing.assert_array_equal(slowed["root_pos"], root)
        np.testing.assert_array_equal(slowed["joint_pos"], joints)
        assert float(slowed["fps"]) == 25
    converted = tmp_path / "amp.npz"
    qpos_to_motion(output, tiny_robot, converted)
    motion = load_motion(converted)
    assert len(motion["joint_pos"]) == 101
    np.testing.assert_allclose(motion["joint_vel"], np.tile([.05, .1], (101, 1)), atol=1e-7)
    root_id = motion["body_names"].tolist().index("pelvis")
    np.testing.assert_allclose(motion["body_lin_vel_w"][:, root_id], np.tile([.5, 0, 0], (101, 1)), atol=1e-7)
    assert report["parent_retarget"]["qpos_sha256"] == sha256(source)
    assert report["dynamics_validated"] is False
    with pytest.raises(FileExistsError):
        retime_qpos(source, output, speed=.5)
    with pytest.raises(ValueError, match="speed"):
        retime_qpos(source, tmp_path / "invalid.npz", speed=0)
