import json
from pathlib import Path

import numpy as np
import pytest
from amp_lab.bundle import create_bundle, verify_bundle
from amp_lab.motion import align_motion, load_motion, qpos_to_motion
from amp_lab.paths import contained_path
from conftest import write_zero_onnx


def test_linear_motion_fk_velocity_and_alignment(tmp_path, tiny_robot):
    fps = 100
    times = np.arange(101) / fps
    source, output = tmp_path / "qpos.npz", tmp_path / "motion.npz"
    np.savez(source, fps=fps, root_pos=np.column_stack([times, times * 0, times * 0 + 0.8]),
             root_quat_wxyz=np.tile([1, 0, 0, 0], (len(times), 1)),
             joint_pos=np.column_stack([times * 0.1, times * 0.2]), joint_names=["b", "a"])
    qpos_to_motion(source, tiny_robot, output, target_fps=50)
    motion = load_motion(output)
    assert motion["joint_pos"].shape == (51, 2)
    np.testing.assert_allclose(motion["joint_vel"], np.tile([0.1, 0.2], (51, 1)), atol=1e-7)
    root = motion["body_names"].tolist().index("pelvis")
    np.testing.assert_allclose(motion["body_lin_vel_w"][:, root], np.tile([1, 0, 0], (51, 1)), atol=1e-7)
    aligned = tmp_path / "aligned.npz"
    align_motion(output, aligned, ["a", "b"], ["second", "pelvis"], required_fps=50)
    np.testing.assert_allclose(load_motion(aligned)["joint_pos"], motion["joint_pos"][:, ::-1])
    with pytest.raises(ValueError, match="fps"):
        align_motion(output, aligned, ["a", "b"], ["pelvis"], required_fps=25)
    with pytest.raises(ValueError, match="missing"):
        align_motion(output, aligned, ["missing"], ["pelvis"], required_fps=50)


def test_reference_initialization_uses_named_joints_and_root(tmp_path, tiny_robot, tiny_spec):
    pytest.importorskip("torch")
    from amp_lab.training.dataset import JointMotionDataset

    source, output = tmp_path / "qpos.npz", tmp_path / "motion.npz"
    np.savez(source, fps=50, root_pos=[[0, 0, .8], [.02, 0, .8], [.04, 0, .8]],
             root_quat_wxyz=np.tile([2**-.5, 0, 0, 2**-.5], (3, 1)),
             joint_pos=[[.3, -.2], [.4, -.3], [.5, -.4]], joint_names=["a", "b"])
    qpos_to_motion(source, tiny_robot, output, target_fps=50)
    dataset = JointMotionDataset([output], tiny_spec, "cpu")
    # Profile order is b, a; reference reset and discriminator must share it.
    np.testing.assert_allclose(dataset.reference_joint_pos[0], [-.2, .3])
    np.testing.assert_allclose(dataset.reference_joint_vel[0], [-5, 5])
    np.testing.assert_allclose(dataset.reference_root_states[:, 7:10], [[1, 0, 0]] * 2, atol=1e-7)
    assert dataset.reference_root_states.shape == (2, 13)
    with_velocity = JointMotionDataset([output], tiny_spec, "cpu", root_velocity=True)
    # World +X motion under a +90 degree yaw is body -Y motion.
    np.testing.assert_allclose(with_velocity.states[:, -3:], [[0, -1, 0]] * 2, atol=1e-7)
    unscaled = JointMotionDataset([output], tiny_spec, "cpu", root_velocity=True, joint_velocity_scale=1.0)
    np.testing.assert_allclose(unscaled.states[:, 2:4], [[-5, 5]] * 2)
    np.testing.assert_allclose(dataset.states[:, 2:4], [[-.25, .25]] * 2)


def test_bundle_detects_tampering_and_unindexed_files(tmp_path, tiny_robot, tiny_spec):
    policy = tmp_path / "fixture.onnx"
    write_zero_onnx(policy, tiny_spec)
    bundle = tmp_path / "bundle"
    create_bundle(policy, tiny_spec, tiny_robot.parent, bundle)
    assert verify_bundle(bundle).joint_names == ["b", "a"]
    assert json.loads((bundle / "metadata.json").read_text())["trained_policy"] is False
    (bundle / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="unindexed"):
        verify_bundle(bundle)
    (bundle / "extra.txt").unlink()
    (bundle / "policy.onnx").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum"):
        verify_bundle(bundle)
    with pytest.raises(FileExistsError):
        create_bundle(policy, tiny_spec, tiny_robot.parent, bundle)


@pytest.mark.parametrize("relative", ["../escape", "/etc/passwd"])
def test_bundle_path_cannot_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        contained_path(tmp_path, relative)


def test_symlink_cannot_escape(tmp_path):
    (tmp_path / "link").symlink_to(Path("/etc"), target_is_directory=True)
    with pytest.raises(ValueError):
        contained_path(tmp_path, "link/passwd")


def test_inspection_runs_from_bundle_without_upstream(tmp_path, tiny_robot, tiny_spec, monkeypatch):
    from amp_lab import paths
    from amp_lab.inspection import inspect_bundle

    policy = tmp_path / "fixture.onnx"
    write_zero_onnx(policy, tiny_spec)
    bundle = tmp_path / "bundle"
    create_bundle(policy, tiny_spec, tiny_robot.parent, bundle)
    # All assets needed for inspection must come from the bundle.
    monkeypatch.setattr(paths, "G1_XML", tmp_path / "absent-upstream.xml")
    report = inspect_bundle(bundle)
    assert report["onnx_inference_verified"] is True
    assert report["model_loaded"] is True
    assert report["metadata"]["trained_policy"] is False
    assert report["action_size"] == 2
