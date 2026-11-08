from dataclasses import replace

import numpy as np
import pytest
from amp_lab.evaluation import tracking_metrics
from amp_lab.mujoco_backend import MujocoBackend
from amp_lab.runtime import ObservationBuilder, OnnxPolicy, rotate_inverse
from conftest import write_zero_onnx


def test_velocity_report_distinguishes_standing_from_tracking():
    command = [0.6, 0, 0.3]
    standing = tracking_metrics([[-0.6, 0]] * 3, [-0.3] * 3, command)
    np.testing.assert_allclose(standing["mean_velocity"], [0, 0, 0])
    assert standing["velocity_rmse"] == pytest.approx(0.6 / np.sqrt(2))
    tracking = tracking_metrics([[0, 0]] * 3, [0] * 3, command)
    np.testing.assert_allclose(tracking["mean_velocity"], command)
    assert tracking["velocity_rmse"] == 0
    assert tracking_metrics([], [], command)["mean_velocity"] is None


def test_body_frame_rotation():
    np.testing.assert_allclose(rotate_inverse([np.sqrt(0.5), 0, 0, np.sqrt(0.5)], [1, 0, 0]),
                               [0, -1, 0], atol=1e-7)
    with pytest.raises(ValueError):
        rotate_inverse([0, 0, 0, 0], [0, 0, 1])


def test_observation_clip_scale_history_reset(tiny_spec):
    builder = ObservationBuilder(replace(tiny_spec, history_length=2))
    state = dict(quat_wxyz=[1, 0, 0, 0], ang_vel_b=[200, 0, 0], joint_pos=[0.3, -0.1],
                 joint_vel=[200, -200], command=[0.5, 0, 0])
    first = builder.build(**state).reshape(2, -1)
    np.testing.assert_equal(first[0], 0)
    assert first[1, 0] == 20  # clip 100, then scale 0.2
    np.testing.assert_allclose(first[1, 9:11], [0.2, 0.1], atol=1e-6)
    np.testing.assert_allclose(first[1, 11:13], [5, -5])
    np.testing.assert_allclose(builder.action_target([20, -20]), [2.5, -3.8], atol=1e-6)
    second = builder.build(**state).reshape(2, -1)
    np.testing.assert_equal(second[0], first[1])
    np.testing.assert_equal(second[1, -2:], [12, -12])
    builder.reset()
    np.testing.assert_equal(builder.build(**state).reshape(2, -1)[0], 0)


def test_name_mapping_pd_and_invalid_action(tiny_robot, tiny_spec):
    backend = MujocoBackend(tiny_robot, tiny_spec)
    assert backend.qadr.tolist() == [8, 7]
    np.testing.assert_allclose(backend.state()["joint_pos"], [0.1, -0.2])
    assert backend.actuator_ids.tolist() == [1, 0]
    before = backend.data.time
    backend.step([1, -1])
    assert backend.data.time == pytest.approx(before + tiny_spec.control_dt)
    assert np.isfinite(backend.data.qvel).all()
    with pytest.raises(ValueError):
        backend.step([np.nan, 0])
    with pytest.raises(ValueError):
        backend.run(lambda obs: [0, 0], -1, np.zeros(3))


def test_onnx_shape_contract(tmp_path, tiny_spec):
    path = tmp_path / "fixture.onnx"
    write_zero_onnx(path, tiny_spec)
    policy = OnnxPolicy(path, tiny_spec)
    np.testing.assert_equal(policy(np.ones(tiny_spec.observation_dim, np.float32)), [0, 0])
    with pytest.raises(ValueError, match="input mismatch"):
        OnnxPolicy(path, replace(tiny_spec, history_length=2))
