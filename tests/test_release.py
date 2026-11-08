import pytest
from amp_lab.evaluation import spec_fingerprint
from amp_lab.release import validate_evidence


def passing_report(spec, backend):
    return {"backend": backend, "policy_sha256": "test", "deployment_sha256": spec_fingerprint(spec),
            "requested_seconds": 20.0, "completed_seconds": 20.0, "completed": True, "fallen": False,
            "velocity_rmse": 0.1, "yaw_rmse": 0.1,
            "command": [0.6, 0, 0], "mean_velocity": [0.55, 0, 0]}


def test_release_needs_same_policy_and_two_simulators(tiny_spec):
    mujoco, isaac = passing_report(tiny_spec, "mujoco"), passing_report(tiny_spec, "isaaclab")
    validate_evidence([mujoco, isaac], tiny_spec, "test")
    with pytest.raises(ValueError, match="both simulators"):
        validate_evidence([mujoco], tiny_spec, "test")
    with pytest.raises(ValueError, match="different policy"):
        validate_evidence([mujoco, dict(isaac, policy_sha256="other")], tiny_spec, "test")


@pytest.mark.parametrize("changes", [{"fallen": True}, {"completed": False}, {"requested_seconds": 1},
                                      {"velocity_rmse": float("nan")}, {"velocity_rmse": 0.8}])
def test_release_refuses_failed_or_short_runs(tiny_spec, changes):
    with pytest.raises(ValueError, match="criteria"):
        validate_evidence([dict(passing_report(tiny_spec, "mujoco"), **changes)], tiny_spec, "test")


def test_release_does_not_accept_standing_at_a_slow_command(tiny_spec):
    # A stationary robot at 0.3 m/s has XY RMSE 0.212 < 0.3, so error alone is insufficient.
    report = dict(passing_report(tiny_spec, "mujoco"), command=[.3, 0, 0],
                  velocity_rmse=.2122, mean_velocity=[0, 0, 0])
    with pytest.raises(ValueError, match="measured forward motion"):
        validate_evidence([report], tiny_spec, "test")


@pytest.mark.parametrize("yaw_error", [1.1, float("nan"), None])
def test_release_rejects_turning_even_when_average_yaw_is_zero(tiny_spec, yaw_error):
    report = dict(passing_report(tiny_spec, "mujoco"), yaw_rmse=yaw_error)
    with pytest.raises(ValueError, match="yaw tracking"):
        validate_evidence([report], tiny_spec, "test")
