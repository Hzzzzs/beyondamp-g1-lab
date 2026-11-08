import numpy as np
import pytest
from amp_lab.gait import hip_periodicity, summarize_gait, support_labels


def test_alternating_gait_has_known_cadence_stride_and_clearance():
    fps = 50
    t = np.arange(301) / fps
    support = (np.arange(len(t)) // 25) % 2 == 0
    heights = np.stack([np.where(support, 0, 0.1), np.where(support, 0.1, 0)], axis=1)
    speeds = np.where(heights < 0.025, 0.02, 1)
    result = summarize_gait(heights, speeds, np.stack([0.6 * t, 0 * t], axis=1), fps)
    assert result["cadence_steps_per_minute"] == pytest.approx(120)
    assert result["mean_stride_length_m"] == pytest.approx(0.6)
    assert result["mean_swing_peak_clearance_m"] == pytest.approx(0.1)
    assert result["near_ground_bottom_speed_mean_m_s"] == pytest.approx(0.02)
    assert result["complete_stride_count"] > 0


def test_standing_is_not_invented_as_steps_and_height_jitter_is_ignored():
    height = np.zeros((100, 2))
    height[25:27, 0] = 0.1
    assert support_labels(height, 50).all()
    result = summarize_gait(height, np.zeros_like(height), np.zeros((100, 2)), 50)
    assert result["cadence_steps_per_minute"] is None
    assert result["mean_stride_length_m"] is None
    assert result["complete_swing_count"] == 0


def test_gait_rejects_nonfinite_signals():
    with pytest.raises(ValueError):
        summarize_gait(np.full((4, 2), np.nan), np.zeros((4, 2)), np.zeros((4, 2)), 50)


def test_hip_periodicity_measures_cycle_without_claiming_contact_events():
    t = np.arange(500) / 50
    joints = np.stack([np.sin(2 * np.pi * t), -np.sin(2 * np.pi * t)], axis=1)
    report = hip_periodicity(joints, ["left_hip_pitch_joint", "right_hip_pitch_joint"], 50)
    # A sampled peak can move by one 20ms frame; this is not a continuous-time fit.
    assert report["feet"][0]["cadence_steps_per_minute"] == pytest.approx(120, abs=0.5)
    standing = hip_periodicity(joints * 0, ["left_hip_pitch_joint", "right_hip_pitch_joint"], 50)
    assert standing["feet"][0]["mean_cycle_hz"] is None
