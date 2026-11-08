import numpy as np
import pytest
from amp_lab.retarget_targets import walking_foot_targets


def test_swing_bridges_meet_stance_without_position_or_velocity_jump():
    fps = 200
    time = np.arange(301) / fps
    source = np.zeros((len(time), 2, 3))
    source[:, :, 0] = time[:, None]
    source[:, :, 1] = [-.1, .1]
    source[:, :, 2] = [.05, .15]
    source[50:150, :, 2] = [.15, .05]
    source[250:, :, 2] = [.15, .05]
    before = source.copy()
    xy, clearance, labels = walking_foot_targets(source, fps)
    np.testing.assert_array_equal(source, before)
    np.testing.assert_allclose(xy[:50, 0, 0], 0)
    np.testing.assert_allclose(xy[150:250, 0, 0], .75)
    # The bridge starts at the old anchor; reverting to source here would jump 25cm.
    speed = np.diff(xy[:, 0, 0]) * fps
    assert abs(speed[49]) < .01
    assert abs(speed[149]) < .01
    assert xy[:, 0, 0].min() >= 0
    assert xy[:, 0, 0].max() <= source[-1, 0, 0]
    np.testing.assert_array_equal(clearance[np.arange(len(time)), labels], 0)
    assert clearance[49, 0] == clearance[150, 0] == 0
    assert clearance[100, 0] == pytest.approx(.1)


def test_unknown_contact_preserves_source_and_invalid_inputs_rejected():
    source = np.zeros((20, 2, 3))
    source[:, 1, 0] = np.sin(np.arange(20))
    source[:, 1, 2] = .2
    xy, _, _ = walking_foot_targets(source, 50)
    np.testing.assert_array_equal(xy[:, 1], source[:, 1, :2])
    with pytest.raises(ValueError):
        walking_foot_targets(source, 0)
