import numpy as np
import pytest
from amp_lab.quality import point_velocity, support_diagnostics


def test_point_velocity_accounts_for_com_and_rotation():
    # COM at z=.1, bottom at zero, rotating about y: bottom x velocity is -.2.
    result = point_velocity(np.array([[1., 0, 0]]), np.array([[0., 2, 0]]),
                            np.array([[0., 0, 0]]), np.array([[0., 0, .1]]))
    np.testing.assert_allclose(result, [[.8, 0, 0]])


def test_support_does_not_select_slow_swing_foot_or_hide_missing_contact():
    heights = [[0, .02], [.005, 0], [.04, .06], [.06, .07], [.03, 0]]
    # High speed must remain in the metric; using speed to infer support would hide it.
    result = support_diagnostics(heights, [[1, 0], [1, 0], [0, 0], [0, 0], [0, .2]], 50)
    assert result['coverage_fraction'] == pytest.approx(.6)
    assert result['no_support_longest_s'] == pytest.approx(.04)
    assert result['selected_foot_horizontal_speed_mean_m_s'] == pytest.approx(2.2 / 3)
    assert result['direct_support_switches'] == 0
    missing = support_diagnostics([[.1, .1]], [[0, 0]], 50)
    assert missing['selected_foot_horizontal_speed_mean_m_s'] is None
