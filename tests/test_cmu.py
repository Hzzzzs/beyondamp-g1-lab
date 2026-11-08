import numpy as np
import pytest
from amp_lab.cmu import CMU_TO_ROBOT, forward_kinematics, read_amc, read_asf
from scipy.spatial.transform import Rotation


def test_asf_axis_basis_and_dof_order(tmp_path):
    path = tmp_path / "tiny.asf"
    path.write_text(''':units
angle deg
:root
order TX TY TZ RX RY RZ
axis XYZ
position 0 0 0
orientation 0 0 0
:bonedata
begin
id 1
name limb
direction 1 0 0
length 1
axis 0 0 45 XYZ
dof ry rx
end
:hierarchy
begin
root limb
end
''')
    bones, parents = read_asf(path)
    frame = {"root": np.zeros(6), "limb": np.array([30, 60])}
    positions, rotations = forward_kinematics(bones, parents, frame)
    basis = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    motion = Rotation.from_euler("x", 60, degrees=True).as_matrix() @ Rotation.from_euler(
        "y", 30, degrees=True).as_matrix()
    expected_rotation = basis @ motion @ basis.T
    np.testing.assert_allclose(rotations["limb"], CMU_TO_ROBOT @ expected_rotation @ CMU_TO_ROBOT.T)
    np.testing.assert_allclose(positions["limb"], CMU_TO_ROBOT @ expected_rotation @ bones["limb"]["offset"])
    amc = tmp_path / "tiny.amc"
    amc.write_text(":FULLY-SPECIFIED\n:DEGREES\n1\nroot 0 0 0 0 0 0\nlimb 30 60\n"
                   "2\nroot 1 0 0 0 0 0\nlimb 40 50\n")
    assert len(read_amc(amc, bones)) == 2
    amc.write_text(amc.read_text().replace("limb 40 50", "limb 40"))
    with pytest.raises(ValueError, match="malformed"):
        read_amc(amc, bones)
