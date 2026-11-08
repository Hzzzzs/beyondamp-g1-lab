import pytest
from amp_lab.model_audit import audit_models


def test_fixed_link_masses_merge_before_comparison(tmp_path, tiny_robot):
    urdf = tmp_path / "robot.urdf"
    inertia = '<inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0005"/>'
    links = "".join(f'<link name="{name}"><inertial><mass value="{mass}"/>{inertia}</inertial></link>'
                    for name, mass in [("pelvis", 0.5), ("fixed_mass", 0.5), ("first", 0.2), ("second", 0.1)])
    urdf.write_text(f'''<robot name="fixture">{links}
      <joint name="fixed" type="fixed"><parent link="pelvis"/><child link="fixed_mass"/></joint>
      <joint name="a" type="revolute"><parent link="pelvis"/><child link="first"/>
        <axis xyz="0 1 0"/><limit lower="-2.96705972839" upper="2.96705972839"/></joint>
      <joint name="b" type="revolute"><parent link="first"/><child link="second"/>
        <origin xyz="0 0 -0.2"/><axis xyz="1 0 0"/>
        <limit lower="-2.96705972839" upper="2.96705972839"/></joint>
    </robot>''')
    report = audit_models(urdf, tiny_robot)
    assert report["urdf_total_mass_kg"] == pytest.approx(1.3)
    assert report["max_mass_difference_kg"] < 1e-10
    assert report["max_zero_pose_origin_distance_m"] < 1e-10
    assert report["max_joint_axis_error"] < 1e-10
    pelvis = next(row for row in report["bodies"] if row["body"] == "pelvis")
    assert pelvis["inertia_max_abs_difference_kg_m2"] < 1e-10
