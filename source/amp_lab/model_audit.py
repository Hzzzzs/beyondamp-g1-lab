"""Compare source URDF and MJCF at zero joint angles, including merged fixed-link inertias."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .motion import sha256


def _numbers(text, size=3):
    values = np.asarray([float(value) for value in text.split()])
    if values.shape != (size,):
        raise ValueError("Malformed URDF vector")
    return values


def _origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = _numbers(element.get("xyz", "0 0 0"))
        transform[:3, :3] = Rotation.from_euler("xyz", _numbers(element.get("rpy", "0 0 0"))).as_matrix()
    return transform


def audit_models(urdf, xml):
    import mujoco

    tree = ET.parse(urdf).getroot()
    links = {link.get("name"): link for link in tree.findall("link")}
    joints = {joint.get("name"): joint for joint in tree.findall("joint")}
    parents = {joint.find("child").get("link"): joint for joint in joints.values()}
    transforms = {}

    def transform(name):
        if name not in transforms:
            joint = parents.get(name)
            transforms[name] = (np.eye(4) if joint is None else
                                transform(joint.find("parent").get("link")) @ _origin(joint.find("origin")))
        return transforms[name]

    merged = {}
    for name, link in links.items():
        inertial = link.find("inertial")
        if inertial is None:
            continue
        ancestor = name
        while ancestor in parents and parents[ancestor].get("type") == "fixed":
            ancestor = parents[ancestor].find("parent").get("link")
        relative = np.linalg.inv(transform(ancestor)) @ transform(name) @ _origin(inertial.find("origin"))
        mass = float(inertial.find("mass").get("value"))
        coefficients = inertial.find("inertia")
        inertia = np.array([[float(coefficients.get(f"i{x}{y}" if x <= y else f"i{y}{x}"))
                             for y in "xyz"] for x in "xyz"])
        rotation, position = relative[:3, :3], relative[:3, 3]
        inertia = rotation @ inertia @ rotation.T
        merged.setdefault(ancestor, []).append((mass, position, inertia))
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Expected a single free root")
    data.qpos[:] = 0
    root_qadr = model.jnt_qposadr[free[0]]
    data.qpos[root_qadr+3] = 1
    mujoco.mj_forward(model, data)
    differences = []
    for index in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        if name not in merged:
            continue
        components = merged[name]
        mass = sum(component[0] for component in components)
        com = sum(m * p for m, p, _ in components) / mass
        inertia = np.zeros((3, 3))
        for m, p, value in components:
            offset = p - com
            inertia += value + m * (np.eye(3) * offset.dot(offset) - np.outer(offset, offset))
        rotation = Rotation.from_quat(model.body_iquat[index, [1, 2, 3, 0]]).as_matrix()
        mj_inertia = rotation @ np.diag(model.body_inertia[index]) @ rotation.T
        differences.append({"body": name, "urdf_mass_kg": mass, "mjcf_mass_kg": float(model.body_mass[index]),
                            "mass_difference_kg": float(model.body_mass[index] - mass),
                            "com_distance_m": float(np.linalg.norm(model.body_ipos[index] - com)),
                            "inertia_max_abs_difference_kg_m2": float(np.abs(mj_inertia - inertia).max()),
                            "zero_pose_origin_distance_m": float(np.linalg.norm(data.xpos[index] - transform(name)[:3, 3]))})
    joint_checks = []
    for index in range(model.njnt):
        if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        joint = joints[name]
        child = joint.find("child").get("link")
        axis = transform(child)[:3, :3] @ _numbers(joint.find("axis").get("xyz"))
        limit = joint.find("limit")
        joint_checks.append({"joint": name, "axis_world_error": float(np.linalg.norm(axis - data.xaxis[index])),
                             "limit_max_error_rad": float(np.abs(model.jnt_range[index] -
                                 [float(limit.get("lower")), float(limit.get("upper"))]).max())})
    return {"urdf_sha256": sha256(urdf), "mjcf_xml_sha256": sha256(xml), "bodies": differences,
            "joints": joint_checks, "urdf_total_mass_kg": sum(m for values in merged.values() for m, _, _ in values),
            "mjcf_total_mass_kg": float(model.body_mass.sum()),
            "max_mass_difference_kg": max(abs(row["mass_difference_kg"]) for row in differences),
            "max_zero_pose_origin_distance_m": max(row["zero_pose_origin_distance_m"] for row in differences),
            "max_joint_axis_error": max(row["axis_world_error"] for row in joint_checks),
            "physx_import_audited": False, "contact_dynamics_equivalence_validated": False}
