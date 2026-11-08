"""Kinematic diagnostics, never a substitute for measured contacts or dynamic rollout."""

import numpy as np
from scipy.spatial.transform import Rotation

from .motion import load_motion, sha256


def point_velocity(com_velocity, angular_velocity, point, centre_of_mass):
    """World velocity field at a point; motion linear velocities are at COM, not link origin."""
    return com_velocity + np.cross(angular_velocity, point - centre_of_mass)


def support_diagnostics(heights, speeds, fps):
    """Height-only support hypothesis, deliberately independent of speed to avoid hiding slip.

    Lowest foot is retained within 1 cm hysteresis; no support is assigned when both
    feet exceed 2.5 cm. This walking assumption cannot detect force or double support.
    """
    heights, speeds = np.asarray(heights), np.asarray(speeds)
    if heights.ndim != 2 or heights.shape[1] != 2 or speeds.shape != heights.shape:
        raise ValueError("Expected matching [frames,2] heights and speeds")
    support = None
    selected, labels = [], []
    for height, speed in zip(heights, speeds):
        candidate = int(np.argmin(height))
        if height[candidate] >= 0.025:
            support = None
        elif support is None or height[support] >= 0.025 or height[support] > height[candidate] + 0.01:
            support = candidate
        labels.append(-1 if support is None else support)
        if support is not None:
            selected.append(speed[support])
    labels = np.asarray(labels)
    missing = labels == -1
    runs = np.diff(np.r_[False, missing, False].astype(int))
    lengths = np.flatnonzero(runs == -1) - np.flatnonzero(runs == 1)
    switches = (labels[1:] >= 0) & (labels[:-1] >= 0) & (labels[1:] != labels[:-1])
    return {"definition": "height-only lowest foot, 1cm hysteresis, bottom<2.5cm; no force measurement",
            "coverage_fraction": float((~missing).mean()),
            "no_support_longest_s": float(lengths.max() / fps) if len(lengths) else 0.0,
            "direct_support_switches": int(switches.sum()),
            "selected_foot_horizontal_speed_mean_m_s": float(np.mean(selected)) if selected else None,
            "selected_foot_horizontal_speed_p95_m_s": float(np.percentile(selected, 95)) if selected else None}


def audit_motion(path, xml):
    import mujoco

    motion = load_motion(path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in motion["joint_names"]]
    if any(index < 0 for index in ids):
        raise ValueError("Motion joint set differs from model")
    limits = model.jnt_range[ids]
    q = motion["joint_pos"]
    excess = np.maximum(limits[:, 0] - q, np.maximum(q - limits[:, 1], 0))
    body_names = motion["body_names"].tolist()
    floor_heights, bottom_speeds, contact_speeds = [], [], []
    for foot in ("left_ankle_roll_link", "right_ankle_roll_link"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot)
        if body < 0 or foot not in body_names:
            raise ValueError(f"Missing foot {foot}")
        geoms = np.flatnonzero((model.geom_bodyid == body) &
                              (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) &
                              ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        if not len(geoms):
            raise ValueError("Foot audit currently supports G1 collision spheres only")
        index = body_names.index(foot)
        rotation = Rotation.from_quat(motion["body_quat_w"][:, index, [1, 2, 3, 0]])
        origin = motion["body_pos_w"][:, index]
        centres = np.stack([rotation.apply(np.tile(model.geom_pos[g], (len(q), 1))) + origin
                            for g in geoms], axis=1)
        bottoms = centres.copy()
        bottoms[:, :, 2] -= model.geom_size[geoms, 0]
        lowest = np.argmin(bottoms[:, :, 2], axis=1)
        points = bottoms[np.arange(len(q)), lowest]
        minimum = points[:, 2]
        com = rotation.apply(np.tile(model.body_ipos[body], (len(q), 1))) + origin
        velocity = point_velocity(motion["body_lin_vel_w"][:, index],
                                  motion["body_ang_vel_w"][:, index], points, com)
        bottom_speeds.append(np.linalg.norm(velocity[:, :2], axis=-1))
        floor_heights.append(minimum)
        # Preserve the original COM-based metric for comparisons with earlier reports.
        mask = minimum < 0.025
        speeds = np.linalg.norm(motion["body_lin_vel_w"][:, index, :2], axis=-1)
        contact_speeds.extend(speeds[mask].tolist())
    heights = np.stack(floor_heights, axis=-1)
    support = support_diagnostics(heights, np.stack(bottom_speeds, axis=-1), float(motion["fps"]))
    warnings = []
    if support["coverage_fraction"] < 0.9:
        warnings.append("LOW_SUPPORT_COVERAGE: fewer than 90% of frames have a foot bottom below 2.5cm")
    if (heights < -0.01).any():
        warnings.append("FOOT_PENETRATION: collision sphere bottom below -1cm")
    if support["selected_foot_horizontal_speed_p95_m_s"] is not None and (
            support["selected_foot_horizontal_speed_p95_m_s"] > 0.3):
        warnings.append("SUPPORT_SLIP_PROXY: selected foot bottom speed p95 exceeds 0.3m/s")
    return {"schema_version": 2, "motion_sha256": sha256(path), "frames": len(q),
            "duration_s": (len(q) - 1) / float(motion["fps"]),
            "max_joint_limit_excess_rad": float(excess.max()),
            "max_joint_velocity_rad_s": float(np.abs(motion["joint_vel"]).max()),
            "max_joint_acceleration_rad_s2": float(np.abs(np.gradient(
                motion["joint_vel"], 1 / float(motion["fps"]), axis=0)).max()),
            "minimum_foot_collision_height_m": float(heights.min()),
            "penetration_fraction_below_minus_1cm": float((heights < -0.01).mean()),
            "near_ground_foot_fraction": float((heights < 0.025).mean()),
            "near_ground_horizontal_speed_mean_m_s": float(np.mean(contact_speeds)) if contact_speeds else None,
            "contact_definition": "collision sphere bottom below 2.5cm; kinematic proxy, not measured force",
            "support_hypothesis": support, "review_warnings": warnings,
            "threshold_note": "engineering review heuristics, not validated acceptance criteria",
            "dynamic_feasibility_validated": False}
