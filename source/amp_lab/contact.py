"""Optional kinematic foot-lock baseline; measured contacts still require physics validation."""

import json
from pathlib import Path

import numpy as np

from .motion import sha256


def project_walking_contacts(source, xml, output):
    import mujoco

    with np.load(source, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in motion["joint_names"]]
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
              for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    if any(index < 0 for index in ids + bodies):
        raise ValueError("Missing G1 joint/foot")
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Expected one free joint")
    root_qadr = model.jnt_qposadr[free[0]]
    qadr = model.jnt_qposadr[ids]
    foot_positions, bottoms = [], []
    geoms = [np.flatnonzero((model.geom_bodyid == body) &
                           (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)) for body in bodies]
    if any(len(group) == 0 for group in geoms):
        raise ValueError("Expected G1 foot collision spheres")
    for frame in range(len(motion["root_pos"])):
        data.qpos[root_qadr:root_qadr+3] = motion["root_pos"][frame]
        data.qpos[root_qadr+3:root_qadr+7] = motion["root_quat_wxyz"][frame]
        data.qpos[qadr] = motion["joint_pos"][frame]
        mujoco.mj_forward(model, data)
        foot_positions.append(data.xpos[bodies].copy())
        bottoms.append([min(data.geom_xpos[group, 2] - model.geom_size[group, 0]) for group in geoms])
    foot_positions, bottoms = np.asarray(foot_positions), np.asarray(bottoms)
    support, anchor = None, None
    offset = np.zeros(2)
    corrections, support_ids = [], []
    for frame in range(len(bottoms)):
        candidate = int(np.argmin(bottoms[frame]))
        # 1 cm hysteresis keeps support from flickering in double stance.
        if support is None or bottoms[frame, support] > bottoms[frame, candidate] + 0.01:
            support = candidate
            anchor = foot_positions[frame, support, :2] + offset
        offset = anchor - foot_positions[frame, support, :2]
        correction = np.r_[offset, -bottoms[frame].min()]
        corrections.append(correction)
        support_ids.append(support)
    corrections = np.asarray(corrections)
    motion["root_pos"] = motion["root_pos"] + corrections
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, **motion)
    upstream_path = Path(source).with_suffix(".retarget.json")
    upstream = json.loads(upstream_path.read_text()) if upstream_path.exists() else {"source": "user_supplied_motion"}
    report = dict(upstream, method="bounded_position_ik_plus_kinematic_foot_lock_v1",
                  status="candidate_requires_visual_and_dynamic_review", input_qpos_sha256=sha256(source),
                  qpos_sha256=sha256(output), support_switches=int(np.count_nonzero(np.diff(support_ids))),
                  root_correction_max_m=float(np.linalg.norm(corrections, axis=-1).max()),
                  root_correction_speed_max_m_s=float(np.linalg.norm(
                      np.diff(corrections, axis=0) * float(motion["fps"]), axis=-1).max()),
                  contact_consistency_validated=False, dynamics_validated=False,
                  contact_assumption="walking_no_flight_lowest_foot_with_1cm_hysteresis")
    output.with_suffix(".retarget.json").write_text(json.dumps(report, indent=2))
    return report
