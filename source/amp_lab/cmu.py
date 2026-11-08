"""CMU ASF/AMC FK and a small bounded position-IK baseline for the G1 model.

This is an inspectable retargeting baseline, not a claim of contact-consistent motion.
ASF bone axes and DOF order are honored; CMU Y-up coordinates become robot Z-up.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .motion import sha256

CMU_METRES = (1 / 0.45) * 2.54 / 100
CMU_TO_ROBOT = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)


def read_asf(path):
    bones, parents, section, bone = {}, {}, None, None
    root = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(":"):
            section = line[1:]
            continue
        fields = line.split()
        if section == "units" and fields[0] == "angle" and fields[1] != "deg":
            raise ValueError("Only degree-valued CMU skeletons supported")
        if section == "root":
            root[fields[0]] = fields[1:]
        elif section == "bonedata":
            if line == "begin":
                bone = {"dof": []}
            elif line == "end":
                bones[bone["name"]] = bone
                bone = None
            elif bone is not None and fields[0] in ("name", "direction", "length", "axis", "dof"):
                key = fields[0]
                bone[key] = fields[1] if key == "name" else fields[1:]
        elif section == "hierarchy" and line not in ("begin", "end"):
            for child in fields[1:]:
                if child in parents:
                    raise ValueError("Multiple parents in ASF hierarchy")
                parents[child] = fields[0]
    if root.get("order") != ["TX", "TY", "TZ", "RX", "RY", "RZ"] or root.get("axis") != ["XYZ"]:
        raise ValueError("Unsupported ASF root order/axis")
    if set(parents) != set(bones):
        raise ValueError("ASF hierarchy must cover every bone")
    for bone in bones.values():
        axis = bone["axis"]
        bone["basis"] = Rotation.from_euler(axis[3].lower(), np.asarray(axis[:3], float), degrees=True).as_matrix()
        bone["offset"] = np.asarray(bone["direction"], float) * float(bone["length"][0]) * CMU_METRES
        if any(dof not in ("rx", "ry", "rz") for dof in bone["dof"]):
            raise ValueError("Unsupported translational bone DOF")
    if np.any(np.asarray(root.get("position", [0, 0, 0]), float)) or np.any(
            np.asarray(root.get("orientation", [0, 0, 0]), float)):
        raise ValueError("Nonzero ASF root rest transform requires explicit adaptation")
    return bones, parents


def read_amc(path, bones):
    frames, frame, numbers = [], None, []
    text = Path(path).read_text()
    if ":DEGREES" not in text or ":FULLY-SPECIFIED" not in text:
        raise ValueError("Expected fully specified AMC in degrees")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#:":
            continue
        if line.isdigit():
            if frame is not None:
                frames.append(frame)
            frame = {}
            numbers.append(int(line))
        else:
            fields = line.split()
            frame[fields[0]] = np.asarray(fields[1:], dtype=float)
    if frame is not None:
        frames.append(frame)
    if len(frames) < 2 or numbers != list(range(1, len(frames) + 1)):
        raise ValueError("AMC frames must be contiguous from 1")
    for frame in frames:
        if frame.get("root", np.zeros(0)).shape != (6,):
            raise ValueError("AMC missing six-DOF root")
        for name, bone in bones.items():
            if bone["dof"] and frame.get(name, np.zeros(0)).shape != (len(bone["dof"]),):
                raise ValueError(f"AMC missing/malformed joint {name}")
        if not all(np.isfinite(values).all() for values in frame.values()):
            raise ValueError("Nonfinite AMC values")
    return frames


def forward_kinematics(bones, parents, frame):
    positions = {"root": frame["root"][:3] * CMU_METRES}
    rotations = {"root": Rotation.from_euler("xyz", frame["root"][3:], degrees=True).as_matrix()}

    def visit(name, active):
        if name in positions:
            return
        if name in active:
            raise ValueError("Cyclic ASF hierarchy")
        parent = parents[name]
        visit(parent, active | {name})
        bone = bones[name]
        motion = np.eye(3)
        if bone["dof"]:
            # Extrinsic rotations in declared order: successive matrices pre-multiply.
            for dof, angle in zip(bone["dof"], frame[name]):
                motion = Rotation.from_euler(dof[1], angle, degrees=True).as_matrix() @ motion
        basis = bone["basis"]
        rotations[name] = rotations[parent] @ basis @ motion @ basis.T
        positions[name] = positions[parent] + rotations[name] @ bone["offset"]

    for name in bones:
        visit(name, set())
    return ({name: CMU_TO_ROBOT @ position for name, position in positions.items()},
            {name: CMU_TO_ROBOT @ rotation @ CMU_TO_ROBOT.T for name, rotation in rotations.items()})


def retarget_cmu(asf, amc, xml, output, *, fps=120.0, output_fps=50.0, max_frames=None,
                 experimental_floor_ik=False, floor_ik_variant="smooth_v4", joint_speed_limit=None, root_height_offset=0.0,
                 swing_height_scale=1.0):
    import mujoco
    from scipy.optimize import least_squares

    from .g1 import default_spec

    if not experimental_floor_ik and (root_height_offset != 0 or joint_speed_limit is not None
                                      or swing_height_scale != 1):
        raise ValueError("Experimental height/speed options require experimental_floor_ik=True")
    if not np.isfinite(swing_height_scale) or not 0 < swing_height_scale <= 1:
        raise ValueError("swing_height_scale must be in (0,1]")
    if not np.isfinite(root_height_offset):
        raise ValueError("root_height_offset must be finite")
    if joint_speed_limit is not None and (not np.isfinite(joint_speed_limit) or joint_speed_limit <= 0):
        raise ValueError("joint_speed_limit must be finite and positive")
    if floor_ik_variant not in ("anchor_v2", "smooth_v3", "smooth_v4"):
        raise ValueError("Unknown floor IK variant")
    if Path(output).exists():
        raise FileExistsError(output)
    if not np.isfinite([fps, output_fps]).all() or fps <= 0 or output_fps <= 0 or output_fps > fps:
        raise ValueError("Require 0 < output_fps <= input fps")
    bones, parents = read_asf(asf)
    frames = read_amc(amc, bones)
    # Interpolate AMC angular channels before FK. Unwrap to preserve rotations across +/-180 degrees.
    source_t = np.arange(len(frames)) / fps
    target_t = np.arange(int(np.floor(source_t[-1] * output_fps)) + 1) / output_fps
    if max_frames:
        target_t = target_t[:max_frames]
    interpolated = [{} for _ in target_t]
    for name in frames[0]:
        values = np.stack([frame[name] for frame in frames])
        angle_start = 3 if name == "root" else 0
        values[:, angle_start:] = np.rad2deg(np.unwrap(np.deg2rad(values[:, angle_start:]), axis=0))
        for col in range(values.shape[1]):
            column = np.interp(target_t, source_t, values[:, col])
            for index, value in enumerate(column):
                if name not in interpolated[index]:
                    interpolated[index][name] = np.empty(values.shape[1])
                interpolated[index][name][col] = value
    human = [forward_kinematics(bones, parents, frame) for frame in interpolated]
    if len(human) < 2:
        raise ValueError("Retargeting needs at least two output frames")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    spec = default_spec()
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in spec.joint_names]
    if any(i < 0 for i in joint_ids):
        raise ValueError("Missing G1 joints")
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Expected one free root")
    root_qadr = model.jnt_qposadr[free[0]]
    pairs = [("left_knee_link", "lfemur", 0.6), ("right_knee_link", "rfemur", 0.6),
             ("left_ankle_roll_link", "ltibia", 2.0), ("right_ankle_roll_link", "rtibia", 2.0),
             ("left_elbow_link", "lhumerus", 0.4), ("right_elbow_link", "rhumerus", 0.4),
             ("left_wrist_yaw_link", "lradius", 0.4), ("right_wrist_yaw_link", "rradius", 0.4)]
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pair[0]) for pair in pairs]
    if any(i < 0 for i in bodies):
        raise ValueError("Missing retargeting body")
    foot_geoms = {}
    if experimental_floor_ik:
        for body in bodies[2:4]:
            group = np.flatnonzero((model.geom_bodyid == body) &
                                   (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) &
                                   ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
            if not len(group):
                raise ValueError("Floor IK requires G1 foot collision spheres")
            foot_geoms[body] = group
    # Use known G1 hip-to-ankle reach; record this baseline anthropometric scaling explicitly.
    human_leg = sum(np.linalg.norm(bones[name]["offset"]) for name in ("lhipjoint", "lfemur", "ltibia"))
    scale = 0.78 / human_leg
    root_initial = human[0][0]["root"].copy()
    all_ankles = [positions[name][2] * scale for positions, _ in human for name in ("ltibia", "rtibia")]
    height_shift = 0.055 - float(np.percentile(all_ankles, 1)) + root_height_offset
    previous = np.asarray(spec.default_joint_pos)
    lower, upper = model.jnt_range[joint_ids, 0] + 1e-4, model.jnt_range[joint_ids, 1] - 1e-4
    positions_out, rotations_out, joints_out, errors = [], [], [], []
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    smooth_targets = None
    if experimental_floor_ik and floor_ik_variant in ("smooth_v3", "smooth_v4"):
        from .retarget_targets import walking_foot_targets
        ankles = np.array([[positions[name] * scale for name in ("ltibia", "rtibia")]
                           for positions, _ in human])
        ankles[:, :, :2] -= root_initial[:2] * scale
        ankles[:, :, 2] += height_shift
        smooth_targets = walking_foot_targets(ankles, output_fps)
    source_support, support_anchor = None, None
    for frame_index, (positions, rotations) in enumerate(human):
        root = positions["root"] * scale
        root[:2] -= root_initial[:2] * scale
        root[2] += height_shift
        rotation = rotations["root"]
        quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        targets = [(positions[name] - positions["root"]) * scale + root for _, name, _ in pairs]
        desired_floor = None
        if smooth_targets is not None:
            xy, clearance, _ = smooth_targets
            for foot in range(2):
                targets[2 + foot][:2] = xy[frame_index, foot]
            desired_floor = clearance[frame_index] * swing_height_scale
            if floor_ik_variant == "smooth_v4":
                for foot, body in enumerate(bodies[2:4]):
                    group = foot_geoms[body]
                    sole_offset = -np.min(model.geom_pos[group, 2] - model.geom_size[group, 0])
                    # Human ankles are not robot sole frames; do not use their absolute Z twice.
                    targets[2 + foot][2] = desired_floor[foot] + sole_offset
        elif experimental_floor_ik:
            ankle_heights = np.array([targets[2][2], targets[3][2]])
            candidate = int(np.argmin(ankle_heights))
            if source_support is None or ankle_heights[source_support] > ankle_heights[candidate] + 0.01:
                source_support = candidate
                support_anchor = targets[2 + source_support][:2].copy()
            targets[2 + source_support][:2] = support_anchor
            # Flat walking, no flight: support foot is at ground; preserve relative swing clearance.
            desired_floor = np.maximum(0.0, ankle_heights - ankle_heights[source_support])
        data.qpos[root_qadr:root_qadr+3] = root
        data.qpos[root_qadr+3:root_qadr+7] = quat
        prior = previous.copy()
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        heading = np.array([np.cos(yaw), np.sin(yaw), 0])

        def evaluate(q, jacobian=False):
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            residual, rows = [], []
            for index, (body, target, (_, _, weight)) in enumerate(zip(bodies, targets, pairs)):
                if jacobian:
                    mujoco.mj_jacBody(model, data, jacp, jacr, body)
                    rows.append(weight * jacp[:, vadr].copy())
                residual.extend(weight * (data.xpos[body] - target))
                if index in (2, 3):
                    # Stronger flat-foot prior only for the opt-in floor-aware experiment.
                    orientation_weight = 0.15 if experimental_floor_ik else 0.03
                    matrix = data.xmat[body].reshape(3, 3)
                    for axis, desired in [(matrix[:, 2], np.array([0, 0, 1])), (matrix[:, 0], heading)]:
                        residual.extend(orientation_weight * (axis - desired))
                        if jacobian:
                            rows.append(orientation_weight * np.cross(jacr[:, vadr].T, axis).T)
                    if experimental_floor_ik:
                        group = foot_geoms[body]
                        heights = data.geom_xpos[group, 2] - model.geom_size[group, 0]
                        geom = group[np.argmin(heights)]
                        # Relative source ankle height defines swing clearance under a no-flight assumption.
                        # No speed threshold is used, and no measured-contact claim is made.
                        desired_bottom = desired_floor[index - 2]
                        residual.append(3.0 * (heights.min() - desired_bottom))
                        if jacobian:
                            mujoco.mj_jac(model, data, jacp, jacr, data.geom_xpos[geom], body)
                            rows.append(3.0 * jacp[2:3, vadr].copy())
            residual.extend(0.02 * (q - prior))
            residual.extend(0.005 * (q - spec.default_joint_pos))
            if jacobian:
                rows.extend([0.02 * np.eye(len(q)), 0.005 * np.eye(len(q))])
                return np.concatenate(rows)
            return np.asarray(residual)

        local_lower, local_upper = lower, upper
        if joint_speed_limit is not None and frame_index > 0:
            delta = joint_speed_limit / output_fps
            local_lower = np.maximum(lower, prior - delta)
            local_upper = np.minimum(upper, prior + delta)
        solution = least_squares(evaluate, np.clip(prior, local_lower, local_upper), jac=lambda q: evaluate(q, True),
                                 bounds=(local_lower, local_upper), max_nfev=30, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        previous = solution.x
        evaluate(previous)
        errors.append([float(np.linalg.norm(data.xpos[body] - target)) for body, target in zip(bodies, targets)])
        positions_out.append(root)
        rotations_out.append(quat)
        joints_out.append(previous.copy())
        if frame_index % 100 == 0:
            print(f"Retarget {frame_index + 1}/{len(human)}", flush=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, fps=np.float64(output_fps), root_pos=np.asarray(positions_out),
                        root_quat_wxyz=np.asarray(rotations_out), joint_pos=np.asarray(joints_out),
                        joint_names=np.asarray(spec.joint_names))
    report = {"method": "cmu_asf_fk_bounded_position_ik_v1", "status": "candidate_requires_visual_and_contact_review",
              "asf_sha256": sha256(asf), "amc_sha256": sha256(amc), "model_xml_sha256": sha256(xml),
              "qpos_sha256": sha256(output), "frames": len(human), "fps": output_fps,
              "human_scale": float(scale), "height_shift_m": height_shift,
              "experimental_floor_ik": experimental_floor_ik,
              "joint_speed_limit_rad_s": joint_speed_limit,
              "root_height_offset_m": root_height_offset,
              "swing_height_scale": swing_height_scale,
              "floor_ik_settings": ({"bottom_weight": 3.0, "orientation_weight": 0.15,
                                     "variant": floor_ik_variant,
                                     "support_assumption": "flat_walk_no_flight_1cm_height_hysteresis",
                                     "root_trajectory_modified": root_height_offset != 0.0} if experimental_floor_ik else None),
              "target_error_mean_m": float(np.mean(errors)), "target_error_p95_m": float(np.percentile(errors, 95)),
              "source": "CMU Graphics Lab Motion Capture Database; NSF EIA-0196217",
              "license_note": "Free redistribution with source attribution; do not resell the data",
              "contact_consistency_validated": False, "dynamics_validated": False}
    output.with_suffix(".retarget.json").write_text(json.dumps(report, indent=2))
    return report
