"""Named, non-pickle motion interchange and MuJoCo forward-kinematics conversion."""

import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_motion(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    required = (*FIELDS, "fps", "joint_names", "body_names")
    missing = set(required) - result.keys()
    if missing:
        raise ValueError(f"Missing motion fields: {sorted(missing)}")
    fps = np.asarray(result["fps"])
    if fps.size != 1 or not np.isfinite(fps).all() or float(fps.reshape(-1)[0]) <= 0:
        raise ValueError("fps must be a positive scalar")
    joint_names, body_names = result["joint_names"].tolist(), result["body_names"].tolist()
    for name, names in [("joint_names", joint_names), ("body_names", body_names)]:
        if not names or any(not isinstance(x, str) for x in names) or len(set(names)) != len(names):
            raise ValueError(f"{name} must contain unique strings")
    frames = result["joint_pos"].shape[0]
    if frames < 2:
        raise ValueError("At least two motion frames are required")
    shapes = dict(joint_pos=(frames, len(joint_names)), joint_vel=(frames, len(joint_names)))
    shapes.update({key: (frames, len(body_names), 4 if key == "body_quat_w" else 3) for key in FIELDS[2:]})
    for key in FIELDS:
        if result[key].shape != shapes[key] or not np.isfinite(result[key]).all():
            raise ValueError(f"Invalid {key}: expected finite {shapes[key]}, got {result[key].shape}")
    if not np.allclose(np.linalg.norm(result["body_quat_w"], axis=-1), 1, atol=1e-3):
        raise ValueError("body_quat_w must contain unit wxyz quaternions")
    return result


def align_motion(path, output, joint_names, body_names, *, required_fps):
    motion = load_motion(Path(path))
    if not np.isclose(float(motion["fps"]), required_fps, rtol=0, atol=1e-4):
        raise ValueError(f"Motion fps must match control frequency {required_fps}; resample before training")
    for names, fields in [(joint_names, FIELDS[:2]), (body_names, FIELDS[2:])]:
        key = "joint_names" if fields == FIELDS[:2] else "body_names"
        saved = motion[key].tolist()
        missing = set(names) - set(saved)
        if missing:
            raise ValueError(f"Motion missing {key}: {sorted(missing)}")
        ids = [saved.index(name) for name in names]
        for field in fields:
            motion[field] = motion[field][:, ids]
        motion[key] = np.asarray(names)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **motion)
    return Path(output)


def qpos_to_motion(source: Path, xml: Path, output: Path, *, target_fps=50.0):
    """Input NPZ: root_pos, root_quat_wxyz, joint_pos, joint_names, fps. All units SI."""
    import mujoco
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp

    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive")
    with np.load(source, allow_pickle=False) as archive:
        root = np.asarray(archive["root_pos"], dtype=float)
        quat = np.asarray(archive["root_quat_wxyz"], dtype=float)
        joints = np.asarray(archive["joint_pos"], dtype=float)
        names = archive["joint_names"].tolist()
        fps = float(archive["fps"])
    frames = len(root)
    if (frames < 2 or root.shape != (frames, 3) or quat.shape != (frames, 4)
            or joints.shape != (frames, len(names)) or len(set(names)) != len(names)
            or not all(np.isfinite(x).all() for x in (root, quat, joints)) or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Invalid named qpos motion")
    model = mujoco.MjModel.from_xml_path(str(xml))
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Expected one free root joint")
    hinges = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    robot_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in hinges]
    if set(robot_names) != set(names):
        raise ValueError("Input joint names must match all robot hinge joints")
    # Retain the declared input joint order. Training alignment is an explicit separate step.
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    qadr = model.jnt_qposadr[joint_ids]
    root_qadr = int(model.jnt_qposadr[free[0]])
    times = np.arange(frames) / fps
    new_times = np.arange(int(np.floor(times[-1] * target_fps)) + 1) / target_fps
    if len(new_times) < 2:
        raise ValueError("Motion is too short at target_fps")
    root = interp1d(times, root, axis=0)(new_times)
    joints = interp1d(times, joints, axis=0)(new_times)
    quat_xyzw = Slerp(times, Rotation.from_quat(quat[:, [1, 2, 3, 0]]))(new_times).as_quat()
    quat = quat_xyzw[:, [3, 0, 1, 2]]
    qpos = np.tile(model.qpos0, (len(new_times), 1))
    qpos[:, root_qadr:root_qadr+3] = root
    qpos[:, root_qadr+3:root_qadr+7] = quat
    qpos[:, qadr] = joints
    body_ids = [i for i in range(1, model.nbody) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)]
    data = mujoco.MjData(model)
    positions, rotations, linear, angular = [], [], [], []
    dt = 1 / target_fps
    for t in range(len(qpos)):
        prev, next_ = max(t - 1, 0), min(t + 1, len(qpos) - 1)
        data.qpos[:] = qpos[t]
        mujoco.mj_differentiatePos(model, data.qvel, (next_ - prev) * dt, qpos[prev], qpos[next_])
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[body_ids].copy())
        rotations.append(data.xquat[body_ids].copy())
        velocities = []
        for body in body_ids:
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            velocities.append(velocity)
        velocities = np.asarray(velocities)
        angular.append(velocities[:, :3])
        linear.append(velocities[:, 3:])
    motion = dict(fps=np.float64(target_fps), joint_names=np.asarray(names), joint_pos=joints,
                  joint_vel=np.gradient(joints, dt, axis=0),
                  body_names=np.asarray([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in body_ids]),
                  body_pos_w=np.asarray(positions), body_quat_w=np.asarray(rotations),
                  body_lin_vel_w=np.asarray(linear), body_ang_vel_w=np.asarray(angular))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **motion)
    load_motion(output)
    report = dict(source=str(source), source_sha256=sha256(source), model_sha256=sha256(xml),
                  output_sha256=sha256(output), frames=len(qpos), fps=target_fps,
                  origin="user_supplied_named_joint_motion", retarget_quality="not_evaluated",
                  redistribution_status="not_reviewed")
    from .provenance import asset_manifest
    robot_directory = xml.parent.parent if xml.parent.name == "mjcf" else xml.parent
    report["model_tree_sha256"] = asset_manifest(robot_directory)["tree_sha256"]
    retarget_report = source.with_suffix(".retarget.json")
    if retarget_report.is_file():
        upstream = json.loads(retarget_report.read_text())
        if upstream.get("qpos_sha256") != report["source_sha256"]:
            raise ValueError("Retarget provenance does not match input qpos")
        report.update(origin=upstream["source"], retarget_quality=upstream["status"],
                      redistribution_status=upstream["license_note"], retarget=upstream)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
