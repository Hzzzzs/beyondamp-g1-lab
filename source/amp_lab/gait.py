"""Phase-free gait descriptors shared by reference motion and policy rollouts.

Foot support is inferred from collision-sphere height, never from force. These
descriptors expose differences; they are not a calibrated imitation score.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from .quality import point_velocity, support_diagnostics


def support_labels(heights, fps, *, enter=0.015, leave=0.035, hold_seconds=0.06):
    """Height hysteresis with sustained transitions; retain original event time."""
    heights = np.asarray(heights)
    labels = np.empty(heights.shape, dtype=bool)
    hold = max(1, int(np.ceil(hold_seconds * fps)))
    for foot in range(heights.shape[1]):
        supported = bool(heights[0, foot] < enter)
        pending = 0
        for frame, height in enumerate(heights[:, foot]):
            transition = height > leave if supported else height < enter
            pending = pending + 1 if transition else 0
            labels[frame, foot] = supported
            if pending >= hold:
                supported = not supported
                labels[frame - hold + 1:frame + 1, foot] = supported
                pending = 0
    return labels


def hip_periodicity(joints, names, fps):
    """Hip-pitch peak intervals, independent of a foot contact threshold."""
    estimates = []
    for side in ["left", "right"]:
        values = joints[:, names.index(side + "_hip_pitch_joint")]
        prominence = max(0.05, 0.2 * np.ptp(values))
        peaks, _ = find_peaks(values, prominence=prominence, distance=max(1, round(0.25 * fps)))
        intervals = np.diff(peaks) / fps
        cycle = 1 / float(intervals.mean()) if len(intervals) else None
        estimates.append(dict(side=side, mean_cycle_hz=cycle, complete_cycle_count=len(intervals),
                              cadence_steps_per_minute=120 * cycle if cycle is not None else None,
                              hip_pitch_range_p95_minus_p5_rad=float(np.diff(
                                  np.percentile(values, [5, 95]))[0])))
    return dict(method="hip pitch peak intervals, prominence=max(0.05rad,20% range), min gap0.25s; "
                       "assumes two steps/cycle; short references contain few complete cycles", feet=estimates)


def summarize_gait(heights, bottom_speeds, root_xy, fps):
    heights, bottom_speeds, root_xy = map(np.asarray, (heights, bottom_speeds, root_xy))
    if (heights.ndim != 2 or heights.shape[1] != 2 or len(heights) < 2
            or bottom_speeds.shape != heights.shape or root_xy.shape != (len(heights), 2)
            or not np.isfinite(fps) or fps <= 0
            or not all(np.isfinite(x).all() for x in (heights, bottom_speeds, root_xy))):
        raise ValueError("Expected finite two-foot signals and root XY at positive fps")
    labels = support_labels(heights, fps)
    periods, strides, peaks = [], [], []
    touchdowns = []
    for foot in range(2):
        touchdown = np.flatnonzero(~labels[:-1, foot] & labels[1:, foot]) + 1
        liftoff = np.flatnonzero(labels[:-1, foot] & ~labels[1:, foot]) + 1
        touchdowns.append(int(len(touchdown)))
        for first, last in zip(touchdown[:-1], touchdown[1:]):
            periods.append((last - first) / fps)
            strides.append(np.linalg.norm(root_xy[last] - root_xy[first]))
        for first in liftoff:
            later = touchdown[touchdown > first]
            if len(later):
                peaks.append(float(np.max(heights[first:later[0] + 1, foot])))
    near_ground = heights < 0.025
    speeds = bottom_speeds[near_ground]
    duration = (len(heights) - 1) / fps
    return dict(duration_s=duration, mean_root_speed_m_s=float(np.linalg.norm(
        np.diff(root_xy, axis=0), axis=1).sum() / duration),
        cadence_steps_per_minute=120 / float(np.mean(periods)) if periods else None,
        mean_stride_period_s=float(np.mean(periods)) if periods else None,
        mean_stride_length_m=float(np.mean(strides)) if strides else None,
        complete_stride_count=len(periods), touchdown_counts_left_right=touchdowns,
        mean_swing_peak_clearance_m=float(np.mean(peaks)) if peaks else None,
        complete_swing_count=len(peaks),
        foot_bottom_height_p95_left_right_m=np.percentile(heights, 95, axis=0).tolist(),
        foot_bottom_height_max_left_right_m=heights.max(axis=0).tolist(),
        height_proxy_support_fraction_left_right=labels.mean(axis=0).tolist(),
        height_proxy_double_support_fraction=float(labels.all(axis=1).mean()),
        height_proxy_no_support_fraction=float((~labels.any(axis=1)).mean()),
        lowest_foot_support_proxy=support_diagnostics(heights, bottom_speeds, fps),
        near_ground_bottom_speed_mean_m_s=float(np.mean(speeds)) if len(speeds) else None,
        near_ground_bottom_speed_p95_m_s=float(np.percentile(speeds, 95)) if len(speeds) else None)


def motion_gait(motion, model, *, start_seconds=0):
    """Use identical sphere-bottom geometry/velocity definitions on both inputs."""
    import mujoco

    fps = float(motion["fps"])
    start = int(round(start_seconds * fps))
    names = motion["body_names"].tolist()
    heights, speeds = [], []
    for name in ["left_ankle_roll_link", "right_ankle_roll_link"]:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0 or name not in names:
            raise ValueError(f"Missing foot {name}")
        geoms = np.flatnonzero((model.geom_bodyid == body) &
                              (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) &
                              ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        if not len(geoms):
            raise ValueError("Gait analysis requires G1 foot collision spheres")
        index = names.index(name)
        origin = motion["body_pos_w"][start:, index]
        rotation = Rotation.from_quat(motion["body_quat_w"][start:, index][:, [1, 2, 3, 0]])
        centres = np.stack([origin + rotation.apply(np.tile(model.geom_pos[g], (len(origin), 1)))
                            for g in geoms], axis=1)
        bottoms = centres.copy()
        bottoms[:, :, 2] -= model.geom_size[geoms, 0]
        lowest = np.argmin(bottoms[:, :, 2], axis=1)
        points = bottoms[np.arange(len(origin)), lowest]
        com = origin + rotation.apply(np.tile(model.body_ipos[body], (len(origin), 1)))
        velocity = point_velocity(motion["body_lin_vel_w"][start:, index],
                                  motion["body_ang_vel_w"][start:, index], points, com)
        heights.append(points[:, 2])
        speeds.append(np.linalg.norm(velocity[:, :2], axis=1))
    root = names.index("pelvis")
    result = summarize_gait(np.stack(heights, axis=1), np.stack(speeds, axis=1),
                            motion["body_pos_w"][start:, root, :2], fps)
    result["hip_periodicity"] = hip_periodicity(motion["joint_pos"][start:],
                                                motion["joint_names"].tolist(), fps)
    return result


def capture_rollout(env, policy, seconds, command):
    """Capture a diagnostic native-MuJoCo run at the contract control frequency."""
    from .runtime import rotate_inverse

    model, data, mj = env.model, env.data, env.mj
    bodies = [i for i in range(1, model.nbody) if mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)]
    fields = {k: [] for k in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                              "body_lin_vel_w", "body_ang_vel_w"]}
    fallen = False
    for _ in range(int(seconds / env.spec.control_dt)):
        env.step(policy(env.observation(command)))
        mj.mj_forward(model, data)
        fields["joint_pos"].append(data.qpos[env.qadr].copy())
        fields["joint_vel"].append(data.qvel[env.vadr].copy())
        fields["body_pos_w"].append(data.xpos[bodies].copy())
        fields["body_quat_w"].append(data.xquat[bodies].copy())
        velocities = []
        for body in bodies:
            velocity = np.empty(6)
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body, velocity, 0)
            velocities.append(velocity)
        velocities = np.asarray(velocities)
        fields["body_lin_vel_w"].append(velocities[:, 3:])
        fields["body_ang_vel_w"].append(velocities[:, :3])
        gravity = rotate_inverse(data.xquat[env.root_id], [0, 0, -1])
        fallen = bool(data.xpos[env.root_id, 2] < 0.45 or gravity[2] > -0.5)
        if fallen:
            break
    return dict(fps=np.float64(1 / env.spec.control_dt),
                joint_names=np.asarray(env.spec.joint_names),
                body_names=np.asarray([mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in bodies]),
                **{k: np.asarray(v) for k, v in fields.items()}), fallen
