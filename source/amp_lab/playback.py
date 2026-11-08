"""Kinematic reference visualization (never confused with policy inference)."""

import time

import numpy as np

from .motion import load_motion


def play_motion(path, xml, video=None):
    import mujoco

    motion = load_motion(path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in motion["joint_names"]]
    if any(i < 0 for i in ids):
        raise ValueError("Unknown joints in motion")
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Expected one free joint")
    root_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[free[0]])
    anchor = motion["body_names"].tolist().index(root_name)
    root_qadr = model.jnt_qposadr[free[0]]
    qadr = model.jnt_qposadr[ids]
    viewer, renderer, writer = None, None, None
    try:
        if video:
            import imageio.v2 as imageio
            video.parent.mkdir(parents=True, exist_ok=True)
            model.vis.global_.offwidth = 960
            model.vis.global_.offheight = 720
            writer = imageio.get_writer(str(video), fps=float(motion["fps"]))
            renderer = mujoco.Renderer(model, height=720, width=960)
            camera = mujoco.MjvCamera()
            camera.distance, camera.elevation, camera.azimuth = 2.5, -15, 135
        else:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(model, data)
        for frame in range(len(motion["joint_pos"])):
            start = time.monotonic()
            data.qpos[qadr] = motion["joint_pos"][frame]
            data.qpos[root_qadr:root_qadr+3] = motion["body_pos_w"][frame, anchor]
            data.qpos[root_qadr+3:root_qadr+7] = motion["body_quat_w"][frame, anchor]
            mujoco.mj_forward(model, data)
            if renderer:
                camera.lookat[:] = data.qpos[root_qadr:root_qadr+3]
                renderer.update_scene(data, camera)
                writer.append_data(renderer.render())
            else:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0, 1 / float(motion["fps"]) - (time.monotonic() - start)))
    finally:
        if viewer:
            viewer.close()
        if renderer:
            renderer.close()
        if writer:
            writer.close()
