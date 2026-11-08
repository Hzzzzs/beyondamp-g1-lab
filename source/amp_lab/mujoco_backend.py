"""Name-based MuJoCo state/control adapter; no positional joint-order assumptions."""

import time

import numpy as np

from .runtime import ObservationBuilder, rotate_inverse
from .spec import DeploymentSpec


class MujocoBackend:
    def __init__(self, xml_path, spec: DeploymentSpec):
        import mujoco

        self.mj = mujoco
        self.spec = spec.validate()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = spec.simulation_dt
        self.data = mujoco.MjData(self.model)
        self.builder = ObservationBuilder(spec)
        self.joint_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in spec.joint_names])
        if np.any(self.model.jnt_type[self.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Only hinge policy joints are supported")
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        self.root_id = self._id(mujoco.mjtObj.mjOBJ_BODY, spec.root_body)
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or self.model.jnt_bodyid[free[0]] != self.root_id:
            raise ValueError("Expected a single free joint attached to root_body")
        self.free_qadr = int(self.model.jnt_qposadr[free[0]])
        actuator_ids = []
        for joint in self.joint_ids:
            candidates = np.flatnonzero((self.model.actuator_trnid[:, 0] == joint) &
                                       (self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
            if len(candidates) != 1:
                raise ValueError(f"Expected one direct actuator for joint {joint}")
            a = int(candidates[0])
            if (self.model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_NONE or
                    self.model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE or
                    self.model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED or
                    self.model.actuator_gainprm[a, 0] != 1 or
                    not np.allclose(self.model.actuator_gear[a], [1, 0, 0, 0, 0, 0])):
                raise ValueError("Deployment requires unit-gain direct torque motors")
            actuator_ids.append(a)
        self.actuator_ids = np.asarray(actuator_ids)
        if self.model.nu != len(actuator_ids):
            raise ValueError("Unmapped actuators in robot model")
        # Explicit PD is evaluated every physics step. Remove the MJCF's legacy passive joint terms.
        self.model.dof_damping[self.vadr] = 0
        self.model.dof_frictionloss[self.vadr] = 0
        if spec.armature:
            self.model.dof_armature[self.vadr] = spec.armature
        ground = (self.model.geom_bodyid == 0) & (self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        collision = ((self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0))
        if not spec.self_collisions:
            robot_collision = collision & (self.model.geom_bodyid != 0)
            self.model.geom_contype[robot_collision] = 1
            self.model.geom_conaffinity[robot_collision] = 0
            self.model.geom_contype[ground] = 0
            self.model.geom_conaffinity[ground] = 1
        self.model.geom_friction[collision | ground, 0] = spec.ground_friction
        self.reset()

    def _id(self, kind, name):
        result = self.mj.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"Missing robot element: {name}")
        return result

    def reset(self, *, yaw=0.0):
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.spec.default_joint_pos
        q = self.free_qadr
        self.data.qpos[q:q+3] = [0, 0, self.spec.default_root_height]
        self.data.qpos[q+3:q+7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.target = np.asarray(self.spec.default_joint_pos, dtype=float)
        self.builder.reset()
        self.mj.mj_forward(self.model, self.data)

    def state(self):
        velocity = np.empty(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY,
                                 self.root_id, velocity, 0)
        quat = self.data.xquat[self.root_id].copy()
        return dict(quat_wxyz=quat, ang_vel_b=rotate_inverse(quat, velocity[:3]),
                    lin_vel_b=rotate_inverse(quat, velocity[3:]),
                    joint_pos=self.data.qpos[self.qadr].copy(),
                    joint_vel=self.data.qvel[self.vadr].copy())

    def observation(self, command):
        return self.builder.build(command=command, **self.state())

    def step(self, action):
        self.target = self.builder.action_target(action)
        for _ in range(self.spec.control_decimation):
            torque = (np.asarray(self.spec.stiffness) * (self.target - self.data.qpos[self.qadr]) -
                      np.asarray(self.spec.damping) * self.data.qvel[self.vadr])
            torque = np.clip(torque, -np.asarray(self.spec.effort_limit), self.spec.effort_limit)
            self.data.ctrl[self.actuator_ids] = torque
            self.mj.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Simulation produced non-finite state")

    def run(self, policy, duration, command, *, headless=True, video=None):
        import contextlib

        if not np.isfinite(duration) or duration < self.spec.control_dt:
            raise ValueError("duration must cover at least one control step")
        command = np.asarray(command, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("command must contain finite vx, vy, yaw_rate")

        viewer = None
        if not headless:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(self.model, self.data)
        errors, yaw_errors = [], []
        fallen = False
        steps = 0
        with contextlib.ExitStack() as stack:
            if viewer:
                stack.enter_context(viewer)
            renderer, writer, camera = None, None, None
            if video:
                import imageio.v2 as imageio
                video.parent.mkdir(parents=True, exist_ok=True)
                self.model.vis.global_.offwidth, self.model.vis.global_.offheight = 960, 720
                renderer = self.mj.Renderer(self.model, 720, 960)
                stack.callback(renderer.close)
                writer = imageio.get_writer(str(video), fps=1 / self.spec.control_dt)
                stack.callback(writer.close)
                camera = self.mj.MjvCamera()
                camera.distance, camera.elevation, camera.azimuth = 2.5, -15, 135
            for _ in range(int(duration / self.spec.control_dt)):
                started = time.monotonic()
                if viewer and not viewer.is_running():
                    break
                self.step(policy(self.observation(command)))
                state = self.state()
                errors.append(np.asarray(state["lin_vel_b"][:2]) - command[:2])
                yaw_errors.append(state["ang_vel_b"][2] - command[2])
                steps += 1
                gravity = rotate_inverse(state["quat_wxyz"], [0, 0, -1])
                fallen = self.data.xpos[self.root_id, 2] < 0.45 or gravity[2] > -0.5
                if renderer:
                    camera.lookat[:] = self.data.xpos[self.root_id]
                    renderer.update_scene(self.data, camera)
                    writer.append_data(renderer.render())
                if viewer:
                    viewer.sync()
                    time.sleep(max(0, self.spec.control_dt - (time.monotonic() - started)))
                if fallen:
                    break
        from .evaluation import evidence, tracking_metrics
        return dict(evidence(self.spec, policy, "mujoco", command),
                    **tracking_metrics(errors, yaw_errors, command),
                    mujoco_version=self.mj.__version__,
                    completed_seconds=steps * self.spec.control_dt, requested_seconds=duration,
                    fallen=bool(fallen), completed=bool(steps == int(duration / self.spec.control_dt)))
