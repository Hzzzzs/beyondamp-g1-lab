"""G1 flat-ground AMP task. Import only AFTER AppLauncher starts Isaac Sim 4.5."""

from dataclasses import asdict

import isaaclab.sim as sim_utils

# robotlib's default asset path assumes its original parent monorepo layout.
import robotlib
import torch
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

from amp_lab.g1 import JOINT_NAMES, default_spec
from amp_lab.paths import ASSETS
from amp_lab.provenance import asset_manifest

robotlib.ROBOTLIB_ASSETLIB_DIR = str(ASSETS)
from robotlib.beyondMimic.robots.g1 import G1_OPENSOURCE_CFG  # noqa: E402

PROFILE = default_spec()


def ordered_joint_ids(env):
    if not hasattr(env, "_amp_joint_ids"):
        env._amp_joint_ids = env.scene["robot"].find_joints(JOINT_NAMES, preserve_order=True)[0]
    return env._amp_joint_ids


def amp_observation(env):
    data, ids = env.scene["robot"].data, ordered_joint_ids(env)
    terms = [data.joint_pos[:, ids] - data.default_joint_pos[:, ids],
             data.joint_vel[:, ids] * env.cfg.amp_joint_velocity_scale]
    if env.cfg.amp_root_velocity:
        terms.append(data.root_lin_vel_b)
    return torch.cat(terms, dim=-1)


def policy_observation(env):
    data, ids = env.scene["robot"].data, ordered_joint_ids(env)
    return torch.cat([
        data.root_ang_vel_b.clamp(-100, 100) * 0.2,
        data.projected_gravity_b,
        env.command_manager.get_command("base_velocity"),
        (data.joint_pos[:, ids] - data.default_joint_pos[:, ids]).clamp(-100, 100),
        data.joint_vel[:, ids].clamp(-100, 100) * 0.05,
        env.action_manager.action.clamp(-12, 12),
    ], dim=-1)


def critic_observation(env):
    return torch.cat([env.scene["robot"].data.root_lin_vel_b, policy_observation(env)], dim=-1)


def velocity_error_xy(env):
    return (env.command_manager.get_command("base_velocity")[:, :2] -
            env.scene["robot"].data.root_lin_vel_b[:, :2]).square().sum(-1)


def apply_training_config(cfg, config):
    cfg.amp_root_velocity = config.get("amp_root_velocity", False)
    cfg.amp_joint_velocity_scale = config.get("amp_joint_velocity_scale", 0.05)
    for name, weight in config.get("reward_weights", {}).items():
        getattr(cfg.rewards, name).weight = float(weight)


def robot_config():
    config = G1_OPENSOURCE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Explicit PD, recomputed on every physics step, matches the MuJoCo controller.
    config.spawn.articulation_props.enabled_self_collisions = False
    config.spawn.replace_cylinders_with_capsules = False
    config.actuators = {"all": IdealPDActuatorCfg(
        joint_names_expr=[".*"],
        stiffness=dict(zip(JOINT_NAMES, PROFILE.stiffness)),
        damping=dict(zip(JOINT_NAMES, PROFILE.damping)),
        effort_limit=dict(zip(JOINT_NAMES, PROFILE.effort_limit)),
        effort_limit_sim=dict(zip(JOINT_NAMES, PROFILE.effort_limit)),
        armature=dict(zip(JOINT_NAMES, PROFILE.armature)), friction=0.0, velocity_limit_sim=1000.0,
    )}
    return config


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    robot = robot_config()
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(5.0, 10.0), rel_standing_envs=0.05,
        heading_command=False, debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.9), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.3, 0.3)),
    )


@configclass
class ActionsCfg:
    joints = mdp.JointPositionActionCfg(asset_name="robot", joint_names=JOINT_NAMES,
                                       preserve_order=True, scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObservationGroupCfg):
        state = ObservationTermCfg(func=policy_observation)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class Critic(ObservationGroupCfg):
        state = ObservationTermCfg(func=critic_observation)
        enable_corruption = False
        concatenate_terms = True

    policy = Policy()
    critic = Critic()


@configclass
class EventsCfg:
    root = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset", params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
    })
    joints = EventTermCfg(func=mdp.reset_joints_by_scale, mode="reset", params={
        "position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0),
    })


@configclass
class RewardsCfg:
    velocity = RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=2.0,
                             params={"command_name": "base_velocity", "std": 0.5})
    velocity_error = RewardTermCfg(func=velocity_error_xy, weight=0.0)
    yaw = RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=1.0,
                        params={"command_name": "base_velocity", "std": 0.5})
    alive = RewardTermCfg(func=mdp.is_alive, weight=0.2)
    vertical_velocity = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    angular_velocity = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02)
    joint_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0)
    acceleration = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)


@configclass
class TerminationsCfg:
    timeout = TerminationTermCfg(func=mdp.time_out, time_out=True)
    height = TerminationTermCfg(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45})
    orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 1.0471975512})


@configclass
class G1AMPEnvCfg(ManagerBasedRLEnvCfg):
    amp_root_velocity: bool = False
    amp_joint_velocity_scale: float = 0.05
    scene = SceneCfg(num_envs=1024, env_spacing=2.5)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.sim.physics_material = self.scene.terrain.physics_material


class TerminalAMPEnv(ManagerBasedRLEnv):
    def step(self, action):
        self._capturing_terminal_amp = True
        self._terminal_ids = None
        self._terminal_amp = None
        self._terminal_critic = None
        self._terminal_velocity = None
        self._terminal_commands = None
        self._terminal_lengths = None
        try:
            return super().step(action)
        finally:
            self._capturing_terminal_amp = False

    def _reset_idx(self, env_ids):
        if getattr(self, "_capturing_terminal_amp", False):
            self._terminal_ids = env_ids.clone()
            self._terminal_amp = amp_observation(self)[env_ids].clone()
            self._terminal_critic = critic_observation(self)[env_ids].clone()
            self._terminal_velocity = self.scene["robot"].data.root_lin_vel_b[env_ids].clone()
            self._terminal_commands = self.command_manager.get_command("base_velocity")[env_ids].clone()
            self._terminal_lengths = self.episode_length_buf[env_ids].clone()
        super()._reset_idx(env_ids)
        dataset = getattr(self, "_reference_dataset", None)
        if dataset is not None:
            selected = env_ids[torch.rand(len(env_ids), device=self.device) < self._reference_probability]
            if len(selected):
                robot = self.scene["robot"]
                frames = torch.randint(len(dataset.reference_root_states), (len(selected),), device=self.device)
                reference = dataset.reference_root_states[frames]
                root = robot.data.root_state_w[selected].clone()
                # Retain reset XY and random heading, rotate reference orientation AND world velocities.
                heading = root[:, 3:7].clone()
                root[:, 2] = reference[:, 2] + self.scene.env_origins[selected, 2]
                root[:, 3:7] = quat_mul(heading, reference[:, 3:7])
                root[:, 7:10] = quat_apply(heading, reference[:, 7:10])
                root[:, 10:13] = quat_apply(heading, reference[:, 10:13])
                robot.write_root_state_to_sim(root, env_ids=selected)
                robot.write_joint_state_to_sim(dataset.reference_joint_pos[frames],
                                              dataset.reference_joint_vel[frames],
                                              joint_ids=ordered_joint_ids(self), env_ids=selected)


class IsaacAdapter:
    backend = "isaaclab_2.1.0_isaacsim_4.5.0"

    def __init__(self, cfg, render_mode=None):
        from .runtime_check import check_runtime
        self.runtime_versions = check_runtime()
        self.env = TerminalAMPEnv(cfg, render_mode=render_mode)
        self.num_envs, self.device = self.env.num_envs, self.env.device
        self.num_obs, self.num_critic_obs, self.num_actions = 96, 99, 29
        self.spec = self.live_spec()
        self.deployment_spec = asdict(self.spec)
        self._obs, _ = self.env.reset()
        actor, critic, _ = self.observations()
        if actor.shape != (self.num_envs, 96) or critic.shape != (self.num_envs, 99):
            raise ValueError("Live observation dimensions disagree with deployment contract")

    def live_spec(self):
        robot = self.env.scene["robot"]
        spec = default_spec()
        ids = ordered_joint_ids(self.env)
        if set(robot.joint_names) != set(spec.joint_names):
            raise ValueError("Unexpected G1 model joint set")
        action_cfg = self.env.cfg.actions.joints
        if action_cfg.joint_names != JOINT_NAMES or not action_cfg.preserve_order or not action_cfg.use_default_offset:
            raise ValueError("Task action order/offset differs from the supported observation contract")
        if not isinstance(action_cfg.scale, (int, float)):
            raise ValueError("This task exporter currently requires a scalar action scale")
        spec.action_scale = [float(action_cfg.scale)] * len(spec.joint_names)
        spec.default_root_height = float(self.env.cfg.scene.robot.init_state.pos[2])
        spec.self_collisions = bool(self.env.cfg.scene.robot.spawn.articulation_props.enabled_self_collisions)
        spec.ground_friction = float(self.env.cfg.scene.terrain.physics_material.dynamic_friction)
        spec.default_joint_pos = robot.data.default_joint_pos[0, ids].cpu().tolist()
        actuator = robot.actuators["all"]
        names = [robot.joint_names[i] for i in range(robot.num_joints)]
        local_ids = [names.index(name) for name in spec.joint_names]
        for key in ("stiffness", "damping", "effort_limit", "armature"):
            setattr(spec, key, getattr(actuator, key)[0, local_ids].cpu().tolist())
        spec.simulation_dt, spec.control_decimation = self.env.physics_dt, self.env.cfg.decimation
        spec.provenance = {"status": "exported_from_live_training_environment", "backend": self.backend,
                           "training_asset": "unitree/unitree_g1/urdf/g1_29dof.urdf",
                           "model_tree_sha256": asset_manifest(ASSETS / "unitree/unitree_g1")["tree_sha256"]}
        return spec.validate()

    def observations(self):
        # Consume observations returned by step/reset once; do not advance noise/history by recomputing.
        return self._obs["policy"], self._obs["critic"], amp_observation(self.env)

    def enable_reference_resets(self, dataset, probability):
        """Training-only RSI; keep some neutral starts to match ordinary deployment."""
        if not 0 <= probability <= 1:
            raise ValueError("reference_state_initialization must be a probability in [0, 1]")
        self.env._reference_dataset = dataset
        self.env._reference_probability = probability
        self._obs, _ = self.env.reset()

    def step(self, actions):
        self._obs, reward, terminated, truncated, extras = self.env.step(actions.clamp(-12, 12))
        terminal = amp_observation(self.env).clone()
        terminal_critic = self._obs["critic"].clone()
        velocity = self.env.scene["robot"].data.root_lin_vel_b.clone()
        commands = self.env.command_manager.get_command("base_velocity").clone()
        episode_seconds = []
        if self.env._terminal_ids is not None:
            terminal[self.env._terminal_ids] = self.env._terminal_amp
            terminal_critic[self.env._terminal_ids] = self.env._terminal_critic
            velocity[self.env._terminal_ids] = self.env._terminal_velocity
            commands[self.env._terminal_ids] = self.env._terminal_commands
            episode_seconds = (self.env._terminal_lengths * self.env.step_dt).cpu().tolist()
        actor, critic, _ = self.observations()
        info = {"terminal_critic": terminal_critic, "timeout_mask": truncated & ~terminated,
                "completed_episode_seconds": episode_seconds,
                "rollout_metrics": {
                    "velocity_mse_xy": (velocity[:, :2] - commands[:, :2]).square().mean().item(),
                    "forward_velocity": velocity[:, 0].mean().item(),
                    "forward_command": commands[:, 0].mean().item(),
                    "termination_fraction": terminated.float().mean().item(),
                }}
        return actor, critic, reward / self.env.step_dt, terminated | truncated, info, terminal

    def close(self):
        self.env.close()
