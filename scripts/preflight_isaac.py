"""Short GPU preflight with observation parity, before paying for a full training run."""

import argparse
import json
import platform
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--motions", type=Path, nargs="+", help="Also test training-only reference-state resets")
parser.add_argument("--amp-root-velocity", action="store_true")
parser.add_argument("--config", type=Path, help="Training configuration, including reward weights")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
try:
    import isaaclab
    import numpy as np
    import torch
    from amp_lab.runtime import ObservationBuilder
    from amp_lab.training.isaac_task import (
        G1AMPEnvCfg,
        IsaacAdapter,
        apply_training_config,
        ordered_joint_ids,
    )

    if platform.python_version_tuple()[:2] != ("3", "10") or isaaclab.__version__ != "0.36.21":
        raise RuntimeError("Expected Python 3.10 and IsaacLab v2.1.0 (extension version 0.36.21)")
    cfg = G1AMPEnvCfg()
    cfg.scene.num_envs, cfg.sim.device = 4, args.device
    cfg.seed = 42
    apply_training_config(cfg, json.loads(args.config.read_text()) if args.config else {})
    if args.amp_root_velocity:
        cfg.amp_root_velocity = True
    env = IsaacAdapter(cfg)
    try:
        reference_report = {}
        if args.motions:
            from amp_lab.training.dataset import JointMotionDataset

            dataset = JointMotionDataset(args.motions, env.spec, env.device, root_velocity=cfg.amp_root_velocity,
                                         joint_velocity_scale=cfg.amp_joint_velocity_scale)
            env.enable_reference_resets(dataset, 1.0)
            robot = env.env.scene["robot"]
            positions = robot.data.joint_pos[:, ordered_joint_ids(env.env)]
            distance = (positions[:, None] - dataset.reference_joint_pos[None]).abs().amax(-1)
            frames = distance.argmin(-1)
            torch.testing.assert_close(env.observations()[2], dataset.states[frames], atol=1e-5, rtol=1e-5)
            joint_error = distance.amin(-1).max().item()
            torch.testing.assert_close(robot.data.joint_vel[:, ordered_joint_ids(env.env)],
                                       dataset.reference_joint_vel[frames])
            speed_error = (robot.data.root_lin_vel_w.norm(dim=-1) -
                           dataset.reference_root_states[frames, 7:10].norm(dim=-1)).abs().max().item()
            if joint_error > 1e-5 or speed_error > 1e-5:
                raise AssertionError("Reference reset does not match named motion state")
            reference_report = {"reference_reset_joint_max_error": joint_error,
                                "reference_reset_speed_norm_max_error": speed_error}
        data = env.env.scene["robot"].data
        ids = ordered_joint_ids(env.env)
        obs = env.observations()[0][0].cpu().numpy()
        builder = ObservationBuilder(env.spec)
        reference = builder.build(
            quat_wxyz=data.root_quat_w[0].cpu().numpy(), ang_vel_b=data.root_ang_vel_b[0].cpu().numpy(),
            joint_pos=data.joint_pos[0, ids].cpu().numpy(), joint_vel=data.joint_vel[0, ids].cpu().numpy(),
            command=env.env.command_manager.get_command("base_velocity")[0].cpu().numpy())
        np.testing.assert_allclose(obs, reference, atol=1e-6, rtol=1e-5)
        for _ in range(10):
            actor, critic, reward, _, _, amp = env.step(torch.zeros((4, 29), device=env.device))
            for tensor in (actor, critic, reward, amp):
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError("Nonfinite GPU rollout")
        # Force a safe midair 90-degree roll so the next step terminates by orientation.
        # This exercises capture-before-autoreset without deep ground penetration.
        robot = env.env.scene["robot"]
        root = robot.data.root_state_w.clone()
        root[:, 3:7] = torch.tensor([2**-0.5, 2**-0.5, 0, 0], device=env.device)
        root[:, 7:] = 0
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos + 0.01,
                                        torch.zeros_like(robot.data.default_joint_vel))
        _, _, _, done, info, terminal_amp = env.step(torch.zeros((4, 29), device=env.device))
        if not done.all() or env.env._terminal_ids is None:
            raise AssertionError("Forced orientation failure did not exercise autoreset capture")
        if torch.max(torch.abs(terminal_amp - env.observations()[2])).item() < 1e-5:
            raise AssertionError("Terminal AMP state was replaced by reset observations")
        durations = info["completed_episode_seconds"]
        if len(durations) != 4 or min(durations) <= 0:
            raise AssertionError("Completed episode duration was read after autoreset")
        if not np.isfinite(list(info["rollout_metrics"].values())).all():
            raise FloatingPointError("Nonfinite GPU rollout metrics")
        report = {"status": "passed", "python": platform.python_version(), "torch": torch.__version__,
                  **reference_report,
                  "isaaclab_extension": isaaclab.__version__, "target": env.backend,
                  "runtime_versions": env.runtime_versions,
                  "device": str(env.device), "num_envs": 4, "control_steps": 10,
                  "initial_observation_max_error": float(np.abs(obs - reference).max()),
                  "terminal_amp_capture_before_reset": True,
                  "terminal_episode_durations_before_reset": durations,
                  "amp_observation_dim": terminal_amp.shape[-1],
                  "walking_policy_validated": False}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        env.close()
finally:
    launcher.app.close()
