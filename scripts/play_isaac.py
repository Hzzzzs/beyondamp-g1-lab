"""Run exactly the bundled ONNX actor in Isaac Sim, using the training observation/action contract."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("bundle", type=Path)
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--velocity", nargs=3, type=float, default=[0.6, 0, 0])
parser.add_argument("--report", type=Path)
parser.add_argument("--video", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
launcher = AppLauncher(args)
try:
    import numpy as np
    import torch
    from amp_lab.bundle import verify_bundle
    from amp_lab.evaluation import evidence, tracking_metrics
    from amp_lab.paths import contained_path
    from amp_lab.runtime import OnnxPolicy
    from amp_lab.training.isaac_task import G1AMPEnvCfg, IsaacAdapter

    if not np.isfinite(args.seconds) or args.seconds < 0.02 or not np.isfinite(args.velocity).all():
        raise ValueError("Expected positive duration and finite commands")
    spec = verify_bundle(args.bundle)
    policy = OnnxPolicy(contained_path(args.bundle, spec.policy_file), spec)
    cfg = G1AMPEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.eye = (2.5, 2.5, 1.4)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    cfg.viewer.resolution = (960, 720)
    cfg.events.root.params["pose_range"] = {}
    cfg.commands.base_velocity.rel_standing_envs = 0
    for name, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), args.velocity):
        setattr(cfg.commands.base_velocity.ranges, name, (value, value))
    cfg.episode_length_s = args.seconds + 1
    env = IsaacAdapter(cfg, render_mode="rgb_array" if args.video else None)
    writer = None
    try:
        if asdict(spec) != asdict(env.spec):
            raise ValueError("Current Isaac environment differs from bundled training contract")
        errors, yaw_errors, fallen = [], [], False
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=1 / spec.control_dt)
        steps = 0
        for _ in range(int(args.seconds / spec.control_dt)):
            obs = env.observations()[0][0].cpu().numpy()
            action = torch.from_numpy(policy(obs)[None]).to(env.device)
            _, _, _, done, _, _ = env.step(action)
            if writer:
                writer.append_data(env.env.render())
            steps += 1
            # Terminal state is captured before autoreset; evaluate tracking before resets only.
            if done[0]:
                fallen = True
                break
            state = env.env.scene["robot"].data
            errors.append(state.root_lin_vel_b[0, :2].cpu().numpy() - args.velocity[:2])
            yaw_errors.append(state.root_ang_vel_b[0, 2].item() - args.velocity[2])
        report = {**evidence(spec, policy, "isaaclab", args.velocity), "runtime": env.backend,
                  **tracking_metrics(errors, yaw_errors, args.velocity),
                  "completed_seconds": steps * spec.control_dt,
                  "requested_seconds": args.seconds, "fallen": fallen,
                  "completed": steps == int(args.seconds / spec.control_dt)}
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2))
    finally:
        if writer:
            writer.close()
        env.close()
finally:
    launcher.app.close()
