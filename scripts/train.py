"""Run with IsaacLab's Python after installing the pinned upstream packages."""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motions", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=Path)
parser.add_argument("--config", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

try:
    from amp_lab.spec import DeploymentSpec
    from amp_lab.training.dataset import JointMotionDataset
    from amp_lab.training.isaac_task import G1AMPEnvCfg, IsaacAdapter, apply_training_config
    from amp_lab.training.runner import DEFAULT_CONFIG, Runner, seed_all

    cfg = G1AMPEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.is_finite_horizon = True
    cfg.seed = args.seed
    train_cfg = json.loads(args.config.read_text()) if args.config else copy.deepcopy(DEFAULT_CONFIG)
    train_cfg["seed"] = args.seed
    apply_training_config(cfg, train_cfg)
    if args.iterations < 1 or args.num_envs < 2:
        raise ValueError("Require at least one iteration and two environments")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output directory or explicitly resume")
    seed_all(args.seed)
    env = IsaacAdapter(cfg)
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        spec_path = args.output / "deployment.yaml"
        if spec_path.exists() and asdict(DeploymentSpec.load(spec_path)) != asdict(env.spec):
            raise ValueError("Existing deployment.yaml differs from live environment; use a new experiment")
        env.spec.save(spec_path)
        dataset = JointMotionDataset(args.motions, env.spec, env.device, root_velocity=cfg.amp_root_velocity,
                                     joint_velocity_scale=cfg.amp_joint_velocity_scale)
        if train_cfg.get("reference_state_initialization", 0):
            env.enable_reference_resets(dataset, train_cfg["reference_state_initialization"])
        runner = Runner(env, dataset, args.output, train_cfg)
        if args.resume:
            runner.resume(args.resume)
        runner.learn(args.iterations)
    finally:
        env.close()
finally:
    launcher.app.close()
