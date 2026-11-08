"""Evaluate the same command/yaw matrix as MuJoCo, in parallel Isaac environments."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("bundle", type=Path)
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
try:
    import numpy as np
    import torch
    from amp_lab.bundle import verify_bundle
    from amp_lab.evaluation import command_yaw_cases, evidence, tracking_metrics
    from amp_lab.paths import contained_path
    from amp_lab.runtime import OnnxPolicy
    from amp_lab.training.isaac_task import G1AMPEnvCfg, IsaacAdapter

    if args.output.exists():
        raise FileExistsError(args.output)
    spec = verify_bundle(args.bundle)
    if not np.isfinite(args.seconds) or args.seconds < spec.control_dt:
        raise ValueError("Duration must cover at least one control step")
    policy = OnnxPolicy(contained_path(args.bundle, spec.policy_file), spec)
    cases = command_yaw_cases()
    cfg = G1AMPEnvCfg()
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = len(cases), args.device, 42
    cfg.events.root.params["pose_range"] = {}
    cfg.commands.base_velocity.rel_standing_envs = 0
    cfg.commands.base_velocity.resampling_time_range = (args.seconds + 1, args.seconds + 1)
    cfg.episode_length_s = args.seconds + 1
    env = IsaacAdapter(cfg)
    try:
        if asdict(env.spec) != asdict(spec):
            raise ValueError("Current Isaac environment differs from bundled training contract")
        robot = env.env.scene["robot"]
        roots = robot.data.root_state_w.clone()
        yaw = torch.tensor([case[1] for case in cases], device=env.device)
        roots[:, 3:7] = 0
        roots[:, 3], roots[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        roots[:, 7:] = 0
        robot.write_root_state_to_sim(roots)
        commands = torch.tensor([case[0] for case in cases], device=env.device)
        env.env.command_manager.get_term("base_velocity").command.copy_(commands)
        env._obs = env.env.observation_manager.compute()
        errors, yaw_errors = [[] for _ in cases], [[] for _ in cases]
        active = np.ones(len(cases), dtype=bool)
        fallen = np.zeros(len(cases), dtype=bool)
        completed_steps = np.zeros(len(cases), dtype=int)
        total_steps = int(args.seconds / spec.control_dt)
        for _ in range(total_steps):
            observations = env.observations()[0].cpu().numpy()
            actions = np.stack([policy(observation) for observation in observations])
            _, _, _, dones, _, _ = env.step(torch.from_numpy(actions).to(env.device))
            done = dones.cpu().numpy()
            velocity = robot.data.root_lin_vel_b.cpu().numpy()
            angular = robot.data.root_ang_vel_b[:, 2].cpu().numpy()
            for i, (command, _) in enumerate(cases):
                if not active[i]:
                    continue
                completed_steps[i] += 1
                if done[i]:
                    # Autoreset state belongs to a new episode; exclude it from tracking.
                    fallen[i], active[i] = True, False
                    continue
                errors[i].append(velocity[i, :2] - command[:2])
                yaw_errors[i].append(angular[i] - command[2])
            if not active.any():
                break
        reports = []
        for i, (command, yaw) in enumerate(cases):
            report = dict(evidence(spec, policy, "isaaclab", command), runtime=env.backend,
                          **tracking_metrics(errors[i], yaw_errors[i], command), initial_yaw=yaw,
                          completed_seconds=int(completed_steps[i]) * spec.control_dt,
                          requested_seconds=args.seconds, fallen=bool(fallen[i]),
                          completed=bool(completed_steps[i] == total_steps))
            reports.append(report)
            print(json.dumps(report), flush=True)
        result = dict(schema_version=1, purpose="fixed_command_initial_yaw_matrix",
                      policy_sha256=policy.sha256, cases=reports,
                      completed_fraction=float(np.mean([r["completed"] and not r["fallen"] for r in reports])))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    finally:
        env.close()
finally:
    launcher.app.close()
