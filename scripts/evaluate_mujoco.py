"""Evaluate a single policy under a fixed command/yaw matrix, recording all failures."""

import argparse
import json
from pathlib import Path

import numpy as np
from amp_lab.bundle import verify_bundle
from amp_lab.evaluation import command_yaw_cases
from amp_lab.mujoco_backend import MujocoBackend
from amp_lab.paths import contained_path
from amp_lab.runtime import OnnxPolicy

parser = argparse.ArgumentParser()
parser.add_argument("bundle", type=Path)
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
spec = verify_bundle(args.bundle)
policy = OnnxPolicy(contained_path(args.bundle, spec.policy_file), spec)
env = MujocoBackend(contained_path(args.bundle, spec.model_xml), spec)
reports = []
for command, yaw in command_yaw_cases():
    env.reset(yaw=yaw)
    result = env.run(policy, args.seconds, np.asarray(command), headless=True)
    reports.append(dict(result, initial_yaw=yaw))
    print(json.dumps(reports[-1]), flush=True)
result = {"schema_version": 1, "purpose": "fixed_command_initial_yaw_matrix",
          "policy_sha256": policy.sha256, "cases": reports,
          "completed_fraction": float(np.mean([r["completed"] and not r["fallen"] for r in reports]))}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
