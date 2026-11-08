"""Compare phase-free reference/policy gait descriptors; not an imitation percentage."""

import argparse
import json
from pathlib import Path

import numpy as np
from amp_lab.bundle import verify_bundle
from amp_lab.evaluation import spec_fingerprint
from amp_lab.gait import capture_rollout, motion_gait
from amp_lab.motion import load_motion, sha256
from amp_lab.mujoco_backend import MujocoBackend
from amp_lab.paths import contained_path
from amp_lab.runtime import OnnxPolicy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bundle", type=Path)
parser.add_argument("--references", type=Path, nargs="+", required=True)
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--warmup", type=float, default=2)
parser.add_argument("--speeds", type=float, nargs="+", default=[0.6, 0.9])
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
if not (np.isfinite(args.seconds) and 0 <= args.warmup < args.seconds - 0.1):
    parser.error("Seconds must be finite and exceed nonnegative warmup by 0.1s")
if not np.isfinite(args.speeds).all():
    parser.error("Speeds must be finite")
spec = verify_bundle(args.bundle)
policy = OnnxPolicy(contained_path(args.bundle, spec.policy_file), spec)
env = MujocoBackend(contained_path(args.bundle, spec.model_xml), spec)
references = [dict(path=str(p), sha256=sha256(p), metrics=motion_gait(load_motion(p), env.model))
              for p in args.references]
rollouts = []
args.output.parent.mkdir(parents=True, exist_ok=True)
for index, speed in enumerate(args.speeds):
    env.reset()
    motion, fallen = capture_rollout(env, policy, args.seconds, np.array([speed, 0, 0]))
    trace = args.output.with_name(args.output.stem + f"_rollout_{index}.npz")
    np.savez_compressed(trace, **motion)
    enough = len(motion["joint_pos"]) > int(args.warmup / spec.control_dt) + 1
    rollouts.append(dict(command=[speed, 0, 0], fallen=fallen,
                         simulated_seconds=len(motion["joint_pos"]) * spec.control_dt,
                         trace=str(trace), trace_sha256=sha256(trace),
                         metrics=motion_gait(motion, env.model, start_seconds=args.warmup) if enough else None))
result = dict(schema_version=1, purpose="phase_free_gait_descriptors",
              policy_sha256=policy.sha256, deployment_sha256=spec_fingerprint(spec),
              warmup_excluded_policy_seconds=args.warmup, references=references, rollouts=rollouts,
              definitions={
                  "support": "sphere bottom <1.5cm enters, >3.5cm leaves; sustained 0.06s; height proxy",
                  "cadence": "120 / mean same-foot touchdown interval in seconds, complete strides only",
                  "hip_periodicity": "hip pitch peak intervals, prominence=max(0.05rad,20% range), "
                                     "min gap0.25s; assumes two steps/cycle",
                  "stride": "root XY displacement between same-foot touchdowns; not step length",
                  "swing_clearance": "mean peak lowest collision-sphere bottom Z per complete swing",
                  "slip_proxy": "lowest sphere-bottom horizontal point speed when bottom<2.5cm",
              },
              limitations=["No measured contact force; reference is kinematic, policy is dynamic",
                           "Low-clearance shuffling can have no height-detected cycles; consult hip periodicity",
                           "Reference clips are short and move at different speeds; compare speeds explicitly",
                           "No phase alignment, calibrated similarity percentage or automatic style acceptance"])
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
