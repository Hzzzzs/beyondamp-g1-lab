"""Evaluate saved checkpoints on CPU while Isaac training continues on the GPU."""

import argparse
import json
import time
from pathlib import Path

import torch
from amp_lab.export import export_checkpoint
from amp_lab.motion import sha256
from amp_lab.mujoco_backend import MujocoBackend
from amp_lab.paths import ASSETS, G1_XML
from amp_lab.provenance import asset_manifest
from amp_lab.runtime import OnnxPolicy
from amp_lab.spec import DeploymentSpec

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--until", type=int, required=True, help="Stop after evaluating this final iteration")
parser.add_argument("--timeout", type=float, default=3600, help="Maximum watcher lifetime in seconds")
args = parser.parse_args()
if args.until < 1 or args.timeout <= 0:
    parser.error("--until and --timeout must be positive")
torch.set_num_threads(1)
deadline = time.monotonic() + args.timeout
spec_path = args.run / "deployment.yaml"
while not spec_path.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError(f"No training deployment contract: {spec_path}")
    time.sleep(3)
spec = DeploymentSpec.load(spec_path)
if asset_manifest(ASSETS / "unitree/unitree_g1")["tree_sha256"] != spec.provenance["model_tree_sha256"]:
    raise ValueError("Evaluation assets differ from the actual training model")
backend = MujocoBackend(G1_XML, spec)
while True:
    checkpoints = sorted(args.run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    for checkpoint in checkpoints:
        iteration = int(checkpoint.stem.split("_")[-1])
        if iteration > args.until:
            continue
        output = args.run / "evaluations" / checkpoint.stem
        report_path = output / "mujoco.json"
        if not report_path.exists():
            policy_path = output / "policy.onnx"
            if policy_path.exists():
                exported = json.loads(policy_path.with_suffix(".export.json").read_text())
                if (exported["checkpoint_sha256"] != sha256(checkpoint)
                        or exported["policy_sha256"] != sha256(policy_path)):
                    raise ValueError(f"Existing export differs from checkpoint: {checkpoint}")
            else:
                exported = export_checkpoint(checkpoint, spec_path, policy_path)
            backend.reset()
            report = backend.run(OnnxPolicy(policy_path, spec), 20, [0.6, 0, 0])
            report.update(iteration=iteration, checkpoint_sha256=exported["checkpoint_sha256"])
            report_path.write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        if iteration == args.until:
            raise SystemExit(0)
    if time.monotonic() >= deadline:
        raise TimeoutError(f"Training did not produce model_{args.until}.pt before timeout")
    time.sleep(3)
