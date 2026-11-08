"""Real G1 + expert data + beyondAMP update + resume + ONNX + bundle + MuJoCo, on CPU.

The output is labeled diagnostic throughout and is not the promised trained walking example.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from amp_lab.bundle import create_bundle, verify_bundle
from amp_lab.export import export_checkpoint
from amp_lab.mujoco_backend import MujocoBackend
from amp_lab.paths import ASSETS, contained_path
from amp_lab.runtime import OnnxPolicy
from amp_lab.training.cpu_smoke import CpuSmokeEnv
from amp_lab.training.dataset import JointMotionDataset
from amp_lab.training.runner import DEFAULT_CONFIG, Runner, seed_all

parser = argparse.ArgumentParser()
parser.add_argument("--motions", "--motion", dest="motions", nargs="+", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError("Use a new diagnostic output directory")
torch.set_num_threads(2)
seed_all(42)
env = CpuSmokeEnv()
args.output.mkdir(parents=True)
env.spec.save(args.output / "deployment.yaml")
data = JointMotionDataset(args.motions, env.spec, "cpu")
cfg = copy.deepcopy(DEFAULT_CONFIG)
cfg["policy"].update(actor_hidden_dims=[64, 32], critic_hidden_dims=[64, 32])
cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=2, amp_replay_buffer_size=256)
cfg.update(discriminator_hidden_dims=[64, 32], steps_per_env=8, save_interval=1)
runner = Runner(env, data, args.output / "train", cfg)
runner.learn(2)
resumed = Runner(CpuSmokeEnv(), data, args.output / "resume", cfg)
resumed.resume(args.output / "train/model_2.pt")
resumed.learn(3)
onnx = args.output / "diagnostic.onnx"
export_report = export_checkpoint(args.output / "resume/model_3.pt", args.output / "deployment.yaml", onnx)
bundle = create_bundle(onnx, env.spec, ASSETS / "unitree/unitree_g1", args.output / "bundle",
                       metadata={"status": "diagnostic", "trained_policy": False, "public_release_ready": False,
                                 "purpose": "Three CPU optimizer iterations verify integration, not walking skill"})
spec = verify_bundle(bundle)
policy = OnnxPolicy(contained_path(bundle, spec.policy_file), spec)
backend = MujocoBackend(contained_path(bundle, spec.model_xml), spec)
evaluation = backend.run(policy, 2.0, np.array([0.6, 0, 0]), headless=True)
report = {"purpose": "cpu_integration_diagnostic", "robot": "g1_29dof", "motion_files": [str(p) for p in args.motions], "export": export_report,
          "mujoco": evaluation, "walking_policy_validated": False, "release_ready": False}
(args.output / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
