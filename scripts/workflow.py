"""Beginner entry points for the existing training and deployment tools."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/release_seed43_review_local"
RUN = ROOT / "outputs/g1_amp"
PYTHON = Path(os.environ.get("AMP_PYTHON", ROOT / ".venv/bin/python"))


def execute(command, *, log=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "source") + os.pathsep + env.get("PYTHONPATH", "")
    print("Running:", " ".join(map(str, command)), flush=True)
    if log is None:
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            proc = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            if proc.wait():
                raise subprocess.CalledProcessError(proc.returncode, command)


def isaaclab():
    root = os.environ.get("AMP_ISAACLAB")
    path = Path(root) / "isaaclab.sh" if root else None
    if not path or not path.is_file():
        raise ValueError("Set AMP_ISAACLAB to your IsaacLab 2.1.0 installation directory")
    return path


def latest(run):
    items = []
    for p in run.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", p.name)
        if match:
            items.append((int(match.group(1)), p))
    if not items:
        raise ValueError(f"No model_<iteration>.pt found in {run}; specify --run-dir or --checkpoint")
    return max(items)[1]


def finish_artifact(out, checkpoint, digest, spec_digest):
    report = out / "policy.export.json"
    if not report.is_file():
        raise ValueError(f"Missing export parity report: {report}")
    evidence = json.loads(report.read_text())
    if (evidence.get("checkpoint_sha256") != digest or evidence.get("policy_sha256") != hashlib.sha256(
            (out / "bundle/policy.onnx").read_bytes()).hexdigest()):
        raise ValueError("Artifact policy does not match its checkpoint and export report")
    saved = out / "checkpoint.pt"
    if not saved.exists():
        temporary = out / "checkpoint.pt.building"
        shutil.copy2(checkpoint, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
            temporary.unlink()
            raise ValueError("Copied checkpoint hash does not match")
        temporary.replace(saved)
    if hashlib.sha256(saved.read_bytes()).hexdigest() != digest:
        raise ValueError("Artifact checkpoint hash does not match")
    manifest = {"schema_version": 1, "checkpoint": "checkpoint.pt", "checkpoint_sha256": digest,
                "bundle": "bundle", "training_spec_sha256": spec_digest,
                "export_report": "policy.export.json",
                "note": "Deployment contract is in bundle/deployment.yaml; training statistics are in checkpoint."}
    (out / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n")


def artifact(args):
    if args.bundle and args.checkpoint:
        raise ValueError("Choose either --bundle or --checkpoint")
    if args.bundle:
        return args.bundle.resolve()
    if args.checkpoint is None:
        if not (EXAMPLE / "bundle/deployment.yaml").is_file():
            raise ValueError("Example bundle missing; download and extract the v0.1.0 release into examples/")
        return EXAMPLE / "bundle"
    checkpoint = latest(args.run_dir) if args.checkpoint == "latest" else Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint}")
    spec = args.spec.resolve() if args.spec else checkpoint.parent / "deployment.yaml"
    if not args.spec and not spec.is_file():
        spec = checkpoint.parent.parent / "bundle/deployment.yaml"
    if not spec.is_file():
        raise ValueError(f"Training deployment.yaml not found: {spec}; use --spec")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    spec_digest = hashlib.sha256(spec.read_bytes()).hexdigest()
    out = ROOT / "outputs/artifacts" / f"{digest}-{spec_digest}"
    bundle = out / "bundle"
    if bundle.exists():
        from dataclasses import asdict

        from amp_lab.spec import DeploymentSpec

        if asdict(DeploymentSpec.load(spec)) != asdict(DeploymentSpec.load(bundle / "deployment.yaml")):
            raise ValueError("Cached bundle differs from the requested deployment configuration")
        execute([PYTHON, "-m", "amp_lab.cli", "check-bundle", bundle])
        finish_artifact(out, checkpoint, digest, spec_digest)
        return bundle
    out.mkdir(parents=True, exist_ok=True)
    policy = out / "policy.onnx"
    if not policy.exists():
        execute([PYTHON, "-m", "amp_lab.cli", "export", checkpoint, "--spec", spec, "--output", policy])
    export_report = policy.with_suffix(".export.json")
    if not export_report.is_file():
        raise ValueError(f"Missing export parity report: {export_report}; remove the incomplete artifact and retry")
    evidence = json.loads(export_report.read_text())
    if evidence.get("checkpoint_sha256") != digest or evidence.get("policy_sha256") != hashlib.sha256(
            policy.read_bytes()).hexdigest():
        raise ValueError("Export parity report does not match the selected checkpoint and policy")
    robot = args.robot_assets or EXAMPLE / "bundle/robot"
    if not robot.is_dir():
        raise ValueError("Robot assets missing; supply --robot-assets or extract the example bundle")
    execute([PYTHON, "-m", "amp_lab.cli", "package", "--policy", policy, "--spec", spec,
             "--robot", robot, "--output", bundle])
    finish_artifact(out, checkpoint, digest, spec_digest)
    return bundle


def common(p):
    p.add_argument("--checkpoint", help="Checkpoint path, or latest within --run-dir")
    p.add_argument("--run-dir", type=Path, default=RUN)
    p.add_argument("--bundle", type=Path, help="Use an existing complete bundle")
    p.add_argument("--spec", type=Path, help="Training deployment.yaml for a checkpoint")
    p.add_argument("--robot-assets", type=Path, help="Robot asset directory for export")
    p.add_argument("--seconds", type=float, default=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Validate data and start IsaacLab training")
    train.add_argument("--robot", choices=["g1"], default="g1")
    train.add_argument("--motion", choices=["cmu_walk"], default="cmu_walk")
    train.add_argument("--motions", type=Path, nargs="+", help="Use custom AMP NPZ files")
    train.add_argument("--config", type=Path, default=ROOT / "configs/train_g1_expert_fixed_yaw1.json")
    train.add_argument("--output", type=Path, default=RUN)
    train.add_argument("--num-envs", type=int, default=4096)
    train.add_argument("--iterations", type=int, default=6000)
    train.add_argument("--seed", type=int, default=43)
    train.add_argument("--resume", help="Checkpoint path or latest in --output")
    train.add_argument("--no-export", action="store_true", help="Skip artifact creation after training")
    train.add_argument("--robot-assets", type=Path, help="Robot asset directory for the resulting artifact")
    dep = sub.add_parser("deploy", help="Run a trained bundle")
    common(dep)
    dep.add_argument("--sim", choices=["mujoco", "isaacsim"], default="mujoco")
    dep.add_argument("--velocity", nargs=3, type=float, default=[0.6, 0, 0])
    dep.add_argument("--video", type=Path)
    dep.add_argument("--report", type=Path)
    dep.add_argument("--show", action="store_true", help="Open a simulation window")
    ev = sub.add_parser("evaluate", help="Run the command matrix in one or both simulators")
    common(ev)
    ev.add_argument("--sim", choices=["mujoco", "isaacsim", "both"], default="mujoco")
    ev.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    ev.add_argument("--gait", action="store_true", help="Also compare MuJoCo gait with example reference clips")
    ev.add_argument("--references", type=Path, nargs="+", help="AMP NPZ reference clips for --gait")
    args = parser.parse_args()
    os.environ["PYTHONPATH"] = str(ROOT / "source") + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, str(ROOT / "source"))
    if not PYTHON.is_file():
        raise ValueError(f"Python interpreter not found: {PYTHON}; install dependencies using the quick start")
    if args.command == "train":
        launcher = isaaclab()
        if args.iterations < 1 or args.num_envs < 2:
            raise ValueError("Training requires at least one iteration and two environments")
        if args.output.exists() and any(args.output.iterdir()) and not args.resume:
            raise ValueError("Training output is not empty; choose a new --output or explicitly --resume")
        motions = args.motions or sorted((EXAMPLE / "data").glob("*_candidate_amp.npz"))
        if not motions:
            raise ValueError("Training motions missing; extract the example or use --motions")
        for motion in motions:
            execute([PYTHON, "-m", "amp_lab.cli", "motion", "validate", motion])
        json.loads(args.config.read_text())
        cmd = [launcher, "-p", "-u", ROOT / "scripts/train.py", "--headless", "--motions", *motions,
               "--config", args.config, "--output", args.output, "--num-envs", args.num_envs,
               "--iterations", args.iterations, "--seed", args.seed]
        if args.resume:
            resume = latest(args.output) if args.resume == "latest" else Path(args.resume)
            match = re.fullmatch(r"model_(\d+)\.pt", resume.name)
            if match and int(match.group(1)) >= args.iterations:
                raise ValueError("--iterations must be greater than the resumed checkpoint iteration")
            cmd += ["--resume", resume]
        execute(cmd, log=ROOT / "logs" / f"{args.output.name}.log")
        final_checkpoint = latest(args.output)
        if final_checkpoint.name != f"model_{args.iterations}.pt":
            raise ValueError(f"Training exited but target checkpoint model_{args.iterations}.pt is missing; "
                             f"latest is {final_checkpoint.name}")
        if not args.no_export:
            bundle = artifact(argparse.Namespace(bundle=None, checkpoint="latest", run_dir=args.output,
                                                spec=None, robot_assets=args.robot_assets))
            print(f"Training artifact: {bundle.parent}")
    elif args.command == "deploy":
        bundle = artifact(args)
        if args.sim == "mujoco":
            cmd = [PYTHON, "-m", "amp_lab.cli", "deploy", bundle, "--seconds", args.seconds, "--velocity",
                   *args.velocity]
        else:
            cmd = [isaaclab(), "-p", ROOT / "scripts/play_isaac.py", bundle, "--seconds", args.seconds,
                   "--velocity", *args.velocity]
        if not args.show:
            cmd.append("--headless")
        if args.video:
            cmd += ["--video", args.video]
        if args.report:
            cmd += ["--report", args.report]
        execute(cmd)
    else:
        if args.sim in ("isaacsim", "both"):
            isaaclab()
        bundle = artifact(args)
        args.output.mkdir(parents=True, exist_ok=True)
        reports = []
        if args.sim in ("mujoco", "both"):
            execute([PYTHON, ROOT / "scripts/evaluate_mujoco.py", bundle, "--seconds", args.seconds,
                     "--output", args.output / "mujoco.json"])
            reports.append(args.output / "mujoco.json")
        if args.sim in ("isaacsim", "both"):
            execute([isaaclab(), "-p", ROOT / "scripts/evaluate_isaac.py", bundle, "--headless",
                     "--seconds", args.seconds, "--output", args.output / "isaac.json"])
            reports.append(args.output / "isaac.json")
        if args.gait:
            refs = args.references or sorted((EXAMPLE / "data").glob("*_candidate_amp.npz"))
            if not refs:
                raise ValueError("Gait reference clips missing; extract example data or use --references")
            execute([PYTHON, ROOT / "scripts/compare_gait.py", bundle, "--references", *refs,
                     "--seconds", args.seconds, "--output", args.output / "gait.json"])
        summary = {"schema_version": 1, "reports": {}, "gait": "gait.json" if args.gait else None,
                   "note": "Gait support and sliding are kinematic proxies, not measured contact forces."}
        matrices = []
        for path in reports:
            data = json.loads(path.read_text())
            matrices.append(data)
            cases = data["cases"]
            velocity = [c["velocity_rmse"] for c in cases if c["velocity_rmse"] is not None]
            yaw = [c["yaw_rmse"] for c in cases if c["yaw_rmse"] is not None]
            summary["reports"][path.stem] = {
                "path": path.name, "policy_sha256": data["policy_sha256"], "cases": len(cases),
                "completed_fraction": data["completed_fraction"],
                "worst_velocity_rmse": max(velocity) if velocity else None,
                "worst_yaw_rmse": max(yaw) if yaw else None,
                "cases_without_tracking_samples": len(cases) - len(velocity),
            }
        if len(matrices) == 2:
            a, b = matrices
            if a["policy_sha256"] != b["policy_sha256"] or len(a["cases"]) != len(b["cases"]):
                raise ValueError("Simulator reports do not describe the same policy and test matrix")
            deltas = []
            for left, right in zip(a["cases"], b["cases"]):
                if left["command"] != right["command"] or left["initial_yaw"] != right["initial_yaw"]:
                    raise ValueError("Simulator report commands or initial yaw do not match")
                if left["mean_velocity"] is not None and right["mean_velocity"] is not None:
                    deltas.append([abs(x - y) for x, y in zip(left["mean_velocity"], right["mean_velocity"])])
            summary["cross_sim"] = {
                "matched_cases": len(a["cases"]), "cases_with_velocity": len(deltas),
                "max_mean_velocity_abs_difference": [max(d[i] for d in deltas) for i in range(3)]
                if deltas else None,
                "units": ["m/s", "m/s", "rad/s"],
            }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
