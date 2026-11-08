"""CPU-side command line; training itself uses scripts/train.py in IsaacLab's Python."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .paths import ASSETS, G1_XML, contained_path


def doctor(training=False):
    modules = ["numpy", "scipy", "yaml", "mujoco", "onnxruntime"]
    if training:
        modules += ["torch", "isaaclab", "rsl_rl_amp", "robotlib"]
    result = {"python": sys.version, "python_executable": sys.executable,
              "modules": {name: importlib.util.find_spec(name) is not None for name in modules},
              "g1_xml_exists": G1_XML.is_file(),
              "training_target": {"isaac_sim": "4.5.0", "isaaclab": "v2.1.0", "python": "3.10"}}
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        result["gpu"] = probe.stdout.strip() if probe.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        result["gpu"] = "unavailable"
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if all(result["modules"].values()) and (not training or result["gpu"] != "unavailable") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="amp-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor", help="Check CPU or GPU training dependencies")
    doctor_parser.add_argument("--training", action="store_true")
    fetch = commands.add_parser("fetch-upstream", help="Fetch repositories at locked commits")
    fetch.add_argument("--install", action="store_true")
    profile = commands.add_parser("profile", help="Write provisional, explicitly untrained G1 configuration")
    profile.add_argument("--output", type=Path, required=True)
    model_audit = commands.add_parser("model-audit", help="Compare source URDF and MJCF mass/inertia/kinematics")
    model_audit.add_argument("--urdf", type=Path, default=ASSETS / "unitree/unitree_g1/urdf/g1_29dof.urdf")
    model_audit.add_argument("--xml", type=Path, default=G1_XML)
    model_audit.add_argument("--output", type=Path)
    motion = commands.add_parser("motion")
    motions = motion.add_subparsers(dest="operation", required=True)
    validate = motions.add_parser("validate")
    validate.add_argument("path", type=Path)
    audit = motions.add_parser("audit")
    audit.add_argument("path", type=Path)
    audit.add_argument("--xml", type=Path, default=G1_XML)
    audit.add_argument("--output", type=Path, help="Save the audit report as JSON")
    retarget = motions.add_parser("retarget-cmu")
    retarget.add_argument("--asf", type=Path, required=True)
    retarget.add_argument("--amc", type=Path, required=True)
    retarget.add_argument("--xml", type=Path, default=G1_XML)
    retarget.add_argument("--output", type=Path, required=True)
    retarget.add_argument("--fps", type=float, default=120)
    smooth = motions.add_parser("smooth", help="Offline joint filtering experiment; re-convert and audit before use")
    smooth.add_argument("source", type=Path)
    smooth.add_argument("--output", type=Path, required=True)
    smooth.add_argument("--sigma-seconds", type=float, default=0.04)
    retime = motions.add_parser("retime", help="Change qpos playback speed; convert again to recompute velocities")
    retime.add_argument("source", type=Path)
    retime.add_argument("--output", type=Path, required=True)
    retime.add_argument("--speed", type=float, required=True, help="0.65 means 65%% playback speed")
    convert = motions.add_parser("convert", help="Named qpos NPZ to AMP data with FK and resampling")
    convert.add_argument("source", type=Path)
    convert.add_argument("--xml", type=Path, default=G1_XML)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--fps", type=float, default=50)
    playback = motions.add_parser("playback", help="Kinematic reference playback; no trained policy involved")
    playback.add_argument("path", type=Path)
    playback.add_argument("--xml", type=Path, default=G1_XML)
    playback.add_argument("--video", type=Path)
    export = commands.add_parser("export", help="Export actor from project checkpoint with ONNX parity check")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("--spec", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    package = commands.add_parser("package")
    package.add_argument("--policy", type=Path, required=True)
    package.add_argument("--spec", type=Path, required=True)
    package.add_argument("--robot", type=Path, default=ASSETS / "unitree/unitree_g1")
    package.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="Inspect a self-contained bundle without Isaac or upstream assets")
    inspect.add_argument("bundle", type=Path)
    check = commands.add_parser("check-bundle")
    check.add_argument("bundle", type=Path)
    deploy = commands.add_parser("deploy", help="Run a bundle in native MuJoCo")
    deploy.add_argument("bundle", type=Path)
    deploy.add_argument("--seconds", type=float, default=20)
    deploy.add_argument("--velocity", nargs=3, type=float, default=[0.6, 0, 0], metavar=("VX", "VY", "WZ"))
    deploy.add_argument("--headless", action="store_true")
    deploy.add_argument("--report", type=Path)
    deploy.add_argument("--video", type=Path)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(args.training)
    if args.command == "fetch-upstream":
        from .bootstrap import fetch_upstream
        fetch_upstream(args.install)
    elif args.command == "profile":
        from .g1 import default_spec
        default_spec().save(args.output)
        print(args.output)
    elif args.command == "model-audit":
        from .model_audit import audit_models
        report = audit_models(args.urdf, args.xml)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    elif args.command == "motion":
        from .motion import load_motion, qpos_to_motion
        if args.operation == "validate":
            data = load_motion(args.path)
            print(json.dumps({"frames": len(data["joint_pos"]), "fps": float(data["fps"]),
                              "joints": data["joint_names"].tolist(), "bodies": len(data["body_names"])}, indent=2))
        elif args.operation == "smooth":
            from .smoothing import smooth_qpos
            print(json.dumps(smooth_qpos(args.source, args.output, sigma_seconds=args.sigma_seconds), indent=2))
        elif args.operation == "retime":
            from .timing import retime_qpos
            print(json.dumps(retime_qpos(args.source, args.output, speed=args.speed), indent=2))
        elif args.operation == "convert":
            print(json.dumps(qpos_to_motion(args.source, args.xml, args.output, target_fps=args.fps), indent=2))
        elif args.operation == "audit":
            from .quality import audit_motion
            report = audit_motion(args.path, args.xml)
            if args.output:
                if args.output.exists():
                    raise FileExistsError("Use a new report path to preserve existing data")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
        elif args.operation == "retarget-cmu":
            from .cmu import retarget_cmu
            print(json.dumps(retarget_cmu(args.asf, args.amc, args.xml, args.output, fps=args.fps), indent=2))
        else:
            from .playback import play_motion
            play_motion(args.path, args.xml, video=args.video)
    elif args.command == "export":
        from .export import export_checkpoint
        print(json.dumps(export_checkpoint(args.checkpoint, args.spec, args.output), indent=2))
    elif args.command == "package":
        from .bundle import create_bundle
        from .runtime import OnnxPolicy
        from .spec import DeploymentSpec
        spec = DeploymentSpec.load(args.spec)
        OnnxPolicy(args.policy, spec)
        create_bundle(args.policy, spec, args.robot, args.output)
        print(args.output)
    elif args.command == "inspect":
        from .inspection import inspect_bundle
        print(json.dumps(inspect_bundle(args.bundle), indent=2, ensure_ascii=False))
    elif args.command in ("check-bundle", "deploy"):
        from .bundle import verify_bundle
        from .runtime import OnnxPolicy
        spec = verify_bundle(args.bundle)
        policy = OnnxPolicy(contained_path(args.bundle, spec.policy_file), spec)
        if args.command == "deploy":
            from .mujoco_backend import MujocoBackend
            backend = MujocoBackend(contained_path(args.bundle, spec.model_xml), spec)
            report = backend.run(policy, duration=args.seconds, command=np.asarray(args.velocity),
                                 headless=args.headless, video=args.video)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
        else:
            print("Bundle checksums, configuration and ONNX shapes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
