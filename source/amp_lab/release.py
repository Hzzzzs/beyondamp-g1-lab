"""Assemble a matching checkpoint/data/model release after both simulators pass evaluation."""

import json
import math
import shutil
from pathlib import Path

from .bundle import create_bundle, verify_bundle, write_checksums
from .evaluation import spec_fingerprint
from .motion import sha256


def validate_evidence(reports, spec, policy_sha256, *, minimum_seconds=20,
                      maximum_velocity_rmse=0.3, maximum_yaw_rmse=0.3):
    backends = set()
    for report in reports:
        if report.get("policy_sha256") != policy_sha256 or report.get("deployment_sha256") != spec_fingerprint(spec):
            raise ValueError("Evaluation is for a different policy/deployment contract")
        if report.get("backend") not in ("mujoco", "isaaclab"):
            raise ValueError("Expected actual MuJoCo or IsaacLab evaluation evidence")
        duration = report.get("requested_seconds", 0)
        elapsed = report.get("completed_seconds", 0)
        error = report.get("velocity_rmse")
        if (not math.isfinite(duration) or not math.isfinite(elapsed) or duration < minimum_seconds
                or elapsed + spec.control_dt < duration or report.get("fallen") is not False
                or report.get("completed") is not True or error is None
                or not math.isfinite(error) or error > maximum_velocity_rmse):
            raise ValueError("Evaluation has not passed the minimum walking release criteria")
        command = report.get("command", [])
        if len(command) != 3 or command[0] < 0.3:
            raise ValueError("Release evidence must include forward walking at >=0.3 m/s")
        measured = report.get("mean_velocity")
        if (not isinstance(measured, list) or len(measured) != 3
                or not all(math.isfinite(value) for value in measured)
                or measured[0] < max(0.15, 0.5 * command[0])):
            raise ValueError("Walking criteria require measured forward motion, not stationary survival")
        yaw_error = report.get("yaw_rmse")
        if yaw_error is None or not math.isfinite(yaw_error) or yaw_error > maximum_yaw_rmse:
            raise ValueError("Walking criteria require yaw tracking, not uncontrolled turning")
        backends.add(report["backend"])
    if backends != {"mujoco", "isaaclab"}:
        raise ValueError("Successful evaluations from both simulators are required")


def build_release(checkpoint, spec_path, policy, motions, reports, robot_directory, output):
    from dataclasses import asdict

    from .spec import DeploymentSpec
    from .training.runner import load_checkpoint

    state = load_checkpoint(checkpoint)
    spec = DeploymentSpec.load(spec_path)
    if state["backend"] != "isaaclab_2.1.0_isaacsim_4.5.0" or state["iteration"] < 1:
        raise ValueError("Diagnostic/test checkpoints cannot be published as trained walking examples")
    if state["deployment_spec"] != asdict(spec):
        raise ValueError("Checkpoint/spec mismatch")
    export = json.loads(policy.with_suffix(".export.json").read_text())
    if (export.get("checkpoint_sha256") != sha256(checkpoint)
            or export.get("policy_sha256") != sha256(policy)
            or export.get("parity_samples", 0) < 32
            or not math.isfinite(export.get("maximum_absolute_error", math.inf))
            or export["maximum_absolute_error"] > 1e-5):
        raise ValueError("Policy has no passing numerical export evidence for this checkpoint")
    expected = sorted(entry["sha256"] for entry in state["data"])
    if sorted(sha256(path) for path in motions) != expected:
        raise ValueError("Release motion files differ from the actual training dataset")
    evidence = [json.loads(path.read_text()) for path in reports]
    validate_evidence(evidence, spec, sha256(policy))
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + ".building")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        create_bundle(policy, spec, robot_directory, temporary / "bundle", metadata={
            "status": "walking_example", "trained_policy": True, "public_release_ready": False,
            "license_review_required": True, "iteration": state["iteration"],
        })
        (temporary / "training").mkdir()
        shutil.copy2(checkpoint, temporary / "training/model.pt")
        (temporary / "data").mkdir()
        clips = []
        for index, motion in enumerate(motions):
            name = f"{index:02d}_{motion.name}"
            shutil.copy2(motion, temporary / "data" / name)
            clips.append({"split": "train", "motion": name, "sha256": sha256(motion),
                          "source_path": str(motion)})
            if motion.with_suffix(".json").exists():
                shutil.copy2(motion.with_suffix(".json"), temporary / "data" / f"{index:02d}_{motion.stem}.json")
        (temporary / "data/manifest.json").write_text(json.dumps({
            "schema_version": 1, "purpose": "expert_training_data_matching_checkpoint",
            "validation_included": False, "clips": clips,
        }, indent=2))
        (temporary / "evaluations.json").write_text(json.dumps(evidence, indent=2))
        (temporary / "release.json").write_text(json.dumps({
            "schema_version": 1, "checkpoint_sha256": sha256(checkpoint), "policy_sha256": sha256(policy),
            "data": state["data"], "training_config": state["config"],
            "deployment_sha256": spec_fingerprint(spec), "iteration": state["iteration"],
            "status": "assembled_pending_license_and_documentation_review",
        }, indent=2))
        verify_bundle(temporary / "bundle")
        write_checksums(temporary)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return output
