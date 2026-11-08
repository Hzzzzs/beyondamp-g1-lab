"""Export the deterministic actor and check ONNX parity before accepting the artifact."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


def export_checkpoint(checkpoint_path, spec_path, output):
    import torch
    from rsl_rl_amp.modules import ActorCritic

    from .motion import sha256
    from .runtime import OnnxPolicy
    from .spec import DeploymentSpec
    from .training.runner import load_checkpoint

    state = load_checkpoint(checkpoint_path)
    spec = DeploymentSpec.load(Path(spec_path))
    if state["deployment_spec"] != asdict(spec):
        raise ValueError("Deployment contract differs from the checkpoint's training contract")
    if state["architecture"]["num_actor_obs"] != spec.observation_dim:
        raise ValueError("Deployment observation dimension differs from training")
    if state["architecture"]["num_actions"] != len(spec.joint_names):
        raise ValueError("Deployment action dimension differs from training")
    actor_critic = ActorCritic(**state["architecture"])
    actor_critic.load_state_dict(state["model_state_dict"])
    actor = copy.deepcopy(actor_critic.actor).cpu().eval()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.stem + ".building.onnx")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        # Legacy exporter is compatible with the target Isaac torch version.
        torch.onnx.export(actor, torch.zeros(1, spec.observation_dim), str(temporary),
                          input_names=["observation"], output_names=["action"], opset_version=17, dynamo=False)
        policy = OnnxPolicy(temporary, spec)
        generator = np.random.default_rng(42)
        maximum = 0.0
        for _ in range(32):
            observation = generator.normal(size=spec.observation_dim).astype(np.float32)
            with torch.no_grad():
                expected = actor(torch.from_numpy(observation[None])).numpy()[0]
            actual = policy(observation)
            maximum = max(maximum, float(np.max(np.abs(expected - actual))))
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {"checkpoint": str(checkpoint_path), "iteration": state["iteration"],
              "checkpoint_sha256": sha256(checkpoint_path), "policy_sha256": sha256(output),
              "backend": state["backend"], "maximum_absolute_error": maximum,
              "parity_samples": 32, "robot_success_validated": False}
    output.with_suffix(".export.json").write_text(json.dumps(report, indent=2))
    return report
