"""Deployment-only inspection: no training dependency or source asset checkout required."""

import json
from pathlib import Path

from .bundle import verify_bundle
from .paths import contained_path
from .runtime import OnnxPolicy


def inspect_bundle(directory):
    from .mujoco_backend import MujocoBackend

    directory = Path(directory)
    spec = verify_bundle(directory)
    policy = OnnxPolicy(contained_path(directory, spec.policy_file), spec)
    backend = MujocoBackend(contained_path(directory, spec.model_xml), spec)
    backend.reset()
    # A dry inference verifies actual outputs; it does not run a physics episode.
    import numpy as np

    action = policy(backend.observation(np.zeros(3)))
    backend.builder.action_target(action)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return {"bundle": str(directory.resolve()), "checksums_verified": True,
            "model_loaded": True, "onnx_inference_verified": True,
            "observation_size": spec.observation_dim, "action_size": len(action),
            "control_hz": 1 / spec.control_dt, "policy_sha256": policy.sha256,
            "metadata": metadata,
            "note": "接口检查通过不代表策略能行走；行走能力请查看同一权重的双模拟器评测报告。"}
