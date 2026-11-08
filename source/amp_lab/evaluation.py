"""Common evidence identifiers for simulator evaluations and later release assembly."""

import hashlib
import json
from dataclasses import asdict

import numpy as np


def command_yaw_cases():
    commands = [[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0], [0.9, 0, 0],
                [0.6, 0.1, 0], [0.6, -0.1, 0], [0.6, 0, 0.3], [0.6, 0, -0.3]]
    return [(command, yaw) for command in commands for yaw in [0, np.pi / 2, -np.pi / 2]]


def tracking_metrics(errors, yaw_errors, command):
    """Report measured body-frame velocity as well as error, exposing stationary policies."""
    return {
        "velocity_rmse": float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else None,
        "yaw_rmse": float(np.sqrt(np.mean(np.square(yaw_errors)))) if len(yaw_errors) else None,
        "mean_velocity": [*(np.mean(errors, axis=0) + np.asarray(command[:2])).tolist(),
                          float(np.mean(yaw_errors) + command[2])] if len(errors) else None,
        "tracking_samples": len(errors),
    }


def spec_fingerprint(spec):
    return hashlib.sha256(json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evidence(spec, policy, backend, command):
    return {"schema_version": 1, "backend": backend,
            "policy_sha256": getattr(policy, "sha256", None), "deployment_sha256": spec_fingerprint(spec),
            "command": list(map(float, command)), "control_dt": spec.control_dt}
