"""Versioned deployment contract shared by training export and both simulators."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class ObservationTerm:
    name: str
    size: int
    scale: float = 1.0
    clip: float | None = None


@dataclass
class DeploymentSpec:
    joint_names: list[str]
    default_joint_pos: list[float]
    stiffness: list[float]
    damping: list[float]
    effort_limit: list[float]
    action_scale: list[float]
    observations: list[ObservationTerm]
    schema_version: int = 1
    robot: str = "unitree_g1_29dof"
    policy_file: str = "policy.onnx"
    model_xml: str = "robot/scene.xml"
    root_body: str = "pelvis"
    simulation_dt: float = 0.005
    control_decimation: int = 4
    history_length: int = 1
    history_order: str = "oldest_first_frame_major"
    history_reset: str = "zeros_then_current"
    quaternion_order: str = "wxyz"
    angular_velocity_frame: str = "body"
    linear_velocity_point: str = "root_center_of_mass"
    self_collisions: bool = False
    ground_friction: float = 1.0
    normalization: str = "embedded_in_policy"
    action_clip: float = 12.0
    default_root_height: float = 0.76
    armature: list[float] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    @property
    def frame_dim(self):
        return sum(term.size for term in self.observations)

    @property
    def observation_dim(self):
        return self.frame_dim * self.history_length

    @property
    def control_dt(self):
        return self.simulation_dt * self.control_decimation

    def validate(self):
        n = len(self.joint_names)
        if not n or len(set(self.joint_names)) != n:
            raise ValueError("joint_names must be nonempty and unique")
        for name in ("default_joint_pos", "stiffness", "damping", "effort_limit", "action_scale"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain {n} finite values")
        for name in ("stiffness", "damping", "effort_limit"):
            if np.any(np.asarray(getattr(self, name)) < 0):
                raise ValueError(f"{name} cannot be negative")
        if self.armature and (len(self.armature) != n or not np.isfinite(self.armature).all()
                             or np.any(np.asarray(self.armature) < 0)):
            raise ValueError("Invalid armature")
        if self.schema_version != 1 or self.quaternion_order != "wxyz":
            raise ValueError("Unsupported schema or quaternion convention")
        if self.angular_velocity_frame != "body" or self.normalization != "embedded_in_policy":
            raise ValueError("Unsupported velocity/normalization convention")
        if self.linear_velocity_point != "root_center_of_mass":
            raise ValueError("Unsupported linear velocity reference point")
        if not isinstance(self.self_collisions, bool) or not np.isfinite(self.ground_friction) or self.ground_friction < 0:
            raise ValueError("Invalid collision/friction settings")
        if self.history_order != "oldest_first_frame_major" or self.history_reset != "zeros_then_current":
            raise ValueError("Unsupported history convention")
        if not isinstance(self.history_length, int) or self.history_length < 1:
            raise ValueError("history_length must be a positive integer")
        if not isinstance(self.control_decimation, int) or self.control_decimation < 1:
            raise ValueError("control_decimation must be a positive integer")
        if not np.isfinite(self.simulation_dt) or self.simulation_dt <= 0:
            raise ValueError("simulation_dt must be positive")
        if not np.isfinite(self.action_clip) or self.action_clip <= 0:
            raise ValueError("action_clip must be positive")
        if not np.isfinite(self.default_root_height) or self.default_root_height <= 0:
            raise ValueError("default_root_height must be positive")
        sizes = {"base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3,
                 "joint_pos_rel": n, "joint_vel": n, "last_action": n}
        names = [term.name for term in self.observations]
        if not names or len(set(names)) != len(names):
            raise ValueError("Observation terms must be nonempty and unique")
        for term in self.observations:
            if term.name not in sizes or term.size != sizes[term.name]:
                raise ValueError(f"Unsupported observation: {term}")
            if not np.isfinite(term.scale) or (term.clip is not None and
                                              (not np.isfinite(term.clip) or term.clip <= 0)):
                raise ValueError(f"Invalid observation scale/clip: {term}")
        return self

    def save(self, path: Path):
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["observations"] = [ObservationTerm(**term) for term in data["observations"]]
        return cls(**data).validate()
