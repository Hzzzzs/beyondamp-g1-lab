"""Named, control-rate expert transitions for beyondAMP's discriminator API."""

import numpy as np
import torch

from amp_lab.motion import load_motion, sha256
from amp_lab.runtime import rotate_inverse


class JointMotionDataset:
    def __init__(self, paths, spec, device, *, root_velocity=False, joint_velocity_scale=0.05):
        self.root_velocity = root_velocity
        self.joint_velocity_scale = joint_velocity_scale
        states, next_states = [], []
        reference_joints, reference_velocities, reference_roots = [], [], []
        self.manifest = []
        for path in paths:
            motion = load_motion(path)
            if not np.isclose(float(motion["fps"]), 1 / spec.control_dt, rtol=0, atol=1e-5):
                raise ValueError(f"{path}: resample to {1 / spec.control_dt} Hz before training")
            names = motion["joint_names"].tolist()
            if set(names) != set(spec.joint_names):
                raise ValueError(f"{path}: joint names differ from live robot")
            indices = [names.index(name) for name in spec.joint_names]
            root = motion["body_names"].tolist().index(spec.root_body)
            # Same convention as task.amp_observation; this scale belongs to AMP, not the actor.
            observations = np.concatenate([
                motion["joint_pos"][:, indices] - np.asarray(spec.default_joint_pos),
                joint_velocity_scale * motion["joint_vel"][:, indices],
            ], axis=-1).astype(np.float32)
            if root_velocity:
                velocity_b = np.stack([rotate_inverse(q, v) for q, v in zip(
                    motion["body_quat_w"][:, root], motion["body_lin_vel_w"][:, root])])
                observations = np.concatenate([observations, velocity_b], axis=-1).astype(np.float32)
            states.append(observations[:-1])
            next_states.append(observations[1:])
            reference_joints.append(motion["joint_pos"][:-1, indices])
            reference_velocities.append(motion["joint_vel"][:-1, indices])
            reference_roots.append(np.concatenate([
                motion["body_pos_w"][:-1, root], motion["body_quat_w"][:-1, root],
                motion["body_lin_vel_w"][:-1, root], motion["body_ang_vel_w"][:-1, root],
            ], axis=-1))
            self.manifest.append({"name": path.name, "sha256": sha256(path), "frames": len(observations)})
        if not states:
            raise ValueError("At least one expert motion is required")
        self.states = torch.tensor(np.concatenate(states), device=device)
        self.next_states = torch.tensor(np.concatenate(next_states), device=device)
        self.reference_joint_pos = torch.tensor(np.concatenate(reference_joints), dtype=torch.float32, device=device)
        self.reference_joint_vel = torch.tensor(np.concatenate(reference_velocities), dtype=torch.float32, device=device)
        self.reference_root_states = torch.tensor(np.concatenate(reference_roots), dtype=torch.float32, device=device)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            indices = torch.randint(len(self.states), (mini_batch_size,), device=self.states.device)
            yield self.states[indices], self.next_states[indices]
