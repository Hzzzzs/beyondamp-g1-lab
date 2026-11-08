"""A small sequential MuJoCo adapter for integration diagnostics, not production AMP training."""

from dataclasses import asdict

import numpy as np
import torch

from amp_lab.g1 import default_spec
from amp_lab.mujoco_backend import MujocoBackend
from amp_lab.paths import ASSETS, G1_XML
from amp_lab.provenance import asset_manifest
from amp_lab.runtime import rotate_inverse


class CpuSmokeEnv:
    backend = "mujoco_cpu_diagnostic_not_release"
    device = "cpu"
    num_obs, num_critic_obs, num_actions = 96, 99, 29

    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.spec = default_spec()
        self.spec.provenance = {"status": "cpu_diagnostic_only", "backend": self.backend,
                                "model_tree_sha256": asset_manifest(ASSETS / "unitree/unitree_g1")["tree_sha256"]}
        self.deployment_spec = asdict(self.spec)
        self.backends = [MujocoBackend(G1_XML, self.spec) for _ in range(num_envs)]
        self.lengths = np.zeros(num_envs, dtype=int)
        self.command = np.array([0.6, 0, 0])
        self._refresh()

    def _one_observation(self, backend):
        state = backend.state()
        policy = backend.observation(self.command)
        critic = np.r_[state["lin_vel_b"], policy]
        amp = np.r_[state["joint_pos"] - self.spec.default_joint_pos, state["joint_vel"] * 0.05]
        return policy, critic, amp

    def _refresh(self):
        groups = list(zip(*(self._one_observation(b) for b in self.backends)))
        self._observations = tuple(torch.tensor(np.stack(group), dtype=torch.float32) for group in groups)

    def observations(self):
        return self._observations

    def step(self, actions):
        rewards, dones, timeouts, terminal_amp, terminal_critic = [], [], [], [], []
        actions = actions.detach().cpu().numpy()
        for index, (backend, action) in enumerate(zip(self.backends, actions)):
            previous_action = backend.builder.last_action.copy()
            backend.step(action)
            self.lengths[index] += 1
            state = backend.state()
            gravity = rotate_inverse(state["quat_wxyz"], [0, 0, -1])
            velocity_error = np.sum((state["lin_vel_b"][:2] - self.command[:2]) ** 2)
            reward = (2 * np.exp(-velocity_error / 0.25) + np.exp(-state["ang_vel_b"][2] ** 2 / 0.25) + 0.2
                      - 2 * state["lin_vel_b"][2] ** 2 - 0.05 * np.sum(state["ang_vel_b"][:2] ** 2)
                      - np.sum(gravity[:2] ** 2) - 0.02 * np.sum((backend.builder.last_action - previous_action) ** 2))
            _, critic, amp = self._one_observation(backend)
            fallen = backend.data.xpos[backend.root_id, 2] < 0.45 or gravity[2] > -0.5
            timeout = self.lengths[index] >= 100
            rewards.append(reward)
            dones.append(fallen or timeout)
            timeouts.append(timeout and not fallen)
            terminal_amp.append(amp)
            terminal_critic.append(critic)
            if fallen or timeout:
                backend.reset()
                self.lengths[index] = 0
        self._refresh()
        info = {"timeout_mask": torch.tensor(timeouts),
                "terminal_critic": torch.tensor(np.stack(terminal_critic), dtype=torch.float32)}
        actor, critic, _ = self.observations()
        return (actor, critic, torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones), info,
                torch.tensor(np.stack(terminal_amp), dtype=torch.float32))
