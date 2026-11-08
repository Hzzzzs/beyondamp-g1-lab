import copy
import random
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl_amp")

from amp_lab.export import export_checkpoint  # noqa: E402
from amp_lab.training.runner import DEFAULT_CONFIG, Runner, seed_all  # noqa: E402


class FixtureEnv:
    backend = "test_fixture_not_a_robot_policy"
    device = "cpu"
    num_envs = 4
    num_actions = 2

    def __init__(self, spec):
        self.num_obs = spec.observation_dim
        self.num_critic_obs = self.num_obs
        self.deployment_spec = asdict(spec)
        self.obs = torch.zeros(self.num_envs, self.num_obs)
        self.amp = torch.zeros(self.num_envs, 4)
        self.steps = 0

    def observations(self):
        return self.obs, self.obs, self.amp

    def step(self, action):
        self.steps += 1
        self.obs = torch.randn_like(self.obs) * 0.1
        self.amp = torch.randn_like(self.amp) * 0.1
        return self.obs, self.obs, -action.square().sum(-1), torch.full((4,), self.steps % 4 == 0), {}, self.amp


class FixtureDataset:
    manifest = [{"name": "fixture", "sha256": "fixture_only"}]

    def __init__(self):
        self.states = torch.randn(32, 4)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(32, (mini_batch_size,))
            yield self.states[ids], self.states[(ids + 1) % 32]


def test_actual_beyondamp_updates_resume_and_onnx_export(tmp_path, tiny_spec):
    seed_all(123)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["policy"].update(actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8])
    config["algorithm"].update(num_learning_epochs=1, num_mini_batches=2, amp_replay_buffer_size=32)
    config.update(discriminator_hidden_dims=[16, 8], steps_per_env=4, save_interval=1)
    runner = Runner(FixtureEnv(tiny_spec), FixtureDataset(), tmp_path / "first", config)
    actor_before = [p.detach().clone() for p in runner.algorithm.actor_critic.actor.parameters()]
    discr_before = [p.detach().clone() for p in runner.algorithm.discriminator.parameters()]
    runner.learn(2)
    assert runner.iteration == 2
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, runner.algorithm.actor_critic.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(discr_before, runner.algorithm.discriminator.parameters()))
    checkpoint = tmp_path / "first/model_2.pt"
    expected_rng = (torch.rand(3), np.random.rand(3), random.random())
    resumed = Runner(FixtureEnv(tiny_spec), FixtureDataset(), tmp_path / "resumed", config)
    resumed.resume(checkpoint)
    assert resumed.iteration == 2
    assert resumed.algorithm.amp_storage.num_samples == 32
    torch.testing.assert_close(resumed.algorithm.amp_storage.states, runner.algorithm.amp_storage.states)
    torch.testing.assert_close(torch.rand(3), expected_rng[0])
    np.testing.assert_equal(np.random.rand(3), expected_rng[1])
    assert random.random() == expected_rng[2]
    resumed.learn(3)
    assert resumed.iteration == 3
    tiny_spec.save(tmp_path / "deployment.yaml")
    report = export_checkpoint(checkpoint, tmp_path / "deployment.yaml", tmp_path / "policy.onnx")
    assert report["maximum_absolute_error"] < 1e-5
    assert report["backend"] == "test_fixture_not_a_robot_policy"
    # Same tensor dimensions do not imply the same observation/action contract.
    tiny_spec.joint_names.reverse()
    tiny_spec.save(tmp_path / "wrong.yaml")
    with pytest.raises(ValueError, match="contract differs"):
        export_checkpoint(checkpoint, tmp_path / "wrong.yaml", tmp_path / "wrong.onnx")


def test_fixed_expert_normalization_trains_and_resumes_with_same_statistics(tmp_path, tiny_spec):
    seed_all(123)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["policy"].update(actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8])
    config["algorithm"].update(num_learning_epochs=1, num_mini_batches=2, amp_replay_buffer_size=32)
    config.update(discriminator_hidden_dims=[16, 8], steps_per_env=4, save_interval=1,
                  amp_feature_normalization={"method": "expert_fixed", "min_std": 0.05})
    dataset = FixtureDataset()
    dataset.states[:, 0] = 3.0  # Constant expert dimension exercises the standard deviation floor.
    runner = Runner(FixtureEnv(tiny_spec), dataset, tmp_path / "first", config)
    assert runner.amp_transform.metadata["std"][0] == 0.05
    transformed = runner.amp_transform(dataset.states)
    torch.testing.assert_close(transformed[:, 0], torch.zeros(32))
    runner.learn(2)
    # The first rollout state is physical zero, and replay must contain normalized values.
    expected = runner.amp_transform(torch.zeros(4))
    torch.testing.assert_close(runner.algorithm.amp_storage.states[0], expected)
    checkpoint = tmp_path / "first/model_2.pt"
    resumed = Runner(FixtureEnv(tiny_spec), dataset, tmp_path / "resumed", config)
    resumed.resume(checkpoint)
    assert resumed.amp_transform.metadata == runner.amp_transform.metadata
    resumed.learn(3)
    changed = FixtureDataset()
    mismatched = Runner(FixtureEnv(tiny_spec), changed, tmp_path / "mismatch", config)
    with pytest.raises(ValueError, match="normalization statistics mismatch"):
        mismatched.resume(checkpoint)
