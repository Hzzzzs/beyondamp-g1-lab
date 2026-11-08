"""Checkpointed beyondAMP AMPPPO loop; no environment state restoration is claimed."""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_CONFIG = {
    "policy": {"actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128],
               "activation": "elu", "init_noise_std": 0.5},
    "algorithm": {"num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 0.0003,
                  "schedule": "adaptive", "desired_kl": 0.01, "gamma": 0.99, "lam": 0.95,
                  "clip_param": 0.2, "entropy_coef": 0.005, "amp_replay_buffer_size": 100000},
    "discriminator_hidden_dims": [512, 256], "amp_reward_coef": 1.0, "task_reward_lerp": 0.5,
    "steps_per_env": 24, "save_interval": 100, "seed": 42,
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device="cpu"):
    # Our files contain tensors and primitive metadata only; reject arbitrary pickle globals.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return checkpoint


class FixedExpertTransform:
    """Use training-only expert statistics consistently for all AMP inputs."""

    def __init__(self, states, min_std):
        if not np.isfinite(min_std) or min_std <= 0:
            raise ValueError("AMP normalization minimum std must be positive and finite")
        sample = states.detach().to(device="cpu", dtype=torch.float64)
        if sample.ndim != 2 or not torch.isfinite(sample).all():
            raise ValueError("AMP expert states must be a finite two-dimensional tensor")
        self.metadata = {"method": "expert_fixed", "min_std": float(min_std),
                         "mean": sample.mean(0).tolist(),
                         "std": sample.std(0, unbiased=False).clamp(min=min_std).tolist()}
        self.mean = torch.tensor(self.metadata["mean"], dtype=states.dtype, device=states.device)
        self.std = torch.tensor(self.metadata["std"], dtype=states.dtype, device=states.device)

    def __call__(self, states):
        return (states - self.mean) / self.std


class TransformedExpertDataset:
    def __init__(self, dataset, transform):
        self.dataset, self.transform = dataset, transform
        self.states = dataset.states
        self.manifest = dataset.manifest

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for state, next_state in self.dataset.feed_forward_generator(num_mini_batch, mini_batch_size):
            yield self.transform(state), self.transform(next_state)


class Runner:
    def __init__(self, env, dataset, output, config=None):
        from rsl_rl_amp.algorithms import AMPPPO
        from rsl_rl_amp.modules import ActorCritic
        from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator

        self.env, self.dataset = env, dataset
        self.config = copy.deepcopy(config or DEFAULT_CONFIG)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.iteration, self.total_steps = 0, 0
        dims = {"num_actor_obs": env.num_obs, "num_critic_obs": env.num_critic_obs,
                "num_actions": env.num_actions}
        self.architecture = dict(dims, **self.config["policy"])
        actor = ActorCritic(**self.architecture).to(env.device)
        discriminator = AMPDiscriminator(
            2 * dataset.states.shape[1], self.config["amp_reward_coef"],
            self.config["discriminator_hidden_dims"], env.device, self.config["task_reward_lerp"],
        )
        if self.config["algorithm"]["amp_replay_buffer_size"] < env.num_envs:
            raise ValueError("Replay capacity must be at least num_envs")
        normalization = self.config.get("amp_feature_normalization")
        self.amp_transform = None
        expert_dataset = dataset
        if normalization is not None:
            if set(normalization) != {"method", "min_std"} or normalization["method"] != "expert_fixed":
                raise ValueError("Unsupported AMP feature normalization")
            self.amp_transform = FixedExpertTransform(dataset.states, normalization["min_std"])
            expert_dataset = TransformedExpertDataset(dataset, self.amp_transform)
        self.algorithm = AMPPPO(actor, discriminator, expert_dataset, amp_normalizer=None,
                                min_std=torch.full((env.num_actions,), 0.05, device=env.device),
                                device=env.device, **self.config["algorithm"])
        self.algorithm.init_storage(env.num_envs, self.config["steps_per_env"],
                                    [env.num_obs], [env.num_critic_obs], [env.num_actions])
        experiment = {
            "config": self.config, "architecture": self.architecture, "data": dataset.manifest,
            "amp_normalization": f"none_fixed_velocity_scale_{getattr(dataset, 'joint_velocity_scale', 0.05):g}" + (
                "+root_lin_vel_b_unscaled" if getattr(dataset, "root_velocity", False) else ""),
            "backend": env.backend,
            "num_envs": env.num_envs,
        }
        if self.amp_transform is not None:
            experiment["amp_normalization"] = "fixed_expert_statistics"
            experiment["amp_transform"] = self.amp_transform.metadata
        manifest_path = self.output / "training.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != experiment:
            raise ValueError("Output directory already belongs to a different experiment")
        manifest_path.write_text(json.dumps(experiment, indent=2))

    def save(self, path):
        replay = self.algorithm.amp_storage
        rng = np.random.get_state()
        checkpoint = {
            "schema_version": 1, "iteration": self.iteration, "total_steps": self.total_steps,
            "architecture": self.architecture, "config": self.config, "data": self.dataset.manifest,
            "backend": self.env.backend, "model_state_dict": self.algorithm.actor_critic.state_dict(),
            "deployment_spec": self.env.deployment_spec,
            "num_envs": self.env.num_envs,
            "discriminator_state_dict": self.algorithm.discriminator.state_dict(),
            "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
            "learning_rate": self.algorithm.learning_rate,
            "replay": {"states": replay.states[:replay.num_samples].clone(),
                       "next_states": replay.next_states[:replay.num_samples].clone(),
                       "step": replay.step, "num_samples": replay.num_samples},
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else [], "python": random.getstate(),
                    "numpy": [rng[0], rng[1].tolist(), int(rng[2]), int(rng[3]), float(rng[4])]},
            "resume_semantics": "optimizer_and_rng_restored_environment_restarts",
        }
        if self.amp_transform is not None:
            checkpoint["amp_transform"] = self.amp_transform.metadata
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def resume(self, path):
        state = load_checkpoint(path, self.env.device)
        for key, expected in (("config", self.config), ("architecture", self.architecture),
                              ("data", self.dataset.manifest), ("backend", self.env.backend),
                              ("num_envs", self.env.num_envs),
                              ("deployment_spec", self.env.deployment_spec)):
            if state[key] != expected:
                raise ValueError(f"Resume {key} mismatch; refusing silently changed experiment")
        expected_transform = self.amp_transform.metadata if self.amp_transform is not None else None
        if state.get("amp_transform") != expected_transform:
            raise ValueError("Resume AMP feature normalization statistics mismatch")
        metrics_path = self.output / "metrics.jsonl"
        if metrics_path.exists():
            records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            if records and records[-1]["iteration"] != state["iteration"]:
                raise ValueError("Checkpoint iteration differs from last logged iteration; resume into a new directory")
        self.algorithm.actor_critic.load_state_dict(state["model_state_dict"])
        self.algorithm.discriminator.load_state_dict(state["discriminator_state_dict"])
        self.algorithm.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.algorithm.learning_rate = state["learning_rate"]
        replay, saved = self.algorithm.amp_storage, state["replay"]
        replay.states[:saved["num_samples"]].copy_(saved["states"])
        replay.next_states[:saved["num_samples"]].copy_(saved["next_states"])
        replay.step, replay.num_samples = saved["step"], saved["num_samples"]
        self.iteration, self.total_steps = state["iteration"], state["total_steps"]
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if torch.cuda.is_available() and state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])
        random.setstate(state["rng"]["python"])
        kind, values, pos, gaussian, cache = state["rng"]["numpy"]
        np.random.set_state((kind, np.asarray(values, dtype=np.uint32), pos, gaussian, cache))

    def learn(self, iterations):
        obs, critic, amp = self.env.observations()
        if self.amp_transform is not None:
            amp = self.amp_transform(amp)
        alg = self.algorithm
        while self.iteration < iterations:
            started = time.monotonic()
            rewards, amp_rewards = [], []
            rollout_metrics, episode_seconds = {}, []
            with torch.inference_mode():
                for _ in range(self.config["steps_per_env"]):
                    action = alg.act(obs, critic, amp)
                    obs, critic, task_reward, dones, info, terminal_amp = self.env.step(action)
                    next_amp = self.env.observations()[2]
                    if self.amp_transform is not None:
                        terminal_amp = self.amp_transform(terminal_amp)
                        next_amp = self.amp_transform(next_amp)
                    blended, _, style = alg.discriminator.predict_amp_reward(amp, terminal_amp, task_reward)
                    # Isaac rewards already integrate dt; discriminator rewards do not.
                    # task_reward is converted to a per-control-step rate by the adapter.
                    if "timeout_mask" in info:
                        # Time limits bootstrap from the captured PRE-reset terminal critic state.
                        value = alg.actor_critic.evaluate(info["terminal_critic"]).reshape(-1)
                        blended = blended.reshape(-1) + alg.gamma * value * info["timeout_mask"]
                    alg.process_env_step(blended.reshape(-1), dones, {}, terminal_amp)
                    amp = next_amp
                    rewards.append(task_reward.mean().item())
                    amp_rewards.append(style.mean().item())
                    episode_seconds.extend(info.get("completed_episode_seconds", []))
                    for name, value in info.get("rollout_metrics", {}).items():
                        rollout_metrics.setdefault(name, []).append(value)
                alg.compute_returns(critic)
            losses = alg.update()
            if not np.isfinite(losses).all():
                raise FloatingPointError(f"Nonfinite losses: {losses}")
            self.iteration += 1
            self.total_steps += self.env.num_envs * self.config["steps_per_env"]
            record = {"iteration": self.iteration, "total_steps": self.total_steps,
                      "task_reward_rate": float(np.mean(rewards)), "amp_reward": float(np.mean(amp_rewards)),
                      "value_loss": losses[0], "surrogate_loss": losses[1], "amp_loss": losses[2],
                      "gradient_penalty": losses[3], "policy_logit": losses[4], "expert_logit": losses[5],
                      "learning_rate": alg.learning_rate, "seconds": time.monotonic() - started}
            for name, values in rollout_metrics.items():
                record[f"rollout_{name}"] = float(np.mean(values))
            record["completed_episode_count"] = len(episode_seconds)
            record["mean_completed_episode_seconds"] = (float(np.mean(episode_seconds))
                                                         if episode_seconds else None)
            with (self.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if self.iteration % self.config["save_interval"] == 0:
                self.save(self.output / f"model_{self.iteration}.pt")
        self.save(self.output / f"model_{self.iteration}.pt")
