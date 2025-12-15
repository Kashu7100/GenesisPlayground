import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from gs_agent.algos.config.schema import DaggerArgs
from gs_agent.bases.algo import BaseAlgo
from gs_agent.bases.env_wrapper import BaseEnvWrapper
from gs_agent.bases.policy import Policy
from gs_agent.buffers.config.schema import DAGGERBufferKey
from gs_agent.buffers.dagger_buffer import DAGGERBuffer
from gs_agent.modules.critics import StateValueFunction
from gs_agent.modules.models import NetworkFactory
from gs_agent.modules.policies import GaussianPolicy

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")
"""Default device for the algorithm."""

_DEQUE_MAXLEN: Final[int] = 100
"""Max length of the deque for storing episode statistics."""


class DAgger(BaseAlgo):
    """
    Dataset Aggregation (DAgger) algorithm implementation.
    Uses indeterministic policy for behavior cloning.
    """

    def __init__(
        self, env: BaseEnvWrapper, cfg: DaggerArgs, device: torch.device = _DEFAULT_DEVICE
    ) -> None:
        super().__init__(env, cfg, device)
        self._actor_obs_dim = self.env.actor_obs_dim
        self._critic_obs_dim = self.env.critic_obs_dim
        self._action_dim = self.env.action_dim

        self._num_envs = self.env.num_envs
        self._num_steps = cfg.rollout_length

        self.current_iter = 0
        self._rewbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._lenbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._curr_reward_sum = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.float
        )
        self._curr_ep_len = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float)

        self.use_clipped_value_loss = cfg.use_clipped_value_loss

        # Load teacher config first (needed for teacher obs dim)
        self._load_teacher_config(cfg.teacher_config_path)
        # Build actor and critic networks
        self._build_actor_critic()
        if not cfg.teacher_path.is_file():
            raise ValueError("Teacher path must be a file.")
        self._build_teacher(cfg.teacher_path)
        self._build_rollouts()

    def _build_actor_critic(self) -> None:
        policy_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.policy_backbone,
            input_dim=self._actor_obs_dim,
            output_dim=self._action_dim,
            device=self.device,
        )
        print(f"Policy backbone: {policy_backbone}")
        # Use GaussianPolicy (indeterministic) instead of DeterministicPolicy
        self._actor = GaussianPolicy(
            policy_backbone=policy_backbone,
            action_dim=self._action_dim,
        ).to(self.device)
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self.cfg.lr)

        critic_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.critic_backbone,
            input_dim=self._critic_obs_dim,
            output_dim=1,
            device=self.device,
        )
        print(f"Critic backbone: {critic_backbone}")
        self._critic = StateValueFunction(critic_backbone).to(self.device)
        value_lr = self.cfg.value_lr if self.cfg.value_lr is not None else self.cfg.lr
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=value_lr)

        print("Device: ", self.device)

    def _build_teacher(self, teacher_path: Path) -> None:
        teacher_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.teacher_backbone,
            input_dim=self._teacher_obs_dim,
            output_dim=self._action_dim,
            device=self.device,
        )
        self._teacher = GaussianPolicy(
            policy_backbone=teacher_backbone,
            action_dim=self._action_dim,
        ).to(self.device)
        self._teacher.load_state_dict(
            torch.load(teacher_path, map_location=self.device)["actor_state_dict"]
        )
        self._teacher.eval()

        # Copy teacher's standard deviation to student's policy
        with torch.no_grad():
            self._actor.log_std.data.copy_(self._teacher.log_std.data)
        print("Copied teacher's log_std to student policy.")

    def _load_teacher_config(self, teacher_config_path: Path) -> None:
        """Load teacher environment config from yaml file."""
        if not teacher_config_path.is_file():
            raise ValueError(f"Teacher config path must be a file: {teacher_config_path}")

        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path as PathLib

        # Add examples to path to import utils
        examples_path = PathLib(__file__).parent.parent.parent.parent / "examples"
        if str(examples_path) not in sys.path:
            sys.path.insert(0, str(examples_path))

        # Import the appropriate config class based on environment type
        # For now, we'll try to detect it from the yaml or use a generic approach
        # The user should ensure the teacher config matches the environment type
        from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs, MotionEnvArgs, WalkingEnvArgs
        from utils import yaml_to_config  # type: ignore

        # Try to load as different config types
        try:
            self._teacher_env_args = yaml_to_config(teacher_config_path, WalkingEnvArgs)
        except Exception:
            try:
                self._teacher_env_args = yaml_to_config(teacher_config_path, MotionEnvArgs)
            except Exception:
                self._teacher_env_args = yaml_to_config(teacher_config_path, LeggedRobotEnvArgs)

        # Calculate teacher observation dimension
        teacher_obs, _ = self.env.get_observations(obs_args=self._teacher_env_args)
        self._teacher_obs_dim = teacher_obs.shape[-1]
        print(f"Teacher observation dimension: {self._teacher_obs_dim}")

    def _build_rollouts(self) -> None:
        # Use gae_lambda=0 to compute simple discounted returns (no GAE)
        self._rollouts = DAGGERBuffer(
            num_envs=self._num_envs,
            max_steps=self._num_steps,
            actor_obs_size=self._actor_obs_dim,
            critic_obs_size=self._critic_obs_dim,
            action_size=self._action_dim,
            device=self.device,
            gae_gamma=self.cfg.gamma,
            gae_lam=0.0,  # Set to 0 to disable GAE and compute simple returns
        )

    def _collect_rollouts(self, num_steps: int) -> dict[str, Any]:
        """Collect rollouts using DAgger: student acts, teacher provides labels."""
        actor_obs, critic_obs = self.env.get_observations()
        termination_buffer = []
        reward_terms_buffer = []
        info_buffer = []
        with torch.inference_mode():
            # collect rollouts and compute returns & advantages
            for _step in range(num_steps):
                # Student acts (indeterministic)
                student_actions, _ = self._actor(actor_obs, deterministic=True)

                # Get teacher observations using teacher config
                teacher_obs, _ = self.env.get_observations(obs_args=self._teacher_env_args)

                # Teacher provides action labels (deterministic)
                teacher_action, _ = self._teacher(teacher_obs, deterministic=True)

                # Step environment with student actions
                _, reward, terminated, truncated, _extra_infos = self.env.step(student_actions)
                next_actor_obs, next_critic_obs = self.env.get_observations()

                # add next value to reward of truncated steps
                if "time_outs" in _extra_infos:
                    time_outs = _extra_infos["time_outs"]
                    next_values = self._critic(_extra_infos["observations"]["critic"])
                    reward = reward + next_values * time_outs

                # all tensors are of shape: [num_envs, dim]
                transition = {
                    DAGGERBufferKey.ACTOR_OBS: actor_obs,
                    DAGGERBufferKey.CRITIC_OBS: critic_obs,
                    DAGGERBufferKey.TEACHER_ACTIONS: teacher_action,  # Teacher actions
                    DAGGERBufferKey.STUDENT_ACTIONS: student_actions,  # Student actions
                    DAGGERBufferKey.REWARDS: reward,
                    DAGGERBufferKey.DONES: terminated,
                    DAGGERBufferKey.VALUES: self._critic(critic_obs),
                }
                self._rollouts.append(transition)

                # Update episode tracking - handle reward and done sequences
                # Extract tensors from reward and done objects
                self._curr_reward_sum += reward.squeeze(-1)
                self._curr_ep_len += 1

                # Update termination buffer
                termination_buffer.append(_extra_infos["termination"])
                reward_terms_buffer.append(_extra_infos["reward_terms"])
                if "info" in _extra_infos:
                    info_buffer.append(_extra_infos["info"])

                # Check for episode completions and reset tracking
                done_mask = terminated.unsqueeze(-1) | truncated.unsqueeze(-1)
                new_ids = (done_mask > 0).nonzero(as_tuple=False)
                if len(new_ids) > 0:
                    # Vectorized environment
                    self._rewbuffer.extend(
                        self._curr_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self._lenbuffer.extend(self._curr_ep_len[new_ids][:, 0].cpu().numpy().tolist())
                    # Reset tracking
                    self._curr_reward_sum[new_ids] = 0
                    self._curr_ep_len[new_ids] = 0

                actor_obs, critic_obs = next_actor_obs, next_critic_obs

        with torch.no_grad():
            last_value = self._critic(critic_obs)
            self._rollouts.set_final_value(last_value)

        mean_reward = 0.0
        mean_ep_len = 0.0
        if len(self._rewbuffer) > 0:
            mean_reward = statistics.mean(self._rewbuffer)
            mean_ep_len = statistics.mean(self._lenbuffer)

        mean_termination = {}
        mean_reward_terms = {}
        mean_info = {}
        if len(termination_buffer) > 0:
            for key in termination_buffer[0].keys():
                terminations = torch.stack([termination[key] for termination in termination_buffer])
                mean_termination[key] = terminations.to(torch.float).mean().item()
        if len(reward_terms_buffer) > 0:
            for key in reward_terms_buffer[0].keys():
                reward_terms = torch.stack(
                    [reward_term[key] for reward_term in reward_terms_buffer]
                )
                mean_reward_terms[key] = reward_terms.mean().item()
        if len(info_buffer) > 0:
            for key in info_buffer[0].keys():
                infos = torch.tensor([info[key] for info in info_buffer])
                mean_info[key] = infos.mean().item()

        return {
            "mean_reward": mean_reward,
            "mean_ep_len": mean_ep_len,
            "termination": mean_termination,
            "reward_terms": mean_reward_terms,
            "info": mean_info,
        }

    def _train_one_batch(self, mini_batch: dict[DAGGERBufferKey, torch.Tensor]) -> dict[str, Any]:
        """Train one batch of rollouts."""
        actor_obs = mini_batch[DAGGERBufferKey.ACTOR_OBS]
        critic_obs = mini_batch[DAGGERBufferKey.CRITIC_OBS]
        teacher_actions = mini_batch[DAGGERBufferKey.TEACHER_ACTIONS]
        target_values = mini_batch[DAGGERBufferKey.VALUES]
        returns = mini_batch[DAGGERBufferKey.RETURNS]

        # Compute MSE loss for behavior cloning (imitation loss)
        student_actions, _ = self._actor(actor_obs, deterministic=True)
        imitation_loss = (teacher_actions - student_actions).pow(2).mean()

        # Calculate value loss
        values = self._critic(critic_obs)

        if self.use_clipped_value_loss:
            clipped_values = target_values + (values - target_values).clamp(
                -self.cfg.clip_ratio, self.cfg.clip_ratio
            )
            value_loss = (values - returns).pow(2)
            clipped_value_loss = (clipped_values - returns).pow(2)
            value_loss = torch.max(value_loss, clipped_value_loss).mean()
        else:
            value_loss = (returns - values).pow(2).mean()

        # Total loss
        total_loss = imitation_loss + self.cfg.value_loss_coef * value_loss

        # Optimization step
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._actor.parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self._critic.parameters(), self.cfg.max_grad_norm)
        self._critic_optimizer.step()
        self._actor_optimizer.step()

        return {
            "imitation_loss": imitation_loss.item(),
            "value_loss": value_loss.item(),
        }

    def train_one_iteration(self) -> dict[str, Any]:
        """Update policy using the collected experience."""
        t0 = time.time()
        rollout_infos = self._collect_rollouts(num_steps=self._num_steps)
        t1 = time.time()
        rollouts_time = t1 - t0
        fps = (self._num_steps * self._num_envs / rollouts_time) if rollouts_time > 0 else 0

        train_metrics_list: list[dict[str, Any]] = []
        for mini_batch in self._rollouts.minibatch_gen(
            num_mini_batches=self.cfg.num_mini_batches,
            num_epochs=self.cfg.num_epochs,
        ):
            metrics = self._train_one_batch(mini_batch)
            train_metrics_list.append(metrics)
        t2 = time.time()
        train_time = t2 - t1

        self._rollouts.reset()

        iteration_infos = {
            "rollout": {
                "mean_reward": rollout_infos["mean_reward"],
                "mean_length": rollout_infos["mean_ep_len"],
            },
            "train": {
                "imitation_loss": statistics.mean(
                    [metrics["imitation_loss"] for metrics in train_metrics_list]
                ),
                "value_loss": statistics.mean(
                    [metrics["value_loss"] for metrics in train_metrics_list]
                ),
            },
            "speed": {
                "rollout_time": rollouts_time,
                "rollout_fps": fps,
                "train_time": train_time,
                "rollout_step": self._num_steps * self._num_envs,
            },
            "termination": rollout_infos["termination"],
            "reward_terms": rollout_infos["reward_terms"],
            "info": rollout_infos["info"],
        }
        return iteration_infos

    def save(self, path: Path, infos: dict[str, Any] | None = None) -> None:
        saved_dict = {
            "actor_state_dict": self._actor.state_dict(),
            "actor_optimizer_state_dict": self._actor_optimizer.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "critic_optimizer_state_dict": self._critic_optimizer.state_dict(),
            "iter": self.current_iter,
        }
        if infos is not None:
            saved_dict.update(infos)
        torch.save(saved_dict, path)

    def load(self, path: Path, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self._actor.load_state_dict(checkpoint["actor_state_dict"])
        self._critic.load_state_dict(checkpoint["critic_state_dict"])
        if load_optimizer:
            self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.current_iter = checkpoint["iter"]

    def train_mode(self) -> None:
        """Set the algorithm to train mode."""
        self._actor.train()
        self._critic.train()

    def eval_mode(self) -> None:
        """Set the algorithm to eval mode."""
        self._actor.eval()
        self._critic.eval()

    def get_inference_policy(self, device: torch.device | None = None) -> Policy:
        """Get the inference policy for evaluation."""
        self.eval_mode()
        if device is not None:
            self._actor.to(device)
        return self._actor
