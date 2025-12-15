from collections.abc import Iterator
from typing import Final

import torch
from tensordict import TensorDict

from gs_agent.bases.buffer import BaseBuffer
from gs_agent.buffers.config.schema import DAGGERBufferKey
from gs_agent.buffers.gae_buffer import compute_gae

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")


class DAGGERBuffer(BaseBuffer):
    """
    A fixed-size buffer for storing rollouts for multi-env DAgger training.
    """

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        actor_obs_size: int,
        critic_obs_size: int,
        action_size: int,
        device: torch.device = _DEFAULT_DEVICE,
        gae_gamma: float = 0.98,
        gae_lam: float = 0.95,
    ) -> None:
        """
        Args:
            num_envs (int): Number of parallel environments.
            max_steps (int): Maximum number of steps to store in the buffer.
            actor_obs_size (int): Dimension of the actor observation space.
            critic_obs_size (int): Dimension of the critic observation space.
            action_size (int): Dimension of the action space.
            device (torch.device): Device to store the buffer on (default: CPU).
            gae_gamma (float): Discount factor for GAE.
            gae_lam (float): Lambda parameter for GAE.
        """
        super().__init__()
        self._num_envs = num_envs
        self._max_steps = max_steps
        self._actor_obs_dim = actor_obs_size
        self._critic_obs_dim = critic_obs_size
        self._action_dim = action_size
        self._device = device

        # GAE parameters (configurable)
        self._gae_gamma = gae_gamma
        self._gae_lam = gae_lam

        self._normalize_adv = True  # Whether to normalize advantages

        # Track current size dynamically
        self._idx, self._idx_hidden = 0, 0

        self._final_value = None

        # Initialize buffer
        self._buffer = self._init_buffers(actor_obs_size, critic_obs_size, action_size)

    def _init_buffers(self, actor_obs_dim: int, critic_obs_dim: int, action_dim: int) -> TensorDict:
        max_steps, num_envs = self._max_steps, self._num_envs
        buffer = TensorDict(
            {
                DAGGERBufferKey.ACTOR_OBS: torch.zeros(max_steps, num_envs, actor_obs_dim),
                DAGGERBufferKey.CRITIC_OBS: torch.zeros(max_steps, num_envs, critic_obs_dim),
                DAGGERBufferKey.TEACHER_ACTIONS: torch.zeros(max_steps, num_envs, action_dim),
                DAGGERBufferKey.REWARDS: torch.zeros(max_steps, num_envs, 1),
                DAGGERBufferKey.DONES: torch.zeros(max_steps, num_envs, 1).byte(),
                DAGGERBufferKey.VALUES: torch.zeros(max_steps, num_envs, 1),
            },
            batch_size=[self._max_steps, self._num_envs],
            device=self._device,
        )
        return buffer

    def reset(self) -> None:
        self._idx = 0
        self._final_value = None

    def append(self, transition: dict[DAGGERBufferKey, torch.Tensor]) -> None:
        if self._idx >= self._max_steps:
            raise ValueError(f"Buffer full! Cannot append more than {self._max_steps} steps.")
        idx = self._idx
        self._buffer[DAGGERBufferKey.ACTOR_OBS][idx] = transition[DAGGERBufferKey.ACTOR_OBS]
        self._buffer[DAGGERBufferKey.CRITIC_OBS][idx] = transition[DAGGERBufferKey.CRITIC_OBS]
        self._buffer[DAGGERBufferKey.TEACHER_ACTIONS][idx] = transition[
            DAGGERBufferKey.TEACHER_ACTIONS
        ]
        self._buffer[DAGGERBufferKey.REWARDS][idx] = transition[DAGGERBufferKey.REWARDS]
        self._buffer[DAGGERBufferKey.DONES][idx] = transition[DAGGERBufferKey.DONES]
        self._buffer[DAGGERBufferKey.VALUES][idx] = transition[DAGGERBufferKey.VALUES]

        # Increment index
        self._idx += 1

    def set_final_value(self, final_value: torch.Tensor) -> None:
        """Set the final value for bootstrapping incomplete episodes.

        This is used to bootstrap the value function for episodes that didn't
        naturally terminate, which is important for short horizon policy updates.

        Args:
            final_value: Value estimate for the final observation [B, 1]
        """
        self._final_value = final_value

    def is_full(self) -> bool:
        return self._idx >= self._max_steps

    def __len__(self) -> int:
        return self._idx * self._num_envs

    def minibatch_gen(
        self, num_mini_batches: int, num_epochs: int, shuffle: bool = True
    ) -> Iterator[dict[DAGGERBufferKey, torch.Tensor]]:
        _, returns = compute_gae(
            rewards=self._buffer[DAGGERBufferKey.REWARDS],
            values=self._buffer[DAGGERBufferKey.VALUES],
            dones=self._buffer[DAGGERBufferKey.DONES],
            final_value=self._final_value,
            gamma=self._gae_gamma,
            gae_lambda=self._gae_lam,
        )

        # Get total number of timesteps and batch size
        total_samples = self._max_steps * self._num_envs
        # Create indices for all samples on the same device as the tensors
        base_indices = torch.arange(total_samples, device=self._device)

        for _epoch in range(num_epochs):
            # Split into minibatches
            if shuffle:
                perm_indices = base_indices[torch.randperm(total_samples, device=self._device)]
            else:
                perm_indices = base_indices

            buckets = torch.tensor_split(perm_indices, num_mini_batches)  # not torch.split here
            for bucket in buckets:
                if bucket.numel() == 0:
                    continue

                t_idx = bucket // self._num_envs
                b_idx = bucket % self._num_envs
                mini_batch_size = bucket.numel()
                yield {
                    DAGGERBufferKey.ACTOR_OBS: self._buffer[DAGGERBufferKey.ACTOR_OBS][
                        t_idx, b_idx
                    ].reshape(mini_batch_size, -1),
                    DAGGERBufferKey.CRITIC_OBS: self._buffer[DAGGERBufferKey.CRITIC_OBS][
                        t_idx, b_idx
                    ].reshape(mini_batch_size, -1),
                    DAGGERBufferKey.TEACHER_ACTIONS: self._buffer[DAGGERBufferKey.TEACHER_ACTIONS][
                        t_idx, b_idx
                    ].reshape(mini_batch_size, -1),
                    DAGGERBufferKey.VALUES: self._buffer[DAGGERBufferKey.VALUES][
                        t_idx, b_idx
                    ].reshape(mini_batch_size, -1),
                    DAGGERBufferKey.RETURNS: returns[t_idx, b_idx].reshape(mini_batch_size, -1),
                }
