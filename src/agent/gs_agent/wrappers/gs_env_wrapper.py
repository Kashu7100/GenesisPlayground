from typing import Any, Final, TypeVar

import torch

from gs_agent.bases.env_wrapper import BaseEnvWrapper

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")

TGSEnv = TypeVar("TGSEnv")


class GenesisEnvWrapper(BaseEnvWrapper):
    def __init__(
        self,
        env: object,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(env, device)
        self.env.reset()
        self._obs_history_len = self.env.args.obs_history_len
        self._obs_history = torch.zeros(
            self.env.num_envs, self.env.actor_obs_dim, self._obs_history_len, device=device
        )

    # ---------------------------
    # BatchEnvWrapper API (batch)
    # ---------------------------
    def reset(self) -> None:
        self.env.reset()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        self.env.reset_idx(envs_idx)
        self._obs_history[envs_idx] = 0.0

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # apply action
        self.env.apply_action(action)
        # get terminated
        terminated = self.env.get_terminated()
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)
        # get truncated
        truncated = self.env.get_truncated()
        if truncated.dim() == 1:
            truncated = truncated.unsqueeze(-1)
        # get reward
        reward, reward_terms = self.env.get_reward()
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        # update history
        self.env.update_history()
        # get extra infos
        extra_infos = self.env.get_extra_infos()
        extra_infos["reward_terms"] = reward_terms
        # reset if terminated or truncated
        done_idx = terminated.nonzero(as_tuple=True)[0]
        if len(done_idx) > 0:
            self.reset_idx(done_idx)
        # get observations
        next_obs, _ = self.env.get_observations(obs_args=None)
        self._obs_history = torch.cat([self._obs_history[..., 1:], next_obs[..., None]], dim=-1)
        obs = self._obs_history.view(self.env.num_envs, -1)
        return obs, reward, terminated, truncated, extra_infos

    def get_observations(self, obs_args: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get observations. Returns both actor and critic observations.

        Args:
            obs_args: Optional environment args to use for observation computation.
                     If None, uses student config.

        Returns:
            Tuple of (actor_obs, critic_obs)
        """
        _, critic_obs = self.env.get_observations(obs_args=obs_args)
        return self._obs_history.view(self.env.num_envs, -1), critic_obs

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    @property
    def actor_obs_dim(self) -> int:
        return self.env.actor_obs_dim * self._obs_history_len

    @property
    def critic_obs_dim(self) -> int:
        return self.env.critic_obs_dim

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def close(self) -> None:
        self.env.close()

    def render(self) -> None:
        self.env.render()
