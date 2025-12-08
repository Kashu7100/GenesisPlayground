import os
import pickle

import torch
import yaml
from tqdm import tqdm

#
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_diff,
    quat_from_angle_axis,
    quat_from_euler,
    quat_inv,
    quat_mul,
    quat_to_angle_axis,
    quat_to_euler,
    quat_to_rotation_6D,
    slerp,
)

_DEFAULT_DEVICE = torch.device("cpu")


class MotionLib:
    def __init__(
        self,
        motion_file: str | None = None,
        device: torch.device = _DEFAULT_DEVICE,
        target_fps: float = 50.0,
        tracking_link_names: list[str] | None = None,
    ) -> None:
        self._device = device
        self._target_fps = target_fps
        self._tracking_link_names = tracking_link_names
        self._motion_obs_steps = None
        if motion_file is not None:
            self._load_motions(motion_file)

    def _load_motions(self, motion_file: str) -> None:
        self._motion_names = []
        self._motion_files = []
        self._link_names = []
        self._dof_names = []
        self._tracking_link_indices = None

        motion_weights = []
        motion_num_frames = []
        motion_lengths = []

        motion_base_pos = []
        motion_base_quat = []
        motion_base_lin_vel = []
        motion_base_ang_vel = []
        motion_base_ang_vel_local = []
        motion_dof_pos = []
        motion_dof_vel = []
        motion_link_pos_global = []
        motion_link_quat_global = []
        motion_link_pos_local = []
        motion_link_quat_local = []
        motion_link_lin_vel = []
        motion_link_ang_vel = []
        motion_foot_contact = []

        full_motion_files, full_motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(full_motion_files)

        for i in tqdm(range(num_motion_files), desc="[MotionLib] Loading motions"):
            curr_file = full_motion_files[i]
            try:
                with open(curr_file, "rb") as f:
                    motion_data = pickle.load(f)

                    if len(self._link_names) == 0:
                        self._link_names = motion_data["link_names"]
                        self._dof_names = motion_data["dof_names"]
                        self._foot_link_indices = motion_data["foot_link_indices"]

                        # Filter to tracking links if specified
                        if self._tracking_link_names is not None:
                            # Find indices of tracking links in the full link_names list
                            tracking_link_indices = []
                            for name in self._tracking_link_names:
                                if name in self._link_names:
                                    tracking_link_indices.append(self._link_names.index(name))
                                else:
                                    raise ValueError(
                                        f"Tracking link name '{name}' not found in motion data link names"
                                    )
                            # Store the tracking link indices for filtering
                            self._tracking_link_indices = tracking_link_indices

                    base_pos = torch.tensor(
                        motion_data["pos"], dtype=torch.float, device=self._device
                    )
                    base_quat = torch.tensor(
                        motion_data["quat"], dtype=torch.float, device=self._device
                    )

                    fps = motion_data["fps"]
                    dt = 1.0 / fps
                    num_frames = base_pos.shape[0]
                    length = dt * (num_frames - 1)

                    base_lin_vel = torch.zeros_like(base_pos)
                    base_lin_vel[:-1, :] = fps * (base_pos[1:, :] - base_pos[:-1, :])
                    base_lin_vel[-1, :] = base_lin_vel[-2, :]
                    base_lin_vel = self.smooth(base_lin_vel, 19, device=self._device)

                    base_ang_vel = torch.zeros_like(base_pos)  # (num_frames, 3)
                    base_dquat = quat_diff(base_quat[:-1], base_quat[1:])
                    base_ang_vel[:-1, :] = fps * quat_to_angle_axis(base_dquat)
                    base_ang_vel[-1, :] = base_ang_vel[-2, :]
                    base_ang_vel = self.smooth(base_ang_vel, 19, device=self._device)

                    dof_pos = torch.tensor(
                        motion_data["dof_pos"], dtype=torch.float, device=self._device
                    )
                    dof_vel = torch.zeros_like(dof_pos)  # (num_frames, num_dof)
                    dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
                    dof_vel[-1, :] = dof_vel[-2, :]
                    dof_vel = self.smooth(dof_vel, 19, device=self._device)

                    link_pos_global = torch.tensor(
                        motion_data["link_pos"], dtype=torch.float, device=self._device
                    )
                    link_quat_global = torch.tensor(
                        motion_data["link_quat"], dtype=torch.float, device=self._device
                    )

                    # Filter to tracking links if specified
                    if self._tracking_link_indices is not None:
                        link_pos_global = link_pos_global[:, self._tracking_link_indices, :]
                        link_quat_global = link_quat_global[:, self._tracking_link_indices, :]

                    foot_contact = torch.tensor(
                        motion_data["foot_contact"], dtype=torch.float, device=self._device
                    )

                    # Resample to target FPS if requested
                    target_fps_curr = float(self._target_fps)

                    if abs(target_fps_curr - fps) > 1e-6:
                        # time length stays the same
                        new_num_frames = int(round(length * target_fps_curr)) + 1
                        t = torch.linspace(0.0, length, steps=new_num_frames, device=self._device)
                        # compute blend weights against original frames
                        phase = torch.clip(t / length, 0.0, 1.0)
                        idx0 = (phase * (num_frames - 1)).long()
                        idx1 = torch.min(
                            idx0 + 1, torch.tensor(num_frames - 1, device=self._device)
                        )
                        blend = phase * (num_frames - 1) - idx0.float()
                        blend_u = blend.unsqueeze(-1)

                        # positions, dof: linear
                        base_pos = (1.0 - blend_u) * base_pos[idx0] + blend_u * base_pos[idx1]
                        dof_pos = (1.0 - blend_u) * dof_pos[idx0] + blend_u * dof_pos[idx1]
                        foot_contact = 1 - (1 - foot_contact[idx0]) * (1 - foot_contact[idx1])
                        link_pos_global = (1.0 - blend_u.unsqueeze(1)) * link_pos_global[
                            idx0
                        ] + blend_u.unsqueeze(1) * link_pos_global[idx1]

                        # quaternions: slerp
                        base_quat = slerp(base_quat[idx0], base_quat[idx1], blend)
                        link_quat_global = slerp(
                            link_quat_global[idx0],
                            link_quat_global[idx1],
                            blend[:, None].repeat(1, link_quat_global.shape[1]),
                        )

                        # update meta based on resampled length
                        fps = target_fps_curr
                        dt = 1.0 / fps
                        num_frames = base_pos.shape[0]
                        length = dt * (num_frames - 1)
                    else:
                        # ensure library fps is set
                        fps = target_fps_curr
                        dt = 1.0 / fps

                    # recompute velocities at current fps
                    base_lin_vel = torch.zeros_like(base_pos)
                    base_lin_vel[:-1, :] = fps * (base_pos[1:, :] - base_pos[:-1, :])
                    base_lin_vel[-1, :] = base_lin_vel[-2, :]
                    base_lin_vel = self.smooth(base_lin_vel, 19, device=self._device)

                    base_ang_vel = torch.zeros_like(base_pos)  # (num_frames, 3)
                    base_dquat = quat_diff(base_quat[:-1], base_quat[1:])
                    base_ang_vel[:-1, :] = fps * quat_to_angle_axis(base_dquat)
                    base_ang_vel[-1, :] = base_ang_vel[-2, :]
                    base_ang_vel = self.smooth(base_ang_vel, 19, device=self._device)

                    base_ang_vel_local = quat_apply(quat_inv(base_quat), base_ang_vel)

                    dof_vel = torch.zeros_like(dof_pos)  # (num_frames, num_dof)
                    dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
                    dof_vel[-1, :] = dof_vel[-2, :]
                    dof_vel = self.smooth(dof_vel, 19, device=self._device)

                    # recompute local link transforms with yaw-only removal from base
                    relative_link_pos_global = link_pos_global.clone()
                    relative_link_pos_global[:, :, :2] -= base_pos[:, None, :2]
                    base_euler = quat_to_euler(base_quat)
                    base_euler[:, :2] = 0.0
                    batched_inv_quat_yaw = quat_from_euler(
                        -base_euler[:, None, :].repeat(1, link_pos_global.shape[1], 1)
                    )
                    link_pos_local = quat_apply(batched_inv_quat_yaw, relative_link_pos_global)
                    link_quat_local = quat_mul(batched_inv_quat_yaw, link_quat_global)

                    # compute link velocities (global)
                    link_lin_vel = torch.zeros_like(link_pos_global)  # (num_frames, num_links, 3)
                    link_lin_vel[:-1, :, :] = fps * (
                        link_pos_global[1:, :, :] - link_pos_global[:-1, :, :]
                    )
                    link_lin_vel[-1, :, :] = link_lin_vel[-2, :, :]
                    # Smooth each link separately across frames
                    link_lin_vel_flat = link_lin_vel.reshape(
                        link_lin_vel.shape[0], -1
                    )  # (num_frames, num_links * 3)
                    link_lin_vel_flat = self.smooth(link_lin_vel_flat, 19, device=self._device)
                    link_lin_vel = link_lin_vel_flat.reshape(link_pos_global.shape)

                    link_ang_vel = torch.zeros_like(link_pos_global)  # (num_frames, num_links, 3)
                    link_dquat_global = quat_diff(
                        link_quat_global[:-1], link_quat_global[1:]
                    )  # (num_frames-1, num_links, 4)
                    link_ang_vel[:-1, :, :] = fps * quat_to_angle_axis(link_dquat_global)
                    link_ang_vel[-1, :, :] = link_ang_vel[-2, :, :]
                    # Smooth each link separately across frames
                    link_ang_vel_flat = link_ang_vel.reshape(
                        link_ang_vel.shape[0], -1
                    )  # (num_frames, num_links * 3)
                    link_ang_vel_flat = self.smooth(link_ang_vel_flat, 19, device=self._device)
                    link_ang_vel = link_ang_vel_flat.reshape(link_pos_global.shape)

                    self._motion_names.append(os.path.basename(curr_file))
                    self._motion_files.append(curr_file)

                    motion_weights.append(full_motion_weights[i])
                    motion_num_frames.append(num_frames)
                    motion_lengths.append(length)

                    motion_base_pos.append(base_pos)
                    motion_base_quat.append(base_quat)
                    motion_base_lin_vel.append(base_lin_vel)
                    motion_base_ang_vel.append(base_ang_vel)
                    motion_base_ang_vel_local.append(base_ang_vel_local)
                    motion_dof_pos.append(dof_pos)
                    motion_dof_vel.append(dof_vel)
                    motion_link_pos_global.append(link_pos_global)
                    motion_link_quat_global.append(link_quat_global)
                    motion_link_pos_local.append(link_pos_local)
                    motion_link_quat_local.append(link_quat_local)
                    motion_link_lin_vel.append(link_lin_vel)
                    motion_link_ang_vel.append(link_ang_vel)
                    motion_foot_contact.append(foot_contact)

            except Exception as e:
                print(f"Error loading motion file {curr_file}: {e}")
                continue

        assert len(self._link_names) > 0, "Link names list is empty"
        assert len(self._dof_names) > 0, "Dof names list is empty"

        motion_weights = torch.tensor(motion_weights, dtype=torch.float, device=self._device)
        self._motion_weights = motion_weights / torch.sum(motion_weights)
        self._motion_num_frames = torch.tensor(
            motion_num_frames, dtype=torch.long, device=self._device
        )
        self._motion_lengths = torch.tensor(motion_lengths, dtype=torch.float, device=self._device)

        self._motion_base_pos = torch.cat(motion_base_pos, dim=0)
        self._motion_base_quat = torch.cat(motion_base_quat, dim=0)
        self._motion_base_lin_vel = torch.cat(motion_base_lin_vel, dim=0)
        self._motion_base_ang_vel = torch.cat(motion_base_ang_vel, dim=0)
        self._motion_base_ang_vel_local = torch.cat(motion_base_ang_vel_local, dim=0)
        self._motion_dof_pos = torch.cat(motion_dof_pos, dim=0)
        self._motion_dof_vel = torch.cat(motion_dof_vel, dim=0)
        self._motion_link_pos_global = torch.cat(motion_link_pos_global, dim=0)
        self._motion_link_quat_global = torch.cat(motion_link_quat_global, dim=0)
        self._motion_link_pos_local = torch.cat(motion_link_pos_local, dim=0)
        self._motion_link_quat_local = torch.cat(motion_link_quat_local, dim=0)
        self._motion_link_lin_vel = torch.cat(motion_link_lin_vel, dim=0)
        self._motion_link_ang_vel = torch.cat(motion_link_ang_vel, dim=0)
        self._motion_foot_contact = torch.cat(motion_foot_contact, dim=0)

        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)  # prefix sum of num frames

        self._motion_ids = torch.arange(self.num_motions, dtype=torch.long, device=self._device)

        print(
            f"Loaded {self.num_motions:d} motions with a total length of {self.total_length:.3f}s."
        )

    def sample_motion_ids(
        self, n: int, motion_difficulty: torch.Tensor | None = None
    ) -> torch.Tensor:
        if motion_difficulty is not None:
            motion_prob = self._motion_weights * motion_difficulty
        else:
            motion_prob = self._motion_weights
        motion_ids = torch.multinomial(motion_prob, num_samples=n, replacement=True)
        return motion_ids

    def sample_motion_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        # Sample integer steps uniformly and convert to times by dividing by fps
        n_steps = self._motion_num_frames[motion_ids] - 1
        phase = torch.rand(motion_ids.shape, device=self._device)
        steps = torch.round(phase * n_steps.float()).long()
        steps = torch.clamp(steps, min=0)  # safety
        motion_times = steps.float() / float(self.fps)
        return motion_times

    def _fetch_motion_files(
        self, motion_file: str, motion_weight: float = 1.0
    ) -> tuple[list[str], list[float]]:
        # Recursively expand YAML motion manifests into flat file and weight lists.
        if motion_file.endswith(".yaml"):
            all_files: list[str] = []
            all_weights: list[float] = []
            try:
                with open(motion_file) as f:
                    motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            except Exception:
                return [], []
            motion_base_path = motion_config["root_path"]
            motion_list = motion_config["motions"]
            for motion_entry in motion_list:
                curr_file = os.path.join(motion_base_path, motion_entry["file"])
                curr_weight = float(motion_entry.get("weight", 1.0))
                assert curr_weight >= 0
                sub_files, sub_weights = self._fetch_motion_files(
                    curr_file, curr_weight * motion_weight
                )
                all_files.extend(sub_files)
                all_weights.extend(sub_weights)
            return all_files, all_weights
        else:
            return [motion_file], [motion_weight]

    def get_observed_steps(self, observed_steps: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        """Convert observed steps lists into tensors on the correct device."""
        steps_map: dict[str, torch.Tensor] = {}
        for term in observed_steps.keys():
            steps_map[term] = torch.tensor(
                observed_steps[term], dtype=torch.long, device=self._device
            )
        return steps_map

    def get_motion_frame(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert motion_times.min() >= 0.0, "motion_times must be non-negative"
        # snap to discrete frame grid using unified fps and clamp within motion length
        fps = self.fps
        motion_len = self._motion_lengths[motion_ids] - 1.0 / fps
        motion_times = torch.min(motion_times, motion_len)
        steps = torch.round(motion_times * fps).long()

        frame_start_idx = self._motion_start_idx[motion_ids]
        frame_idx = frame_start_idx + steps + 1

        base_pos = self._motion_base_pos[frame_idx]
        base_quat = self._motion_base_quat[frame_idx]
        base_lin_vel = self._motion_base_lin_vel[frame_idx]
        base_ang_vel = self._motion_base_ang_vel[frame_idx]
        dof_pos = self._motion_dof_pos[frame_idx]
        dof_vel = self._motion_dof_vel[frame_idx]

        return (
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            dof_pos,
            dof_vel,
        )

    def get_ref_motion_frame(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert motion_times.min() >= 0.0, "motion_times must be non-negative"
        # snap to discrete frame grid using unified fps and clamp within motion length
        fps = self.fps
        motion_len = self._motion_lengths[motion_ids] - 1.0 / fps
        motion_times = torch.min(motion_times, motion_len)
        steps = torch.round(motion_times * fps).long()

        frame_start_idx = self._motion_start_idx[motion_ids]
        frame_idx = frame_start_idx + steps + 1

        base_pos = self._motion_base_pos[frame_idx]
        base_quat = self._motion_base_quat[frame_idx]
        base_lin_vel = self._motion_base_lin_vel[frame_idx]
        base_ang_vel = self._motion_base_ang_vel[frame_idx]
        base_ang_vel_local = self._motion_base_ang_vel_local[frame_idx]
        dof_pos = self._motion_dof_pos[frame_idx]
        dof_vel = self._motion_dof_vel[frame_idx]
        link_pos_local = self._motion_link_pos_local[frame_idx]
        link_quat_local = self._motion_link_quat_local[frame_idx]
        link_lin_vel = self._motion_link_lin_vel[frame_idx]
        link_ang_vel = self._motion_link_ang_vel[frame_idx]
        foot_contact = self._motion_foot_contact[frame_idx]

        return (
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            base_ang_vel_local,
            dof_pos,
            dof_vel,
            link_pos_local,
            link_quat_local,
            link_lin_vel,
            link_ang_vel,
            foot_contact,
        )

    def get_motion_future_obs(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        observed_steps: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute current-frame and future-step motion observations.

        Returns:
            (curr_obs_dict, future_obs_dict)
        """
        if len(observed_steps) == 0:
            return {}, {}
        assert motion_times.min() >= 0.0, "motion_times must be non-negative"
        fps = self.fps
        motion_len = self._motion_lengths[motion_ids] - 1.0 / fps
        motion_times = torch.min(motion_times, motion_len)
        steps = torch.round(motion_times * fps).long()

        frame_start_idx = self._motion_start_idx[motion_ids]
        max_steps = self._motion_num_frames[motion_ids] - 1

        # current frame (step = 0)
        curr_idx = frame_start_idx + steps
        curr_obs: dict[str, torch.Tensor] = {}
        for key in observed_steps.keys():
            tensor = getattr(self, f"_motion_{key}")
            curr_obs[key] = tensor[curr_idx]

        def gather(term: str) -> torch.Tensor:
            steps_tensor = observed_steps[term]
            future_steps = steps[:, None] + steps_tensor[None, :]
            future_steps = torch.minimum(future_steps, max_steps[:, None])
            future_idx_local = frame_start_idx[:, None] + future_steps  # (B, K)
            B, K = future_idx_local.shape
            tensor = getattr(self, f"_motion_{term}")
            flat = tensor[future_idx_local.reshape(-1)]
            return flat.reshape(B, K, *tensor.shape[1:])

        future_obs_dict: dict[str, torch.Tensor] = {}
        for key in observed_steps.keys():
            future_obs_dict[key] = gather(key)
        return curr_obs, future_obs_dict

    def get_link_idx_local_by_name(self, name: str) -> int:
        return self._link_names.index(name)

    def get_joint_idx_by_name(self, name: str) -> int:
        return self._dof_names.index(name)

    def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_lengths[motion_ids]

    def get_motion_num_frames(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_num_frames[motion_ids]

    def get_motion_weights(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_weights[motion_ids]

    @staticmethod
    def smooth(x: torch.Tensor, box_pts: int, device: torch.device) -> torch.Tensor:
        box = torch.ones(box_pts, device=device) / box_pts
        num_channels = x.shape[1]
        x_reshaped = x.T.unsqueeze(0)
        smoothed = torch.nn.functional.conv1d(
            x_reshaped,
            box.view(1, 1, -1).expand(num_channels, 1, -1),
            groups=num_channels,
            padding="same",
        )
        return smoothed.squeeze(0).T

    @property
    def link_names(self) -> list[str]:
        return self._link_names

    @property
    def tracking_link_names(self) -> list[str]:
        assert self._tracking_link_names is not None
        return self._tracking_link_names

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @property
    def foot_link_indices(self) -> list[int]:
        return self._foot_link_indices

    @property
    def num_motions(self) -> int:
        return self._motion_weights.shape[0]

    @property
    def motion_names(self) -> list[str]:
        return self._motion_names

    @property
    def total_length(self) -> float:
        return torch.sum(self._motion_lengths).item()

    @property
    def fps(self) -> float:
        return self._target_fps


def batched_global_to_local(base_quat: torch.Tensor, global_vec: torch.Tensor) -> torch.Tensor:
    assert base_quat.shape[0] == global_vec.shape[0]
    global_vec_shape = global_vec.shape
    global_vec = global_vec.reshape(global_vec_shape[0], -1, global_vec_shape[-1])
    B, L, D = global_vec.shape
    global_flat = global_vec.reshape(B * L, D)
    quat_rep = base_quat[:, None, :].repeat(1, L, 1).reshape(B * L, 4)
    if D == 3:
        local_flat = quat_apply(quat_inv(quat_rep), global_flat)
    elif D == 4:
        local_flat = quat_mul(quat_inv(quat_rep), global_flat)
    else:
        raise ValueError(
            f"Global vector shape must be (B, L, 3) or (B, L, 4), but got {global_flat.shape}"
        )
    return local_flat.reshape(global_vec_shape)


def build_motion_obs_from_dict(
    curr_obs: dict[str, torch.Tensor],
    future_obs: dict[str, torch.Tensor],
    envs_idx: torch.Tensor,
    tracking_link_idx_local: list[int] | None = None,
    base_quat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform motion observations into local-yaw space and 6D rotations.

    Returns processed (curr_dict, future_dict), without touching self.ref_ variables.
    """
    B = envs_idx.shape[0]
    motion_obs_list: list[torch.Tensor] = []

    # Compute yaw quaternion from current base quat if available
    quat_yaw = quat_from_angle_axis(
        quat_to_euler(curr_obs["base_quat"])[:, -1],
        torch.tensor([0, 0, 1], device=curr_obs["base_quat"].device, dtype=torch.float),
    )

    if "base_pos" in future_obs:
        pos_diff = future_obs["base_pos"] - curr_obs["base_pos"][:, None, :]
        motion_obs_list.append(batched_global_to_local(quat_yaw, pos_diff).reshape(B, -1))
    if "base_quat" in future_obs:
        qy = quat_yaw[:, None, :].repeat(1, future_obs["base_quat"].shape[1], 1)
        base_quat_local = quat_mul(quat_inv(qy), future_obs["base_quat"])
        if base_quat is None:
            base_quat = curr_obs["base_quat"][:, None, :]
        else:
            base_quat = base_quat[:, None, :].repeat(1, future_obs["base_quat"].shape[1], 1)
        base_quat_diff = quat_mul(quat_inv(base_quat), future_obs["base_quat"])
        motion_obs_list.append(quat_to_rotation_6D(base_quat_local).reshape(B, -1))
        motion_obs_list.append(quat_to_rotation_6D(base_quat_diff).reshape(B, -1))
    if "base_lin_vel" in future_obs:
        motion_obs_list.append(
            batched_global_to_local(quat_yaw, future_obs["base_lin_vel"]).reshape(B, -1)
        )
    if "base_ang_vel" in future_obs:
        motion_obs_list.append(
            batched_global_to_local(quat_yaw, future_obs["base_ang_vel"]).reshape(B, -1)
        )
    if "base_ang_vel_local" in future_obs:
        motion_obs_list.append(future_obs["base_ang_vel_local"].reshape(B, -1))
    if "dof_pos" in future_obs:
        motion_obs_list.append(future_obs["dof_pos"].reshape(B, -1))
    if "dof_vel" in future_obs:
        motion_obs_list.append(0.1 * future_obs["dof_vel"].reshape(B, -1))
    if "link_pos_local" in future_obs:
        if tracking_link_idx_local is not None:
            tracking_link_pos_local = future_obs["link_pos_local"][:, :, tracking_link_idx_local, :]
        else:
            tracking_link_pos_local = future_obs["link_pos_local"]
        motion_obs_list.append(tracking_link_pos_local.reshape(B, -1))
    if "link_quat_local" in future_obs:
        if tracking_link_idx_local is not None:
            tracking_link_quat_local = future_obs["link_quat_local"][
                :, :, tracking_link_idx_local, :
            ]
        else:
            tracking_link_quat_local = future_obs["link_quat_local"]
        motion_obs_list.append(quat_to_rotation_6D(tracking_link_quat_local).reshape(B, -1))
    if "link_lin_vel" in future_obs:
        if tracking_link_idx_local is not None:
            tracking_link_lin_vel = future_obs["link_lin_vel"][:, :, tracking_link_idx_local, :]
        else:
            tracking_link_lin_vel = future_obs["link_lin_vel"]
        motion_obs_list.append(
            batched_global_to_local(quat_yaw, tracking_link_lin_vel).reshape(B, -1)
        )
    if "link_ang_vel" in future_obs:
        if tracking_link_idx_local is not None:
            tracking_link_ang_vel = future_obs["link_ang_vel"][:, :, tracking_link_idx_local, :]
        else:
            tracking_link_ang_vel = future_obs["link_ang_vel"]
        motion_obs_list.append(
            batched_global_to_local(quat_yaw, tracking_link_ang_vel).reshape(B, -1)
        )
    if "foot_contact" in future_obs:
        motion_obs_list.append(future_obs["foot_contact"].reshape(B, -1))

    return torch.cat(motion_obs_list, dim=-1)
