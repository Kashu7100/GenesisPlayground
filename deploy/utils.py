import json

import redis
import torch
from gs_env.common.utils.math_utils import (
    pose_diff_quat,
    pose_mul_quat,
    quat_apply,
    quat_diff,
    quat_from_angle_axis,
    quat_from_euler,
    quat_inv,
    quat_mul,
    quat_to_angle_axis,
    quat_to_euler,
    quat_to_rotation_6D,
)


class RedisClient:
    def __init__(
        self,
        url: str,
        key: str,
        device: str,
        num_tracking_links: int = 0,
    ) -> None:
        self._r = redis.from_url(url)
        self._key = key
        self._device = device
        self._dof_dim = 29
        self.num_tracking_links = num_tracking_links
        # Ref variables (after transformations)
        self.ref_dof_pos = torch.zeros(1, self._dof_dim, device=device)
        self.ref_dof_vel = torch.zeros(1, self._dof_dim, device=device)
        self.ref_base_pos = torch.zeros(1, 3, device=device)
        self.last_ref_base_pos = torch.zeros(1, 3, device=device)
        self.ref_base_quat = torch.zeros(1, 4, device=device)
        self.last_ref_base_quat = torch.zeros(1, 4, device=device)
        self.ref_base_euler = torch.zeros(1, 3, device=device)
        self.ref_base_rotation_6D = torch.zeros(1, 6, device=device)
        self.ref_base_lin_vel_local = torch.zeros(1, 3, device=device)
        self.ref_base_ang_vel_local = torch.zeros(1, 3, device=device)
        self.ref_foot_contact = torch.ones(1, 2, device=device)
        self.link_pos_local_yaw = torch.zeros(1, self.num_tracking_links, 3, device=device)
        self.link_quat_local_yaw = torch.zeros(1, self.num_tracking_links, 4, device=device)
        self.link_lin_vel = torch.zeros(1, self.num_tracking_links, 3, device=device)
        self.link_ang_vel = torch.zeros(1, self.num_tracking_links, 3, device=device)
        self._zero_all()

        # Yaw difference quaternion (stored and applied to all subsequent updates)
        self._yaw_diff_quat = torch.zeros(1, 4, device=device)
        self._yaw_diff_quat[:, 0] = 1.0
        # Motion obs element selection (None or empty => compute none by default)
        self._motion_obs_elements: set[str] | None = None

    def _zero_all(self) -> None:
        self.base_pos_timestamp = -1
        self.base_quat_timestamp = -1
        self.base_lin_vel_timestamp = -1
        self.base_ang_vel_timestamp = -1
        self.base_ang_vel_local_timestamp = -1
        self.dof_pos_timestamp = -1
        self.dof_vel_timestamp = -1
        self.link_pos_local_timestamp = -1
        self.link_quat_local_timestamp = -1
        self.foot_contact_timestamp = -1
        self._r.set(f"{self._key}:timestamp:base_pos", -1)
        self._r.set(f"{self._key}:timestamp:base_quat", -1)
        self._r.set(f"{self._key}:timestamp:base_lin_vel", -1)
        self._r.set(f"{self._key}:timestamp:base_ang_vel", -1)
        self._r.set(f"{self._key}:timestamp:base_ang_vel_local", -1)
        self._r.set(f"{self._key}:timestamp:dof_pos", -1)
        self._r.set(f"{self._key}:timestamp:dof_vel", -1)
        self._r.set(f"{self._key}:timestamp:link_pos_local", -1)
        self._r.set(f"{self._key}:timestamp:link_quat_local", -1)
        self._r.set(f"{self._key}:timestamp:link_lin_vel", -1)
        self._r.set(f"{self._key}:timestamp:link_ang_vel", -1)
        self._r.set(f"{self._key}:timestamp:foot_contact", -1)
        # Zero ref variables
        self.ref_dof_pos.zero_()
        self.ref_dof_vel.zero_()
        self.ref_base_pos.zero_()
        self.ref_base_pos[0, 2] = 0.76
        self.last_ref_base_pos.copy_(self.ref_base_pos)
        self.ref_base_quat.zero_()
        self.ref_base_quat[:, 0] = 1.0
        self.last_ref_base_quat.copy_(self.ref_base_quat)
        self.ref_base_euler.zero_()
        self.ref_base_rotation_6D.zero_()
        self.ref_base_rotation_6D[:, [0, 4]] = 1.0
        self.ref_base_lin_vel_local.zero_()
        self.ref_base_ang_vel_local.zero_()
        self.ref_foot_contact[:] = 1.0
        self.link_pos_local_yaw.zero_()
        self.link_pos_local_yaw = torch.tensor(
            [
                [0.0, 0.13, -0.73],
                [0.0, -0.13, -0.73],
                [0.37, 0.1, 0.28],
                [0.37, -0.1, 0.28],
                [0.0, 0.0, 0.044],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=self._device,
        )[None, : self.num_tracking_links, :]
        self.link_quat_local_yaw.zero_()
        self.link_quat_local_yaw[:, :, 0] = 1.0
        self.link_lin_vel.zero_()
        self.link_ang_vel.zero_()

    def _fit_dim(self, data: list[float], dim: int) -> torch.Tensor:
        out = torch.zeros(1, dim, device=self._device)
        if len(data) > 0:
            n = min(dim, len(data))
            out[0, :n] = torch.tensor(data[:n], dtype=torch.float32, device=self._device)
        return out

    def _fit_link_data(self, data: list[float], n_links: int, dim_per_link: int) -> torch.Tensor:
        """Reshape flattened link data into (1, n_links, dim_per_link) tensor."""
        out = torch.zeros(1, n_links, dim_per_link, device=self._device)
        if len(data) > 0 and n_links > 0:
            expected_size = n_links * dim_per_link
            n = min(expected_size, len(data))
            reshaped = torch.tensor(data[:n], dtype=torch.float32, device=self._device).view(
                -1, dim_per_link
            )
            actual_links = min(n_links, reshaped.shape[0])
            out[0, :actual_links, :] = reshaped[:actual_links, :]
        return out

    def _get_field(self, field_name: str, default: list[float]) -> list[float]:
        """Get a field from Redis as a JSON-decoded list."""
        # Try both formats: with :motion: prefix (new format) and without (old format)
        raw = self._r.get(f"{self._key}:motion:{field_name}")
        if raw is None:
            raw = self._r.get(f"{self._key}:{field_name}")
        if raw is None:
            return default
        s = raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else str(raw)
        return json.loads(s)

    def _get_timestamp(self, field_name: str) -> int:
        """Get a timestamp from Redis for a given field."""
        # Try both formats: with :timestamp: prefix (new format) and without (old format)
        raw = self._r.get(f"{self._key}:timestamp:{field_name}")
        if raw is None:
            # Fallback: check if field exists at all
            return -1
        try:
            if isinstance(raw, bytes | bytearray):
                return int(raw.decode("utf-8"))
            return int(str(raw))
        except (ValueError, TypeError):
            return -1

    def update(self) -> None:
        try:
            # Update base_pos if timestamp changed
            new_timestamp = self._get_timestamp("base_pos")
            if new_timestamp != self.base_pos_timestamp:
                # Store previous value before updating
                self.last_ref_base_pos.copy_(self.ref_base_pos)
                base_pos = torch.tensor(
                    self._get_field("base_pos", [0.0, 0.0, 0.0]),
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, 3)
                self.ref_base_pos = quat_apply(self._yaw_diff_quat, base_pos)
                self.base_pos_timestamp = new_timestamp

            # Update base_quat if timestamp changed
            new_timestamp = self._get_timestamp("base_quat")
            if new_timestamp != self.base_quat_timestamp:
                # Store previous value before updating
                self.last_ref_base_quat.copy_(self.ref_base_quat)
                base_quat = torch.tensor(
                    self._get_field("base_quat", [1.0, 0.0, 0.0, 0.0]),
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, 4)
                self.ref_base_quat = quat_mul(self._yaw_diff_quat, base_quat)
                # self.ref_base_quat = base_quat
                self.base_quat_timestamp = new_timestamp
                # Update derived quantities when quat changes
                self.ref_base_euler = quat_to_euler(self.ref_base_quat)
                self.ref_base_rotation_6D = quat_to_rotation_6D(self.ref_base_quat)

            # Update base_lin_vel if timestamp changed
            new_timestamp = self._get_timestamp("base_lin_vel")
            if new_timestamp != self.base_lin_vel_timestamp:
                base_lin_vel = torch.tensor(
                    self._get_field("base_lin_vel", [0.0, 0.0, 0.0]),
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, 3)
                # Apply yaw_diff_quat first (equivalent to base_yaw_offset_quat in motion_env)
                base_lin_vel = quat_apply(self._yaw_diff_quat, base_lin_vel)
                # Convert to local frame using ref_base_quat (equivalent to batched_global_to_local)
                inv_quat = quat_inv(self.ref_base_quat)
                self.ref_base_lin_vel_local = quat_apply(inv_quat, base_lin_vel)
                self.base_lin_vel_timestamp = new_timestamp

            # Update base_ang_vel if timestamp changed
            new_timestamp = self._get_timestamp("base_ang_vel")
            if new_timestamp != self.base_ang_vel_timestamp:
                base_ang_vel = torch.tensor(
                    self._get_field("base_ang_vel", [0.0, 0.0, 0.0]),
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, 3)
                # Apply yaw_diff_quat first (equivalent to base_yaw_offset_quat in motion_env)
                base_ang_vel = quat_apply(self._yaw_diff_quat, base_ang_vel)
                # Convert to local frame using ref_base_quat (equivalent to batched_global_to_local)
                inv_quat = quat_inv(self.ref_base_quat)
                self.ref_base_ang_vel_local = quat_apply(inv_quat, base_ang_vel)
                self.base_ang_vel_timestamp = new_timestamp

            # Update base_ang_vel_local if timestamp changed (from motion library, used directly)
            new_timestamp = self._get_timestamp("base_ang_vel_local")
            if new_timestamp != self.base_ang_vel_local_timestamp:
                base_ang_vel_local = torch.tensor(
                    self._get_field("base_ang_vel_local", [0.0, 0.0, 0.0]),
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, 3)
                # Use directly without transformation (as in motion_env.py line 627)
                self.ref_base_ang_vel_local = base_ang_vel_local
                self.base_ang_vel_local_timestamp = new_timestamp

            # Update dof_pos if timestamp changed
            new_timestamp = self._get_timestamp("dof_pos")
            if new_timestamp != self.dof_pos_timestamp:
                dof_pos = self._fit_dim(self._get_field("dof_pos", []), self._dof_dim)
                self.ref_dof_pos = dof_pos
                self.dof_pos_timestamp = new_timestamp

            # Update dof_vel if timestamp changed
            new_timestamp = self._get_timestamp("dof_vel")
            if new_timestamp != self.dof_vel_timestamp:
                dof_vel = self._fit_dim(self._get_field("dof_vel", []), self._dof_dim)
                self.ref_dof_vel = dof_vel
                self.dof_vel_timestamp = new_timestamp

            # Parse link positions and quaternions if available
            if self.num_tracking_links > 0:
                # Update link_pos_local if timestamp changed
                new_timestamp = self._get_timestamp("link_pos_local")
                if new_timestamp != self.link_pos_local_timestamp:
                    link_pos_local = self._fit_link_data(
                        self._get_field("link_pos_local", []), self.num_tracking_links, 3
                    )
                    self.link_pos_local = link_pos_local
                    # link_pos_local is already in local-yaw frame, so set link_pos_local_yaw
                    self.link_pos_local_yaw = link_pos_local
                    self.link_pos_local_timestamp = new_timestamp

                # Update link_quat_local if timestamp changed
                new_timestamp = self._get_timestamp("link_quat_local")
                if new_timestamp != self.link_quat_local_timestamp:
                    link_quat_local = self._fit_link_data(
                        self._get_field("link_quat_local", []), self.num_tracking_links, 4
                    )
                    self.link_quat_local = link_quat_local  # Store raw value
                    # link_quat_local is already in local-yaw frame, so set link_quat_local_yaw
                    self.link_quat_local_yaw = link_quat_local
                    self.link_quat_local_timestamp = new_timestamp

            # Update foot_contact if timestamp changed (used directly without transformation)
            new_timestamp = self._get_timestamp("foot_contact")
            if new_timestamp != self.foot_contact_timestamp:
                foot_contact_data = self._get_field("foot_contact", [])
                if len(foot_contact_data) > 0:
                    # Resize buffers if needed
                    n_feet = len(foot_contact_data)
                    if self.ref_foot_contact.shape[1] != n_feet:
                        self.ref_foot_contact = torch.zeros(1, n_feet, device=self._device)
                        self.ref_foot_contact = torch.zeros(1, n_feet, device=self._device)
                    foot_contact = torch.tensor(
                        foot_contact_data, dtype=torch.float32, device=self._device
                    ).view(1, n_feet)
                    self.ref_foot_contact = foot_contact  # Store raw value
                    # Use directly without transformation (as in motion_env.py line 638)
                    self.ref_foot_contact = foot_contact
                    self.foot_contact_timestamp = new_timestamp

        except Exception as e:
            print(f"Error updating Redis client: {e}")
            self._zero_all()

    @staticmethod
    def _batched_global_to_local(base_quat: torch.Tensor, global_vec: torch.Tensor) -> torch.Tensor:
        """Convert global vectors to local frame using quaternion.

        Args:
            base_quat: Quaternion tensor of shape (B, 4)
            global_vec: Global vector tensor of shape (B, L, 3) or (B, L, 4)

        Returns:
            Local vector tensor of same shape as global_vec
        """
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

    def update_quat(self, quat: torch.Tensor) -> None:
        """Compute and store yaw difference from input quaternion.
        The stored yaw difference will be applied to all subsequent updates.

        Args:
            quat: Input quaternion tensor of shape (1, 4) in (w, x, y, z) format.
        """
        # Extract yaw from input quaternion
        input_euler = quat_to_euler(quat)
        input_yaw = input_euler[:, 2]  # yaw is the third element (z-axis rotation)

        # Extract yaw from current ref_base_quat
        current_euler = quat_to_euler(self.ref_base_quat)
        current_yaw = current_euler[:, 2]

        # Compute yaw difference
        yaw_diff = input_yaw - current_yaw

        # Create a quaternion representing only the yaw rotation (around z-axis)
        # Euler angles: (roll, pitch, yaw) - we only want yaw rotation
        yaw_only_euler = torch.zeros(1, 3, device=self._device)
        yaw_only_euler[:, 2] = yaw_diff  # only set yaw component
        self._yaw_diff_quat = quat_from_euler(yaw_only_euler)

        # Apply yaw difference to current ref_base_quat
        self.ref_base_quat = quat_mul(self._yaw_diff_quat, self.ref_base_quat)

        # Update derived quantities
        self.ref_base_euler = quat_to_euler(self.ref_base_quat)
        self.ref_base_rotation_6D = quat_to_rotation_6D(self.ref_base_quat)
        return

    def set_motion_obs_elements(self, elements: list[str] | None) -> None:
        """Select which elements to include in compute_motion_obs.

        Args:
            elements: List of term names among:
                ['base_pos','base_quat','base_lin_vel','base_ang_vel',
                 'base_ang_vel_local','dof_pos','dof_vel',
                 'link_pos_local','link_quat_local','foot_contact'].
                 If None or empty, no elements are computed (empty observation).
        """
        valid = {
            "base_pos",
            "base_quat",
            "base_lin_vel",
            "base_ang_vel",
            "base_ang_vel_local",
            "dof_pos",
            "dof_vel",
            "link_pos_local",
            "link_quat_local",
            "link_lin_vel",
            "link_ang_vel",
            "foot_contact",
        }
        if elements is None or len(elements) == 0:
            self._motion_obs_elements = None
            return
        filtered = [e for e in elements if e in valid]
        self._motion_obs_elements = set(filtered)

    def get_future_dict(self, motion_obs_elements: list[str]) -> dict[str, torch.Tensor]:
        """Build and filter future_dict from current Redis data based on motion_obs_elements.

        Args:
            motion_obs_elements: List of motion observation element names to include.

        Returns:
            Dictionary with keys from motion_obs_elements and corresponding tensor values.
            Each tensor is shaped as (1, 1, ...) to represent a single future step.
        """
        future_dict: dict[str, torch.Tensor] = {}

        # Mapping from motion obs element names to RedisClient attributes
        # Note: We use ref_* attributes where available, as they're already transformed
        for key in motion_obs_elements:
            if key == "base_pos":
                future_dict[key] = self.ref_base_pos.unsqueeze(1)
            elif key == "base_quat":
                future_dict[key] = self.ref_base_quat.unsqueeze(1)
            elif key == "base_lin_vel":
                future_dict[key] = self.ref_base_lin_vel_local.unsqueeze(1)
            elif key == "base_ang_vel":
                base_ang_vel_global = quat_apply(self.ref_base_quat, self.ref_base_ang_vel_local)
                future_dict[key] = base_ang_vel_global.unsqueeze(1)
            elif key == "base_ang_vel_local":
                future_dict[key] = self.ref_base_ang_vel_local.unsqueeze(1)
            elif key == "dof_pos":
                future_dict[key] = self.ref_dof_pos.unsqueeze(1)
            elif key == "dof_vel":
                future_dict[key] = self.ref_dof_vel.unsqueeze(1)
            elif key == "link_pos_local":
                future_dict[key] = self.link_pos_local_yaw.unsqueeze(1)
            elif key == "link_quat_local":
                future_dict[key] = self.link_quat_local_yaw.unsqueeze(1)
            elif key == "link_lin_vel":
                future_dict[key] = self.link_lin_vel.unsqueeze(1)
            elif key == "link_ang_vel":
                future_dict[key] = self.link_ang_vel.unsqueeze(1)
            elif key == "foot_contact":
                future_dict[key] = self.ref_foot_contact.unsqueeze(1)

        return future_dict


class G1Retargeter:
    """Stateful retargeter: calibrates once, then retargets frames into robot-space motion."""

    LINK_ORDER = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "torso_link",
        "pelvis",
    )

    def __init__(self, joint_space_retarget: bool = False) -> None:
        self.frame_rate = 120.0

        # Indices into the 6-link tensors (fixed order above)
        self.l_foot_idx = 0
        self.r_foot_idx = 1
        self.l_hand_idx = 2
        self.r_hand_idx = 3
        self.torso_idx = 4
        self.base_idx = 5

        self.motion_quat_inv = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(6, 1)
        self.global_yaw_inv = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.global_xy = torch.tensor([0.0, 0.0])
        # Manual
        self.g1_shoulder_y = 0.100
        self.g1_arm_length = 0.419 * 0.9
        self.g1_pelvis_shoulder_z = 1.082 - 0.793
        self.g1_pelvis_torso_z = 0.837 - 0.793
        self.g1_pelvis_z = 0.793 * 0.98
        self.foot_offset_x = 0.06
        self.g1_shoulder_anchor = torch.tensor(
            [
                [0.0, self.g1_shoulder_y, self.g1_pelvis_shoulder_z],
                [0.0, -self.g1_shoulder_y, self.g1_pelvis_shoulder_z],
            ],
            dtype=torch.float32,
        )
        # Calibrated
        self.aug_arm_length = torch.tensor([self.g1_arm_length, self.g1_arm_length]) * 1.0
        self.aug_pelvis_z = self.g1_pelvis_z * 1.0
        self.aug_shoulder_anchor = self.g1_shoulder_anchor.clone()

        self.torso_quat_scale = 1.0

        self._calibrated = False

        self.vel_ema_alpha = 0.25

        self.prev_frame_id: int = -1
        self.prev_tracked_pos: torch.Tensor
        self.prev_tracked_quat: torch.Tensor

        self.ema_base_lin_vel = torch.zeros(3)
        self.ema_base_ang_vel = torch.zeros(3)
        self.ema_base_ang_vel_local = torch.zeros(3)
        self.ema_link_lin_vel = torch.zeros(6, 3)
        self.ema_link_ang_vel = torch.zeros(6, 3)

        self.joint_space_retarget = joint_space_retarget
        if self.joint_space_retarget:
            from GMR.general_motion_retargeting.motion_retarget import GeneralMotionRetargeting

            self.joint_space_retargeter = GeneralMotionRetargeting(
                src_human="optitrack",
                tgt_robot="unitree_g1",
                aligned_fps=60.0,
            )

    def _ema(self, prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        a = self.vel_ema_alpha
        return (1.0 - a) * prev + a * x

    def _reorient_quat(self, quat_local: torch.Tensor, idxs: list[int]) -> torch.Tensor:
        quat_local[idxs] = quat_mul(quat_local[idxs], self.motion_quat_inv[idxs])
        return quat_local

    def _apply_yaw_inv(
        self, pos: torch.Tensor, quat: torch.Tensor, yaw_inv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        yaw_inv = yaw_inv.view(1, 4).repeat(pos.shape[0], 1)
        link_pos_global = quat_apply(yaw_inv, pos)
        link_quat_global = quat_mul(yaw_inv, quat)
        return link_pos_global, link_quat_global

    def _localize(
        self, pos: torch.Tensor, quat: torch.Tensor, keep_height: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        link_pos_global = pos
        link_quat_global = quat
        base_pos = link_pos_global[self.base_idx, :]
        base_quat = link_quat_global[self.base_idx, :]
        relative_link_pos_global = link_pos_global.clone()
        if keep_height:
            relative_link_pos_global[:, :2] -= base_pos[:2]
        else:
            relative_link_pos_global -= base_pos
        base_euler = quat_to_euler(base_quat)
        base_euler[0] = 0.0
        base_euler[1] = 0.0
        inv_yaw = quat_from_euler(-base_euler)
        link_pos_local, link_quat_local = self._apply_yaw_inv(
            relative_link_pos_global, link_quat_global, inv_yaw
        )
        return link_pos_local, link_quat_local

    def calibrate(
        self,
        tracked_pos: torch.Tensor,
        tracked_quat: torch.Tensor,
    ) -> None:
        ### Global
        base_quat = tracked_quat[self.base_idx]
        base_euler = quat_to_euler(base_quat)
        base_euler[0] = 0.0
        base_euler[1] = 0.0
        yaw = quat_from_euler(base_euler)
        self.global_yaw_inv = quat_inv(yaw)
        self.global_xy = tracked_pos[self.base_idx, :2].clone()
        ### Local
        tracked_pos, tracked_quat = self._localize(tracked_pos, tracked_quat)
        self.motion_quat_inv = quat_inv(tracked_quat)
        ### Scale
        left_hand_pos = tracked_pos[self.l_hand_idx]
        right_hand_pos = tracked_pos[self.r_hand_idx]
        self.aug_arm_length = torch.stack([left_hand_pos[0], right_hand_pos[0]])
        aug_l_shoulder_y = left_hand_pos[1].item()
        aug_r_shoulder_y = right_hand_pos[1].item()
        aug_l_shoulder_z = left_hand_pos[2].item()
        aug_r_shoulder_z = right_hand_pos[2].item()
        self.aug_pelvis_z = tracked_pos[self.base_idx, 2].item()
        self.aug_shoulder_anchor = torch.tensor(
            [
                [0.0, aug_l_shoulder_y, aug_l_shoulder_z - self.aug_pelvis_z],
                [0.0, aug_r_shoulder_y, aug_r_shoulder_z - self.aug_pelvis_z],
            ],
            dtype=torch.float32,
        )
        self._calibrated = True
        print("Calibration result:")
        print(f"  - Pelvis Z: {self.aug_pelvis_z:.3f}")
        print(f"  - Arm Length (Mean): {self.aug_arm_length.mean().item():.3f}")
        print(f"  - Shoulder Y (Mean): {self.aug_shoulder_anchor[:, 1].abs().mean().item():.3f}")
        print(f"  - Pelvis Shoulder Z (Mean): {(self.aug_shoulder_anchor[:, 2].mean().item()):.3f}")

    def step(
        self, tracked_pos: torch.Tensor, tracked_quat: torch.Tensor, frame_id: int
    ) -> dict[str, torch.Tensor]:
        if not self._calibrated:
            self.calibrate(tracked_pos, tracked_quat)

        # Local re-orientation
        tracked_quat = self._reorient_quat(tracked_quat, list(range(6)))
        # Global
        tracked_pos[:, :2] = tracked_pos[:, :2] - self.global_xy
        tracked_pos, tracked_quat = self._apply_yaw_inv(
            tracked_pos, tracked_quat, self.global_yaw_inv
        )
        # EE scaling (arm use base frame + torso rotation)
        _ee_idxs = [self.l_hand_idx, self.r_hand_idx, self.l_foot_idx, self.r_foot_idx]
        _ee_base_pos_idxs = [self.base_idx] * 4
        _ee_base_quat_idxs = [self.torso_idx] * 2 + [self.base_idx] * 2
        p, q = pose_diff_quat(
            tracked_pos[_ee_base_pos_idxs + [self.base_idx]],
            tracked_quat[_ee_base_quat_idxs + [self.base_idx]],
            tracked_pos[_ee_idxs + [self.torso_idx]],
            tracked_quat[_ee_idxs + [self.torso_idx]],
        )
        hand_pos_local, hand_quat_local = p[0:2], q[0:2]
        hand_pos_local = self.g1_shoulder_anchor + (hand_pos_local - self.aug_shoulder_anchor) * (
            self.g1_arm_length / self.aug_arm_length.view(2, 1)
        )
        foot_pos_local, foot_quat_local = p[2:4], q[2:4]
        foot_pos_local = foot_pos_local * (self.g1_pelvis_z / self.aug_pelvis_z)
        # Base scaling
        tracked_pos[self.base_idx] = tracked_pos[self.base_idx] * (
            self.g1_pelvis_z / self.aug_pelvis_z
        )
        # Torso replacing
        torso_pos_local = torch.tensor(
            [[0.0, 0.0, self.g1_pelvis_torso_z]],
            dtype=torch.float32,
        )
        torso_quat_local = q[4:5]
        torso_quat_local = quat_from_angle_axis(
            quat_to_angle_axis(torso_quat_local) * self.torso_quat_scale
        )
        # Update back
        p, q = pose_mul_quat(
            tracked_pos[_ee_base_pos_idxs + [self.base_idx]],
            tracked_quat[_ee_base_quat_idxs + [self.base_idx]],
            torch.cat(
                [
                    hand_pos_local,
                    foot_pos_local,
                    torso_pos_local,
                ],
                dim=0,
            ),
            torch.cat(
                [
                    hand_quat_local,
                    foot_quat_local,
                    torso_quat_local,
                ],
                dim=0,
            ),
        )
        tracked_pos[_ee_idxs + [self.torso_idx]] = p[:]
        tracked_quat[_ee_idxs + [self.torso_idx]] = q[:]
        # Foot move forward
        p = tracked_pos[[self.l_foot_idx, self.r_foot_idx]] + quat_apply(
            tracked_quat[[self.l_foot_idx, self.r_foot_idx]],
            torch.tensor([self.foot_offset_x, 0.0, 0.0]).view(1, 3).repeat(2, 1),
        )
        tracked_pos[self.l_foot_idx] = p[0]
        tracked_pos[self.r_foot_idx] = p[1]
        base_pos = tracked_pos[self.base_idx]
        base_quat = tracked_quat[self.base_idx]

        dof_pos = torch.zeros(29, device=tracked_pos.device)
        if self.joint_space_retarget:
            human_data = {}
            for i, link_name in enumerate(self.LINK_ORDER):
                link_pos = tracked_pos[i].detach().cpu().numpy()
                link_quat = tracked_quat[i].detach().cpu().numpy()
                human_data[link_name] = (link_pos, link_quat)

            qpos = self.joint_space_retargeter.retarget(human_data)
            qpos_t = torch.from_numpy(qpos).to(tracked_pos.device).float()
            base_pos = qpos_t[0:3]
            base_quat = qpos_t[3:7]
            dof_pos = qpos_t[7:]

        # Localize
        tracked_pos_local, tracked_quat_local = self._localize(
            tracked_pos, tracked_quat, keep_height=False
        )

        if self.prev_frame_id != -1:
            df = frame_id - self.prev_frame_id
            if df > 0:
                dt = df / self.frame_rate
                link_lin_vel_raw = (tracked_pos - self.prev_tracked_pos) / dt
                base_lin_vel_raw = link_lin_vel_raw[self.base_idx]
                q_delta = quat_diff(tracked_quat, self.prev_tracked_quat)
                axis_angle = quat_to_angle_axis(q_delta)
                link_ang_vel_raw = axis_angle / dt
                base_ang_vel_raw = link_ang_vel_raw[self.base_idx]

                base_q = tracked_quat[self.base_idx]
                base_ang_vel_local_raw = quat_apply(quat_inv(base_q), base_ang_vel_raw)

                self.ema_link_lin_vel = self._ema(self.ema_link_lin_vel, link_lin_vel_raw)
                self.ema_base_lin_vel = self._ema(self.ema_base_lin_vel, base_lin_vel_raw)
                self.ema_link_ang_vel = self._ema(self.ema_link_ang_vel, link_ang_vel_raw)
                self.ema_base_ang_vel = self._ema(self.ema_base_ang_vel, base_ang_vel_raw)
                self.ema_base_ang_vel_local = self._ema(
                    self.ema_base_ang_vel_local, base_ang_vel_local_raw
                )

        self.prev_tracked_pos = tracked_pos.detach().clone()
        self.prev_tracked_quat = tracked_quat.detach().clone()
        self.prev_frame_id = frame_id

        return {
            "base_pos": base_pos,
            "base_quat": base_quat,
            "link_pos_local": tracked_pos_local,
            "link_quat_local": tracked_quat_local,
            "base_lin_vel": self.ema_base_lin_vel,
            "base_ang_vel": self.ema_base_ang_vel,
            "base_ang_vel_local": self.ema_base_ang_vel_local,
            "link_lin_vel": self.ema_link_lin_vel,
            "link_ang_vel": self.ema_link_ang_vel,
            "dof_pos": dof_pos,
        }

    @property
    def calibrated(self) -> bool:
        return self._calibrated
