import torch

from .leggedrobot_terms import (
    ActionLimitPenalty,  # noqa
    ActionRatePenalty,  # noqa
    AngVelXYPenalty,  # noqa
    AngVelZReward,  # noqa
    BaseHeightPenalty,
    DofPosLimitPenalty,  # noqa
    DofVelPenalty,  # noqa
    FeetAirTimePenalty,  # noqa
    FeetAirTimeReward,  # noqa
    FeetContactForceLimitPenalty,
    FeetHeightPenalty,
    FeetZVelocityPenalty,  # noqa
    LinVelXYReward,  # noqa
    LinVelZPenalty,  # noqa
    OrientationPenalty,  # noqa
    StandStillActionRatePenalty,  # noqa
    StandStillFeetContactPenalty,  # noqa
    TorquePenalty,  # noqa
)
from .reward_terms import RewardTerm


### ---- Reward Terms ---- ###
class G1BaseHeightPenalty(BaseHeightPenalty):
    target_height = 1.0


class UpperBodyDofPenalty(RewardTerm):
    """
    Penalize the upper body DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos", "dof_vel")

    def _compute(self, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, 12:]), dim=-1) - 0.01 * torch.sum(
            torch.square(dof_vel[:, 12:]), dim=-1
        )


class UpperBodyActionPenalty(RewardTerm):
    """
    Penalize the upper body action position.

    Args:
        action: Action tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("action",)

    def _compute(self, action: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(action[:, 12:]), dim=-1)


class HipYawPenalty(RewardTerm):
    """
    Penalize the hip yaw DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos",)

    def _compute(self, dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, [2, 8]]), dim=-1)


class HipRollPenalty(RewardTerm):
    """
    Penalize the hip roll DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos",)

    def _compute(self, dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, [0, 6]]), dim=-1)


class AnkleTorquePenalty(RewardTerm):
    """
    Penalize the ankle torque.

    Args:
        torque: Torque tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("torque",)

    def _compute(self, torque: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(torque[:, [4, 5, 10, 11]]), dim=-1)


class StandStillAnkleTorquePenalty(RewardTerm):
    """
    Penalize the ankle torque.

    Args:
        torque: Torque tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("torque", "commands")

    def _compute(self, torque: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(torque[:, [4, 5, 10, 11]]), dim=-1) * (
            torch.norm(commands, dim=1) < 0.1
        )


class G1FeetHeightPenalty(FeetHeightPenalty):
    target_height = 0.2


class G1FeetContactForcePenalty(RewardTerm):
    """
    Penalize the feet contact force.

    Args:
        foot_contact_force: Feet contact force tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("foot_contact_force", "commands")

    def _compute(self, foot_contact_force: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:  # type: ignore
        contact_force_diff = 200 - foot_contact_force.max(dim=-1).values.clamp(max=200)
        contact_force_diff *= torch.norm(commands, dim=1) > 0.1
        return -torch.square(contact_force_diff / 200)


class FeetOrientationPenalty(RewardTerm):
    """
    Penalize the feet orientation.

    Args:
        feet_orientation: Feet orientation tensor of shape (B, 2, 3) where B is the batch size.
    """

    required_keys = ("feet_orientation",)

    def _compute(self, feet_orientation: torch.Tensor) -> torch.Tensor:  # type: ignore
        feet_orientation_deviation = feet_orientation[:, :, :2].square().sum(dim=-1)
        return -feet_orientation_deviation.sum(dim=-1)


class G1FeetContactForceLimitPenalty(FeetContactForceLimitPenalty):
    contact_force_limit = 300.0


class LinVelYPenalty(RewardTerm):
    """
    Penalize the linear velocity in the Y direction.

    Args:
        base_lin_vel: Linear velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_lin_vel",)

    def _compute(self, base_lin_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.square(base_lin_vel[:, 1])
