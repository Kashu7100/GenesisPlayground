#!/usr/bin/env python3
"""Evaluate BC or PPO policies by traversing all motions in the motion library."""

import glob
import os
from pathlib import Path
from typing import Any, cast

import fire
import gs_env.sim.envs as gs_envs
import torch
from gs_agent.algos.bc import BC
from gs_agent.algos.config.schema import BCArgs, PPOArgs
from gs_agent.algos.ppo import PPO
from gs_agent.utils.policy_loader import load_latest_model
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.common.utils.math_utils import quat_apply, quat_from_angle_axis, quat_mul
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.scenes.config.registry import SceneArgsRegistry
from utils import apply_overrides_generic, yaml_to_config


def create_gs_env(
    show_viewer: bool = False,
    num_envs: int = 1,
    device: str = "cuda",
    args: Any = None,
    eval_mode: bool = False,
) -> gs_envs.MotionEnv:
    """Create Genesis Motion environment with optional config overrides."""
    if torch.cuda.is_available() and device == "cuda":
        device_tensor = torch.device("cuda")
    else:
        device_tensor = torch.device("cpu")
    print(f"Using device: {device_tensor}")

    env_class = getattr(gs_envs, args.env_name)

    return env_class(
        args=args,
        num_envs=num_envs,
        show_viewer=show_viewer,
        device=device_tensor,  # type: ignore
        eval_mode=eval_mode,
    )


def evaluate_policy(
    exp_name: str,
    policy_type: str = "auto",  # "auto", "bc", or "ppo"
    num_ckpt: int | None = None,
    device: str = "cuda",
    env_overrides: dict[str, Any] | None = None,
    show_viewer: bool = False,
    motion_file: str | None = None,
) -> None:
    """Evaluate a trained BC or PPO policy by traversing all motions.

    Args:
        exp_name: Name of the experiment directory
        policy_type: Type of policy ("auto" to detect from config, "bc", or "ppo")
        num_ckpt: Checkpoint number to load. If None, loads latest.
        device: Device to use ("cuda" or "cpu")
        env_overrides: Optional environment config overrides
        show_viewer: Whether to show viewer (default: False for batch evaluation)
        motion_file: Optional motion file path to override the one from experiment config
    """
    if env_overrides is None:
        env_overrides = {}

    print("=" * 80)
    print("EVALUATION MODE: Disabling observation noise and domain randomization")
    print("=" * 80)

    # Locate experiment directory
    log_pattern = f"logs/{exp_name}/*"
    log_dirs = glob.glob(log_pattern)
    if not log_dirs:
        raise FileNotFoundError(f"No experiment directories found matching pattern: {log_pattern}")
    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    exp_dir = log_dirs[0]
    print(f"Loading policy from experiment: {exp_dir}")

    # Resolve checkpoint
    if num_ckpt is not None:
        ckpt_path = Path(exp_dir) / "checkpoints" / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    else:
        ckpt_path = load_latest_model(Path(exp_dir))
        num_ckpt = int(ckpt_path.stem.split("_")[-1])
    print(f"Loading checkpoint: {ckpt_path}")

    # Load configs
    print(f"Loading configs from experiment: {exp_dir}")
    env_args = yaml_to_config(Path(exp_dir) / "configs" / "env_args.yaml", MotionEnvArgs)
    env_args = apply_overrides_generic(env_args, env_overrides, prefixes=("cfgs.", "env."))

    # Override motion_file if provided
    if motion_file is not None:
        env_args = env_args.model_copy(update={"motion_file": motion_file})
        print(f"Using motion file: {motion_file}")

    # Auto-detect policy type if needed
    if policy_type == "auto":
        algo_cfg_path = Path(exp_dir) / "configs" / "algo_cfg.yaml"
        if algo_cfg_path.exists():
            algo_cfg_dict = yaml_to_config(algo_cfg_path, None)
            # Check if it's BC by looking for teacher_path
            if "teacher_path" in algo_cfg_dict:
                policy_type = "bc"
            else:
                policy_type = "ppo"
        else:
            # Default to PPO if we can't determine
            policy_type = "ppo"
        print(f"Auto-detected policy type: {policy_type}")

    # Load algorithm config
    if policy_type == "bc":
        algo_cfg = yaml_to_config(Path(exp_dir) / "configs" / "algo_cfg.yaml", BCArgs)
    else:
        algo_cfg = yaml_to_config(Path(exp_dir) / "configs" / "algo_cfg.yaml", PPOArgs)

    # Disable observation noise and domain randomization for evaluation
    env_args = env_args.model_copy(update={"obs_noises": {}})
    env_args = cast(MotionEnvArgs, env_args).model_copy(
        update={"scene_args": SceneArgsRegistry["custom_scene_g1_mocap"]}
    )

    from gs_env.sim.robots.config.registry import DRArgsRegistry

    robot_args = env_args.robot_args.model_copy(
        update={"dr_args": DRArgsRegistry["no_randomization"]}
    )
    env_args = env_args.model_copy(update={"robot_args": robot_args})

    # Build eval environment
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=1,
        device=device,
        args=env_args,
        eval_mode=True,
    )
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Create algorithm and load weights
    if policy_type == "bc":
        algorithm = BC(env=wrapped_env, cfg=algo_cfg, device=wrapped_env.device)
    else:
        algorithm = PPO(env=wrapped_env, cfg=algo_cfg, device=wrapped_env.device)

    algorithm.load(ckpt_path, load_optimizer=False)
    inference_policy = algorithm.get_inference_policy()

    # Get an example observation and trace the policy
    obs, _ = wrapped_env.get_observations()

    # For PPO, wrap to ensure deterministic=True
    if policy_type == "ppo":

        class DeterministicWrapper(torch.nn.Module):
            def __init__(self, policy: Any) -> None:
                super().__init__()
                self.policy = policy

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                action, _ = self.policy(obs, deterministic=True)
                return action

        wrapped_policy = DeterministicWrapper(inference_policy)
        traced_policy = torch.jit.trace(wrapped_policy, obs)
    else:
        traced_policy = torch.jit.trace(inference_policy, obs)

    print("Starting evaluation across all motions...")
    print(f"Total motions in library: {env.motion_lib.num_motions}")

    # Prepare link tracking indices for visualization
    link_name_to_idx: dict[str, int] = {}
    for link_name in env.scene.objects.keys():
        link_name_to_idx[link_name] = env_args.tracking_link_names.index(link_name)

    # Statistics collection
    motion_results: list[dict[str, Any]] = []
    total_steps = 0
    total_terminations = 0

    # Iterate through all motions
    for motion_id in range(env.motion_lib.num_motions):
        motion_name = env.motion_lib.motion_names[motion_id]
        motion_length = env.motion_lib.get_motion_length(torch.IntTensor([motion_id])).item()

        print(f"\nEvaluating motion {motion_id}/{env.motion_lib.num_motions - 1}: {motion_name}")
        print(f"  Motion length: {motion_length:.2f}s")

        # Reset environment for this motion
        env.time_since_reset[0] = 0.0
        env.hard_reset_motion(torch.IntTensor([0]), motion_id)
        env.hard_sync_motion(torch.IntTensor([0]))
        obs, _ = wrapped_env.get_observations()

        # Track metrics for this motion
        motion_steps = 0
        motion_terminated = False

        # Run until motion completes or terminates
        while (
            env.motion_times[0]
            < env.motion_lib.get_motion_length(torch.IntTensor([motion_id])) - 0.02
        ):
            with torch.no_grad():
                action = traced_policy(obs)  # type: ignore[misc]

            env.apply_action(action)
            terminated = env.get_terminated()
            if terminated[0]:
                motion_terminated = True
                env.hard_sync_motion(torch.IntTensor([0]))

            env.update_buffers()
            env.update_history()
            obs, _ = wrapped_env.get_observations()

            motion_steps += 1

            # Exit loop on termination
            if terminated[0]:
                env.hard_sync_motion(torch.IntTensor([0]))

            # Update reference visualization
            ref_quat_yaw = quat_from_angle_axis(
                env.ref_base_euler[:, 2],
                torch.tensor([0, 0, 1], device=env.device, dtype=torch.float),
            )
            for link_name in env.scene.objects.keys():
                ref_link_pos = env.ref_tracking_link_pos_local_yaw[:, link_name_to_idx[link_name]]
                ref_link_quat = env.ref_tracking_link_quat_local_yaw[:, link_name_to_idx[link_name]]
                ref_link_pos = quat_apply(ref_quat_yaw, ref_link_pos)
                ref_link_pos[:, :2] += env.ref_base_pos[:, :2]
                ref_link_quat = quat_mul(ref_quat_yaw, ref_link_quat)
                env.scene.set_obj_pose(link_name, pos=ref_link_pos, quat=ref_link_quat)

        # Store results for this motion
        motion_result = {
            "motion_id": motion_id,
            "motion_name": motion_name,
            "motion_length": motion_length,
            "steps": motion_steps,
            "terminated": motion_terminated,
        }
        motion_results.append(motion_result)

        # Update global statistics
        total_steps += motion_steps
        if motion_terminated:
            total_terminations += 1

        print(f"  Steps: {motion_steps}, Terminated: {motion_terminated}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total motions evaluated: {len(motion_results)}")
    print(f"Total steps: {total_steps}")
    print(
        f"Terminations: {total_terminations} ({100.0 * total_terminations / len(motion_results):.1f}%)"
    )
    print(
        f"Success rate: {100.0 * (len(motion_results) - total_terminations) / len(motion_results):.1f}%"
    )

    # Per-motion statistics
    print("\nPer-motion statistics:")
    print(f"{'Motion ID':<10} {'Name':<30} {'Steps':<8} {'Status':<10}")
    print("-" * 60)
    for result in motion_results:
        status = "TERMINATED" if result["terminated"] else "COMPLETED"
        print(
            f"{result['motion_id']:<10} "
            f"{result['motion_name'][:28]:<30} "
            f"{result['steps']:<8} "
            f"{status:<10}"
        )

    print("=" * 80)


def main(
    exp_name: str,
    policy_type: str = "auto",
    num_ckpt: int | None = None,
    device: str = "cpu",
    show_viewer: bool = False,
    motion_file: str | None = None,
    **cfg_overrides: Any,
) -> None:
    """Entry point for motion evaluation.

    Args:
        exp_name: Name of the experiment directory
        policy_type: Type of policy ("auto", "bc", or "ppo")
        num_ckpt: Checkpoint number to load. If None, loads latest.
        device: Device to use ("cuda" or "cpu")
        show_viewer: Whether to show viewer
        motion_file: Optional motion file path to override the one from experiment config
        **cfg_overrides: Optional config overrides (e.g., --env.reward_args.AngVelZReward=5)
    """
    # Bucket overrides into env
    env_overrides: dict[str, Any] = {}

    for k, v in cfg_overrides.items():
        if k.startswith("cfgs.env.") or k.startswith("env.") or k.startswith("reward_args."):
            env_overrides[k] = v

    evaluate_policy(
        exp_name=exp_name,
        policy_type=policy_type,
        num_ckpt=num_ckpt,
        device=device,
        env_overrides=env_overrides,
        show_viewer=show_viewer,
        motion_file=motion_file,
    )


if __name__ == "__main__":
    fire.Fire(main)
