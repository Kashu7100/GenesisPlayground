#!/usr/bin/env python3
"""Example: Train DAgger to distill policy from g1_motion_teacher to g1_motion."""

import glob
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import fire
import gs_env.sim.envs as gs_envs
import torch
from gs_agent.algos.config.registry import DAGGER_MOTION_MLP
from gs_agent.algos.dagger import DAgger
from gs_agent.runners.config.registry import RUNNER_DAGGER_MOTION_MLP
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.utils.policy_loader import load_latest_model
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.scenes.config.registry import SceneArgsRegistry
from utils import apply_overrides_generic, config_to_yaml


def create_gs_env(
    show_viewer: bool = False,
    num_envs: int = 4096,
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


def create_dagger_runner_from_registry(
    env: gs_envs.MotionEnv,
    teacher_exp_name: str,
    exp_name: str | None = None,
    teacher_ckpt: int | None = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
) -> tuple[OnPolicyRunner, Path]:
    """Create DAgger runner to distill from teacher to student.

    Args:
        env: Student environment (g1_motion)
        teacher_exp_name: Name of teacher experiment directory (e.g., "g1_motion_teacher")
        exp_name: Name of DAgger experiment directory
        teacher_ckpt: Checkpoint number to load from teacher. If None, loads latest.
        algo_cfg: DAgger algorithm configuration (from registry)
        runner_args: Runner configuration (from registry)
    """
    # Environment setup
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Find teacher experiment directory
    log_pattern = f"logs/{teacher_exp_name}/*"
    log_dirs = glob.glob(log_pattern)
    if not log_dirs:
        raise FileNotFoundError(
            f"No teacher experiment directories found matching pattern: {log_pattern}"
        )

    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    teacher_exp_dir = log_dirs[0]
    print(f"Loading teacher from experiment: {teacher_exp_dir}")

    # Load teacher checkpoint
    if teacher_ckpt is not None:
        teacher_ckpt_path = (
            Path(teacher_exp_dir) / "checkpoints" / f"checkpoint_{teacher_ckpt:04d}.pt"
        )
        if not teacher_ckpt_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint {teacher_ckpt_path} not found")
    else:
        teacher_ckpt_path = load_latest_model(Path(teacher_exp_dir))
        teacher_ckpt = int(teacher_ckpt_path.stem.split("_")[-1])

    print(f"Loading teacher checkpoint: {teacher_ckpt_path}")

    # Load teacher environment config
    teacher_config_path = Path(teacher_exp_dir) / "configs" / "env_args.yaml"
    if not teacher_config_path.exists():
        raise FileNotFoundError(
            f"Teacher config not found at {teacher_config_path}. "
            "Make sure the teacher experiment has configs/env_args.yaml"
        )
    print(f"Loading teacher config from: {teacher_config_path}")

    # Update algo_cfg with teacher path and config
    algo_cfg = algo_cfg.model_copy(
        update={
            "teacher_path": teacher_ckpt_path,
            "teacher_config_path": teacher_config_path,
        }
    )

    # Create DAgger algorithm
    dagger = DAgger(
        env=wrapped_env,
        cfg=algo_cfg,
        device=wrapped_env.device,
    )

    # Create DAgger runner
    if exp_name is not None:
        # Avoid mutating a frozen Pydantic model; create a copied config with updated save_path
        runner_args = runner_args.model_copy(update={"save_path": Path(f"./logs/{exp_name}")})
    runner = OnPolicyRunner(
        algorithm=dagger,
        runner_args=runner_args,
        device=wrapped_env.device,
    )

    return runner, teacher_config_path


def train_policy(
    teacher_exp_name: str = "g1_motion_teacher",
    exp_name: str | None = None,
    show_viewer: bool = False,
    num_envs: int = 4096,
    device: str = "cuda",
    use_wandb: bool = True,
    teacher_ckpt: int | None = None,
    env_args: Any = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
) -> None:
    """Train DAgger policy to distill from teacher to student.

    Args:
        teacher_exp_name: Name of teacher experiment directory
        exp_name: Name of DAgger experiment directory. If None, uses "dagger_motion"
        show_viewer: Whether to show viewer
        num_envs: Number of parallel environments
        device: Device to use ("cuda" or "cpu")
        use_wandb: Whether to use wandb logging
        teacher_ckpt: Checkpoint number to load from teacher. If None, loads latest.
        env_args: Student environment configuration
        algo_cfg: DAgger algorithm configuration
        runner_args: Runner configuration
    """
    env_args = cast(MotionEnvArgs, env_args).model_copy(
        update={"scene_args": SceneArgsRegistry["flat_scene_legged"]}
    )

    # Create student environment
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=num_envs,
        device=device,
        args=env_args,
    )

    # Create DAgger runner
    runner, teacher_config_path = create_dagger_runner_from_registry(
        env=env,
        teacher_exp_name=teacher_exp_name,
        exp_name=exp_name,
        teacher_ckpt=teacher_ckpt,
        algo_cfg=algo_cfg,
        runner_args=runner_args,
    )

    # Set up logging with proper configuration
    if exp_name is not None:
        save_path = Path(f"./logs/{exp_name}")
    else:
        save_path = runner_args.save_path
    logger_folder = save_path / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logger_configure(
        folder=str(logger_folder),
        format_strings=["stdout", "csv", "wandb"],
        entity=None,
        project=None,
        exp_name=exp_name or "dagger_motion",
        mode="online" if use_wandb and (not show_viewer) else "disabled",
    )

    # Save configuration files to YAML
    print("Saving configuration files to YAML...")
    config_dir = logger_folder / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_to_yaml(env_args, config_dir / "env_args.yaml")
    config_to_yaml(algo_cfg, config_dir / "algo_cfg.yaml")
    config_to_yaml(runner_args, config_dir / "runner_args.yaml")

    # Copy teacher config for reference
    import shutil

    shutil.copy(teacher_config_path, config_dir / "teacher_env_args.yaml")
    print(f"Saved teacher config to: {config_dir / 'teacher_env_args.yaml'}")
    print("Note: DAgger uses teacher observations (from teacher config) for teacher policy.")
    print("Student uses student observations. Teacher provides action labels.")
    print("DAgger also trains a value function using rewards from the environment.")

    # Train using Runner
    print("Starting DAgger training...")
    print("Student environment: g1_motion")
    print(f"Teacher environment: g1_motion_teacher (from {teacher_exp_name})")
    print(f"Student observation dim: {env.actor_obs_dim}")
    print(f"Student critic observation dim: {env.critic_obs_dim}")

    def train() -> None:
        nonlocal runner, logger
        train_summary_info = runner.train(metric_logger=logger)
        print("Training completed successfully!")
        print(f"Training completed in {train_summary_info['total_time']:.2f} seconds.")
        print(f"Total iterations: {train_summary_info['total_iterations']}.")
        print(f"Total steps: {train_summary_info['total_steps']}.")
        print(f"Final reward: {train_summary_info['final_reward']:.2f}.")

    try:
        if platform.system() == "Darwin" and show_viewer:
            import threading

            threading.Thread(target=train).start()
            env.scene.scene.viewer.run()  # type: ignore
        else:
            train()
    except KeyboardInterrupt:
        pass


def main(
    teacher_exp_name: str = "g1_motion_teacher",
    exp_name: str | None = None,
    show_viewer: bool = False,
    num_envs: int = 4096,
    device: str = "cuda",
    use_wandb: bool = True,
    teacher_ckpt: int | None = None,
    env_name: str = "g1_motion",
    **cfg_overrides: Any,
) -> None:
    """Entry point for DAgger training and evaluation.

    You can override configs via dot-notation, e.g.:
    --env.reward_args.AngVelZReward=5
    --algo.lr=1e-4
    --runner.total_iterations=2000
    """
    # Bucket overrides into env / algo / runner
    env_overrides: dict[str, Any] = {}
    algo_overrides: dict[str, Any] = {}
    runner_overrides: dict[str, Any] = {}

    for k, v in cfg_overrides.items():
        if k.startswith("cfgs.env.") or k.startswith("env.") or k.startswith("reward_args."):
            env_overrides[k] = v
            continue
        if k.startswith("cfgs.algo.") or k.startswith("algo."):
            algo_overrides[k] = v
            continue
        if k.startswith("cfgs.runner.") or k.startswith("runner."):
            runner_overrides[k] = v
            continue

    env_args = EnvArgsRegistry[env_name]
    env_args = apply_overrides_generic(env_args, env_overrides, prefixes=("cfgs.", "env."))
    algo_cfg = apply_overrides_generic(
        DAGGER_MOTION_MLP, algo_overrides, prefixes=("cfgs.", "algo.")
    )
    runner_args = apply_overrides_generic(
        RUNNER_DAGGER_MOTION_MLP, runner_overrides, prefixes=("cfgs.", "runner.")
    )

    train_policy(
        teacher_exp_name=teacher_exp_name,
        exp_name=exp_name,
        show_viewer=show_viewer,
        num_envs=num_envs,
        device=device,
        use_wandb=use_wandb,
        teacher_ckpt=teacher_ckpt,
        env_args=env_args,
        algo_cfg=algo_cfg,
        runner_args=runner_args,
    )


if __name__ == "__main__":
    fire.Fire(main)
