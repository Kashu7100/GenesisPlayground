import sys
import time
from pathlib import Path

import fire
import numpy as np
import torch
from gs_env.sim.envs.config.schema import WalkingEnvArgs

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from utils import yaml_to_config  # type: ignore


def load_checkpoint_and_env_args(
    exp_name: str, num_ckpt: int | None = None, device: str = "cuda"
) -> tuple[torch.jit.ScriptModule, WalkingEnvArgs]:
    """Load JIT checkpoint and env_args from deploy/logs directory.

    Args:
        exp_name: Experiment name
        num_ckpt: Checkpoint number. If None, loads the latest checkpoint.

    Returns:
        Tuple of (checkpoint_path, env_args)
    """

    deploy_dir = Path(__file__).parent / "logs" / exp_name
    if not deploy_dir.exists():
        raise FileNotFoundError(f"Deploy directory not found: {deploy_dir}")

    # Load env_args from YAML
    env_args_path = deploy_dir / "env_args.yaml"
    if not env_args_path.exists():
        raise FileNotFoundError(f"env_args.yaml not found: {env_args_path}")

    print(f"Loading env_args from: {env_args_path}")
    env_args = yaml_to_config(env_args_path, WalkingEnvArgs)

    # Load checkpoint
    if num_ckpt is not None:
        ckpt_path = deploy_dir / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # Find latest checkpoint
        ckpts = list(deploy_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {deploy_dir}")
        ckpt_path = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))

    print(f"Loading checkpoint from: {ckpt_path}")
    # Load policy
    policy = torch.jit.load(str(ckpt_path))
    policy.to(device)
    policy.eval()

    return policy, env_args


def to_numpy(data: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return np.array(data)


def main(
    exp_name: str = "walk",
    num_ckpt: int | None = None,
    device: str = "cpu",
    show_viewer: bool = True,
    sim: bool = True,
    action_scale: float = 0.0,  # only for real robot
) -> None:
    """Run policy on either simulation or real robot.

    Args:
        exp_name: Experiment name (subdirectory in deploy/logs)
        num_ckpt: Checkpoint number. If None, loads latest.
        device: Device for policy inference ('cuda' or 'cpu')
        show_viewer: Show viewer (only for sim mode)
        sim: If True, run in simulation. If False, run on real robot.
        num_envs: Number of environments (only for sim mode)
    """
    device = "cpu" if not torch.cuda.is_available() else device

    # Load checkpoint and env_args
    policy, env_args = load_checkpoint_and_env_args(exp_name, num_ckpt, device)

    if sim:
        print("Running in SIMULATION mode")
        import gs_env.sim.envs as envs

        envclass = getattr(envs, env_args.env_name)
        env = envclass(
            args=env_args,
            num_envs=1,
            show_viewer=show_viewer,
            device=torch.device(device),
            eval_mode=True,
        )
        env.eval()
        env.reset()

    else:
        print("Running in REAL ROBOT mode")
        from gs_env.real import UnitreeLeggedEnv

        env = UnitreeLeggedEnv(
            env_args, action_scale=action_scale, interactive=True, device=torch.device(device)
        )

        print("Press Start button to start the policy")
        while not env.robot.Start:
            time.sleep(0.1)

    print("=" * 80)
    print("Starting policy execution")
    print(f"Mode: {'SIMULATION' if sim else 'REAL ROBOT'}")
    print(f"Device: {device}")
    print("=" * 80)

    def deploy_loop() -> None:
        nonlocal env

        # Initialize tracking variables
        last_action_t = torch.zeros(1, env.action_dim, device=device)
        commands_t = torch.zeros(1, 3, device=device)
        last_update_time = time.time()
        total_inference_time = 0
        step_id = 0

        while True:
            # Check termination condition (only for real robot)
            if not sim and hasattr(env, "is_emergency_stop") and env.is_emergency_stop:  # type: ignore
                print("Emergency stop triggered!")
                break

            # Control loop timing (50 Hz)
            if time.time() - last_update_time < 0.02:
                time.sleep(0.001)
                continue
            last_update_time = time.time()

            if not sim:
                commands_t[0, 0] = env.robot.Ly  # forward velocity (m/s)
                commands_t[0, 1] = -env.robot.Lx  # lateral velocity (m/s)
                commands_t[0, 2] = -env.robot.Rx  # angular velocity (rad/s)
            else:
                # Update commands (can be modified for different behaviors)
                commands_t[0, 0] = 0.0  # forward velocity (m/s)
                commands_t[0, 1] = 0.0  # lateral velocity (m/s)
                commands_t[0, 2] = 0.0  # angular velocity (rad/s)

            # Construct observation (matching training observation structure)
            obs_components = []
            for key in env_args.actor_obs_terms:
                if key == "last_action":
                    obs_gt = last_action_t
                elif key == "commands":
                    obs_gt = commands_t
                else:
                    obs_gt = getattr(env, key) * env_args.obs_scales.get(key, 1.0)
                obs_components.append(obs_gt)
            obs_t = torch.cat(obs_components, dim=-1)

            # Get action from policy
            with torch.no_grad():
                start_time = time.time()
                action_t = policy(obs_t)
                end_time = time.time()
                total_inference_time += end_time - start_time

            env.apply_action(action_t)

            last_action_t = action_t.clone()
            step_id += 1

            if step_id % 100 == 0:
                print(f"Step {step_id}: Average inference time: {total_inference_time / 100:.4f}s")
                total_inference_time = 0

    try:
        deploy_loop()
    except KeyboardInterrupt:
        if not sim:
            env.emergency_stop()
        print("\nKeyboardInterrupt received, stopping...")
    finally:
        if not sim:
            print("Stopping robot handler...")
            # Handler cleanup if needed
        else:
            print("Simulation stopped.")


if __name__ == "__main__":
    fire.Fire(main)
