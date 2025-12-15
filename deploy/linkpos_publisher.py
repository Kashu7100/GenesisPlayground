import json
import math

# Add examples to path to import utils
import sys
import time
from pathlib import Path

import fire
import redis
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from gs_env.sim.envs.config.schema import MotionEnvArgs
from utils import yaml_to_config  # type: ignore


def _to_list(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().flatten().tolist()


def load_motion_file_from_exp(exp_name: str) -> str:
    deploy_dir = Path(__file__).parent / "logs" / exp_name
    env_args_path = deploy_dir / "env_args.yaml"
    if not env_args_path.exists():
        raise FileNotFoundError(f"env_args.yaml not found: {env_args_path}")
    env_args = yaml_to_config(env_args_path, MotionEnvArgs)
    return env_args.motion_file


def load_env_args_from_exp(exp_name: str) -> MotionEnvArgs:
    deploy_dir = Path(__file__).parent / "logs" / exp_name
    env_args_path = deploy_dir / "env_args.yaml"
    if not env_args_path.exists():
        raise FileNotFoundError(f"env_args.yaml not found: {env_args_path}")
    return yaml_to_config(env_args_path, MotionEnvArgs)


def publish_motion(
    redis_url: str = "redis://localhost:6379/0",
    key: str = "ref:",
    freq_hz: float = 50.0,
    device: str = "cpu",
    tracking_link_names: list[str] | None = None,
) -> None:
    """Publish reference motion frames to Redis at a fixed rate.

    The publisher writes each field as a separate Redis key:
      - {key}:motion:base_pos [3]
      - {key}:motion:base_quat [4] (w, x, y, z)
      - {key}:motion:base_lin_vel [3]
      - {key}:motion:base_ang_vel [3]
      - {key}:motion:base_ang_vel_local [3]
      - {key}:motion:dof_pos [D]
      - {key}:motion:dof_vel [D]
      - {key}:motion:link_pos_local [N*3] (filtered to tracking links if specified)
      - {key}:motion:link_quat_local [N*4] (filtered to tracking links if specified)
      - {key}:motion:foot_contact [F]
      - {key}:timestamp:base_pos [1]
      - {key}:timestamp:base_quat [1]
      - {key}:timestamp:base_lin_vel [1]
      - {key}:timestamp:base_ang_vel [1]
      - {key}:timestamp:base_ang_vel_local [1]
      - {key}:timestamp:dof_pos [1]
      - {key}:timestamp:dof_vel [1]
      - {key}:timestamp:link_pos_local [1]
      - {key}:timestamp:link_quat_local [1]
      - {key}:timestamp:foot_contact [1]
    """
    r = redis.from_url(redis_url)

    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    timestamp = 0

    publish_dt = 1.0 / freq_hz
    next_publish_time = time.time() + publish_dt

    zero_link_pos_local = torch.tensor(
        [
            [0.0, 0.1, 0.04],
            [0.0, -0.1, 0.04],
            [0.2, 0.2, 0.87],
            [0.2, -0.2, 0.87],
            [0.0, 0.0, 0.83],
            [0.0, 0.0, 0.79],
        ],
        dtype=torch.float32,
        device=device_t,
    )
    zero_link_quat_local = torch.zeros(6, 4, device=device_t)
    zero_link_quat_local[:, 0] = 1.0
    zero_link_lin_vel = torch.zeros(6, 3, device=device_t)
    zero_link_ang_vel = torch.zeros(6, 3, device=device_t)

    print("=" * 80)
    print("Linkpos Publisher started")
    print(f"Redis: {redis_url}")
    print(f"Key: {key}")
    print(f"Publish rate: {1.0 / publish_dt:.2f} Hz")
    print("=" * 80)

    try:
        while True:
            link_pos_local = zero_link_pos_local.clone()
            link_pos_local[2, 2] += 0.1 * math.sin(timestamp * 2 * math.pi * publish_dt)
            link_pos_local[3, 2] += 0.1 * math.cos(timestamp * math.pi * publish_dt)
            link_quat_local = zero_link_quat_local.clone()
            link_lin_vel = zero_link_lin_vel.clone()
            link_ang_vel = zero_link_ang_vel.clone()
            # Publish each field as a separate Redis key
            r.set(f"{key}:motion:link_pos_local", json.dumps(_to_list(link_pos_local)))
            r.set(f"{key}:motion:link_quat_local", json.dumps(_to_list(link_quat_local)))
            r.set(f"{key}:motion:link_lin_vel", json.dumps(_to_list(link_lin_vel)))
            r.set(f"{key}:motion:link_ang_vel", json.dumps(_to_list(link_ang_vel)))
            r.set(f"{key}:timestamp:link_pos_local", timestamp)
            r.set(f"{key}:timestamp:link_quat_local", timestamp)
            r.set(f"{key}:timestamp:link_lin_vel", timestamp)
            r.set(f"{key}:timestamp:link_ang_vel", timestamp)
            timestamp += 1

            # Advance time, loop by motion length
            t_now = time.time()
            if t_now < next_publish_time:
                time.sleep(max(0, next_publish_time - t_now))
                next_publish_time = next_publish_time + publish_dt
            else:
                next_publish_time = time.time() + publish_dt

    except KeyboardInterrupt:
        print("\nStopping motion publisher...")


def main(
    redis_url: str = "redis://localhost:6379/0",
    key: str = "motion:ref:latest",
    freq_hz: float = 50.0,
    device: str = "cpu",
) -> None:
    # Resolve motion_file and tracking_link_names
    tracking_link_names: list[str] = []
    publish_motion(
        redis_url=redis_url,
        key=key,
        freq_hz=freq_hz,
        device=device,
        tracking_link_names=tracking_link_names,
    )


if __name__ == "__main__":
    fire.Fire(main)
