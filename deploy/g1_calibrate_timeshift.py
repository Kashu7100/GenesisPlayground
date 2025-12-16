import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gs_env.real.config.registry import EnvArgsRegistry as real_env_registry
from gs_env.real.config.schema import OptitrackEnvArgs
from gs_env.real.leggedrobot_env import UnitreeLeggedEnv
from gs_env.real.optitrack_env import OptitrackEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry as sim_env_registry
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.robots.config.schema import HumanoidRobotArgs

FREQ = 50.0  # Hz
DT = 1.0 / FREQ
NUM_TOTAL_PERIODS = 5


def main(args: argparse.Namespace) -> None:
    # Create OptiTrack env with zero offsets
    load_config_path = (
        Path(__file__).resolve().parent.parent / "config" / "optitrack_offset" / "default.yaml"
    )
    optitrack_env_args = real_env_registry["g1_links_tracking"].model_copy(
        update={"offset_config": load_config_path},
    )
    assert isinstance(optitrack_env_args, OptitrackEnvArgs)
    optitrack_env = OptitrackEnv(num_envs=1, args=optitrack_env_args)

    # Create Low State Env
    real_env_args = sim_env_registry["g1_walk"]
    assert isinstance(real_env_args, LeggedRobotEnvArgs)
    real_env = UnitreeLeggedEnv(args=real_env_args, interactive=False)

    test_dofs = {
        "hip_roll": [0.0, 0.8],
        "hip_pitch": [-0.5, 0.5],
        "hip_yaw": [-0.5, 0.5],
        "knee": [0.0, 1.0],
        "ankle_roll": [-0.2, 0.2],
        "ankle_pitch": [-0.5, 0.5],
        "waist_yaw": [-1.0, 1.0],
        "waist_roll": [-0.3, 0.3],
        "waist_pitch": [-0.3, 0.3],
        "shoulder_roll": [0.0, 1.0],
        "shoulder_pitch": [-0.5, 0.5],
        "shoulder_yaw": [0.0, 1.0],
        "elbow": [0.0, 1.0],
        "wrist_roll": [-1.0, 1.0],
        "wrist_pitch": [-1.0, 1.0],
        "wrist_yaw": [0.0, 1.0],
    }

    dof_name = args.dof
    assert isinstance(real_env_args.robot_args, HumanoidRobotArgs)
    gs_joint_order = real_env_args.robot_args.dof_names
    num_dofs = real_env.action_dim
    dof_idx = gs_joint_order.index(dof_name)
    lower_bound, upper_bound = [0.0, 0.0]
    for name in test_dofs:
        if name in dof_name:
            lower_bound, upper_bound = test_dofs[name]
            break
    amplitude = (upper_bound - lower_bound) / 2
    offset = (upper_bound + lower_bound) / 2

    device = torch.device("cpu")
    action = torch.zeros((1, num_dofs), device=device)
    action[:, dof_idx] = offset / real_env.action_scale

    last_update_time = time.time()

    # Reset to initial position
    current_dof_pos = real_env.dof_pos[0] - real_env.default_dof_pos
    TOTAL_RESET_STEPS = 50
    for i in range(TOTAL_RESET_STEPS):
        while time.time() - last_update_time < DT:
            time.sleep(0.001)
        last_update_time = time.time()
        real_env.apply_action(
            current_dof_pos / real_env.action_scale * (1 - i / TOTAL_RESET_STEPS)
            + action * (i / TOTAL_RESET_STEPS)
        )

    link_pos_list = []
    dof_pos_list = []

    period = 100

    def linear_phase(x: int) -> float:
        return x / period * 2 * np.pi

    def wave_func(x: int) -> float:
        return np.sin(linear_phase(x)) * amplitude

    for i in range(period * NUM_TOTAL_PERIODS):
        while time.time() - last_update_time < DT:
            time.sleep(0.001)
        last_update_time = time.time()

        target_dof_pos = wave_func(i) + offset
        action[:, dof_idx] = target_dof_pos / real_env.action_scale
        dof_pos = (
            real_env.dof_pos[0, dof_idx].cpu().item() - real_env.robot.default_dof_pos[dof_idx]
        )
        link_pos = optitrack_env.get_tracked_links()[args.link][0]
        dof_pos_list.append(dof_pos)
        link_pos_list.append(link_pos[0])  # x position
        real_env.apply_action(action)

        if i % period == 0:
            print(f"Step {i} of {period * NUM_TOTAL_PERIODS}")

    # Visualize results
    dof_pos_arr = np.array(dof_pos_list)
    link_pos_arr = np.array(link_pos_list)

    t = np.arange(len(dof_pos_arr)) * DT
    N = min(len(dof_pos_arr), len(link_pos_arr))
    t = t[:N]
    dof_pos_arr = dof_pos_arr[:N]
    link_pos_arr = link_pos_arr[:N]

    a = dof_pos_arr - dof_pos_arr.mean()
    b = link_pos_arr - link_pos_arr.mean()
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    best_lag_pos = lags[np.argmax(corr)]
    best_lag_neg = lags[np.argmin(corr)]
    best_lag = best_lag_pos if abs(best_lag_pos) < abs(best_lag_neg) else best_lag_neg
    time_shift = best_lag * DT
    print(f"[EST] Mocap time offset (mocap delay positive) = {time_shift:+.4f} s")

    plt.figure(figsize=(10, 4))
    plt.plot(t, dof_pos_arr, label="qpos (rad)")
    plt.plot(t, link_pos_arr, label="mocap link pos (m)")
    plt.title("Raw Signals")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dof", type=str, default="left_elbow_joint")
    parser.add_argument("--link", type=str, default="left_rubber_hand")
    args = parser.parse_args()
    main(args)
