import time

import fire
import torch
from gs_env.real.leggedrobot_env import UnitreeLeggedEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.locomotion.leggedrobot_env import LeggedRobotEnv


def main(
    device: str = "cpu",
) -> None:
    device = "cpu" if not torch.cuda.is_available() else device
    device = torch.device(device)  # type: ignore[arg-type]

    env_args = EnvArgsRegistry["g1_fixed"]
    sim_env = LeggedRobotEnv(args=env_args, num_envs=1, show_viewer=True, device=device)  # type: ignore[arg-type]

    env_args = EnvArgsRegistry["g1_walk"]
    real_env = UnitreeLeggedEnv(args=env_args, interactive=False, device=device)  # type: ignore[arg-type]

    print("=" * 80)
    print("Starting visualization")
    print(f"Device: {device}")
    print("=" * 80)

    try:
        last_update_time = time.time()
        while True:
            # Control loop timing (50 Hz)
            if time.time() - last_update_time < 0.02:
                time.sleep(0.005)
                continue
            last_update_time = time.time()

            sim_env.set_dof_pos(real_env.dof_pos[0])
            # sim_env.robot.set_state(quat=real_env.quat[0])
            link_idx_local = sim_env.get_link_idx_local_by_name("pelvis")
            sim_env.set_link_pose(link_idx_local, quat=real_env.base_quat[0])
            sim_env.step_visualizer()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    fire.Fire(main)
