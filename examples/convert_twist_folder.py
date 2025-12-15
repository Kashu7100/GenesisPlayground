import os
import pickle
import time
from pathlib import Path
from typing import Any, cast
import argparse

import gs_env.sim.envs as gs_envs
import torch
from gs_env.common.utils.math_utils import quat_to_euler
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.scenes.config.registry import SceneArgsRegistry


def twist_to_motion_data(
    env: gs_envs.MotionEnv, data: dict[str, Any], show_viewer: bool = False
) -> None | dict[str, Any]:
    """
    Convert TWIST data to motion data
    If show_viewer is True, the environment will be rendered in a separate thread.
    Returns:
        motion_data: The motion data
        None: If show_viewer is True, the environment will be rendered in a separate thread.
    """

    links = env.robot.robot.links
    link_names = [link.name for link in links]
    dof_names = env.dof_names

    twist_order = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    dof_index = []
    for dof_name in twist_order:
        dof_index.append(dof_names.index(dof_name))

    motion_data = {}
    motion_data["fps"] = data["fps"]
    motion_data["link_names"] = link_names
    motion_data["dof_names"] = dof_names
    # save foot link indices and names for downstream usage
    foot_links_idx = env.robot.foot_links_idx
    motion_data["foot_link_indices"] = foot_links_idx
    pos_list = []
    quat_list = []
    dof_pos_list = []
    link_pos_list = []
    link_quat_list = []
    foot_contact_list = []

    def run() -> dict[str, Any]:
        nonlocal env, data, motion_data, show_viewer, dof_index
        last_update_time = time.time()
        foot_links_idx = env.robot.foot_links_idx
        for i in range(len(data["root_pos"])):
            pos = torch.tensor(data["root_pos"][i], dtype=torch.float32)
            quat = torch.tensor(data["root_rot"][i], dtype=torch.float32)[[3, 0, 1, 2]]
            dof_pos = torch.zeros(29, dtype=torch.float32)
            dof_pos[dof_index] = torch.tensor(data["dof_pos"][i], dtype=torch.float32)
            env.robot.set_state(pos=pos, quat=quat, dof_pos=dof_pos)
            env.update_buffers()
            pos_list.append(env.base_pos[0].clone())
            quat_list.append(env.base_quat[0].clone())
            dof_pos_list.append(env.dof_pos[0].clone())
            link_pos_list.append(env.link_positions[0].clone())
            link_quat_list.append(env.link_quaternions[0].clone())

            # compute foot contact
            foot_pos = env.link_positions[0, foot_links_idx, :]
            foot_quat = env.link_quaternions[0, foot_links_idx, :]
            foot_euler = quat_to_euler(foot_quat)
            foot_tilt = torch.clamp(
                (torch.abs(foot_euler[:, 0]) + torch.abs(foot_euler[:, 1]) - 0.4) / 0.4, 0.0, 1.0
            )
            foot_lift = torch.clamp((foot_pos[:, 2] - 0.15) / 0.15, 0.0, 1.0)
            if i == 0:
                foot_last_pos = foot_pos.clone()
            foot_vel = torch.clamp(
                torch.norm((foot_pos[..., :2] - foot_last_pos[..., :2]) / env.dt, dim=-1) - 0.15,
                0.0,
                1.0,
            )
            foot_last_pos = foot_pos.clone()
            foot_not_contact = ((foot_tilt + foot_lift + foot_vel) / 1.5).clamp(0.0, 1.0)
            foot_contact = 1 - foot_not_contact
            foot_contact_list.append(foot_contact)
            if show_viewer:
                env.scene.scene.clear_debug_objects()
                for i in range(len(foot_links_idx)):
                    env.scene.scene.draw_debug_arrow(
                        foot_pos[i],
                        foot_contact[i] * torch.tensor([0.0, 0.0, 0.5]),
                        radius=0.01,
                        color=(0.0, 0.0, 1.0),
                    )

            if show_viewer:
                env.scene.scene.step(refresh_visualizer=False)
                while time.time() - last_update_time < 1 / motion_data["fps"]:
                    time.sleep(0.01)
                last_update_time = time.time()

        motion_data["pos"] = torch.stack(pos_list).numpy()
        motion_data["quat"] = torch.stack(quat_list).numpy()
        motion_data["dof_pos"] = torch.stack(dof_pos_list).numpy()
        motion_data["link_pos"] = torch.stack(link_pos_list).numpy()
        motion_data["link_quat"] = torch.stack(link_quat_list).numpy()
        motion_data["foot_contact"] = torch.stack(foot_contact_list).numpy()
        return motion_data

    try:
        if show_viewer:
            import threading

            threading.Thread(target=run).start()
            env.scene.scene.viewer.run()  # type: ignore
        else:
            return run()
    except KeyboardInterrupt:
        return None


if __name__ == "__main__":
    show_viewer = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str, required=True, help="Folder of .pkl motions")
    parser.add_argument("--tgt_folder", type=str, required=False, help="Output folder")
    args = parser.parse_args()

    src = Path(args.src_folder)
    tgt = Path(args.tgt_folder) if args.tgt_folder else Path("./assets/motion_out")
    tgt.mkdir(parents=True, exist_ok=True)

    env_args = cast(MotionEnvArgs, EnvArgsRegistry["g1_motion"]).model_copy(
        update={"scene_args": SceneArgsRegistry["flat_scene_legged"]}
    )
    env = gs_envs.MotionEnv(
        args=env_args,
        num_envs=1,
        show_viewer=show_viewer,
        device=torch.device("cpu"),
        eval_mode=True,
    )
    env.reset()

    for pkl_file in src.rglob("*.pkl"):
        print(f"Processing {pkl_file}")
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        motion_data = twist_to_motion_data(env, data, show_viewer=show_viewer)
        if motion_data:
            # path of this file relative to src root
            rel_path = pkl_file.relative_to(src)      # e.g. Subject_1_F_MoSh/seq1.pkl

            # same structure under target root
            out_path = tgt / rel_path                 # BMLmovi_retarget/Subject_1_F_MoSh/seq1.pkl
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "wb") as f:
                pickle.dump(motion_data, f)
            print(f"Saved -> {out_path}")

