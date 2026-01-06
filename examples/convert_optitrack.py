import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, cast

import gs_env.sim.envs as gs_envs
import torch
import yaml
from gs_env.common.utils.math_utils import quat_apply, quat_from_angle_axis, quat_mul, quat_to_euler
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import MotionEnvArgs

sys.path.insert(0, str(Path(__file__).parent.parent))
from deploy.utils import G1Retargeter


def optitrack_to_motion_data(
    env: gs_envs.MotionEnv,
    optitrack_data: dict[str, Any],
    retargeter: G1Retargeter,
    show_viewer: bool = False,
) -> None | dict[str, Any]:
    """
    Convert OptiTrack pickle data (from optitrack_publisher.py) to motion data.
    Uses G1Retargeter to convert tracked link poses to robot-space motion.
    DOF positions are set to zeros as requested.

    Args:
        env: Motion environment instance
        optitrack_data: Dictionary with keys: fps, link_names, pos, quat, frame_id
        retargeter: G1Retargeter instance for converting tracked links to robot state
        show_viewer: Whether to render the environment

    Returns:
        motion_data: The motion data dictionary
        None: If interrupted by user
    """

    dof_names = env.dof_names

    optitrack_link_names = list(optitrack_data["link_names"])
    link_names = env.args.tracking_link_names
    link_name_to_idx = {
        "left_ankle_roll_link": optitrack_link_names.index("LeftFoot"),
        "right_ankle_roll_link": optitrack_link_names.index("RightFoot"),
        "left_wrist_yaw_link": optitrack_link_names.index("LeftHand"),
        "right_wrist_yaw_link": optitrack_link_names.index("RightHand"),
        "torso_link": optitrack_link_names.index("Spine1"),
        "pelvis": optitrack_link_names.index("Hips"),
    }

    motion_data = {}
    motion_data["fps"] = optitrack_data["fps"]
    motion_data["link_names"] = link_names
    motion_data["dof_names"] = dof_names
    # save foot link indices and names for downstream usage
    foot_links_idx = env.robot.foot_links_idx
    motion_data["foot_link_indices"] = foot_links_idx

    # Extract data from optitrack pickle
    tracked_pos = optitrack_data["pos"]  # (num_frames, 6, 3)
    tracked_quat = optitrack_data["quat"]  # (num_frames, 6, 4)
    frame_ids = optitrack_data["frame_id"]  # (num_frames,)

    num_frames = tracked_pos.shape[0]
    dof_dim = env.robot.dof_dim

    pos_list = []
    quat_list = []
    dof_pos_list = []
    link_pos_list = []
    link_quat_list = []
    foot_contact_list = []

    def run() -> dict[str, Any] | None:
        nonlocal env, retargeter, motion_data, show_viewer
        last_update_time = time.time()
        foot_links_idx = env.robot.foot_links_idx

        prev_frame_id: int | None = None
        last_valid_retargeted: dict[str, torch.Tensor] | None = None

        i = 0
        while i < num_frames - 1:
            # Get tracked link poses for this frame
            frame_tracked_pos = tracked_pos[i]  # (6, 3)
            frame_tracked_quat = tracked_quat[i]  # (6, 4)
            frame_id = frame_ids[i]
            frame_foot_contact = optitrack_data["foot_contact"][i]  # (2,)

            # Check if there are skipped frame IDs
            if prev_frame_id is not None and frame_id - prev_frame_id > 1:
                # Frame IDs were skipped, use the latest valid retargeted value
                if last_valid_retargeted is not None:
                    retargeted = last_valid_retargeted
                    frame_id = prev_frame_id + 1
                    i -= 1
                else:
                    continue
            else:
                # Retarget tracked links to robot-space base pose and link poses
                retargeted = retargeter.step(
                    tracked_pos=frame_tracked_pos,
                    tracked_quat=frame_tracked_quat,
                    frame_id=frame_id,
                )
            i += 1

            # Store the last valid retargeted output
            last_valid_retargeted = retargeted
            prev_frame_id = frame_id

            # Extract retargeted base pose
            base_pos = retargeted["base_pos"].clone()  # (3,)
            base_quat = retargeted["base_quat"].clone()  # (4,)
            dof_pos = torch.zeros(dof_dim, dtype=torch.float32)
            env.robot.set_state(
                pos=base_pos.unsqueeze(0),
                quat=base_quat.unsqueeze(0),
                dof_pos=dof_pos.unsqueeze(0),
            )
            quat_yaw = quat_from_angle_axis(
                quat_to_euler(base_quat)[2], torch.tensor([0.0, 0.0, 1.0])
            )
            link_pos = quat_apply(quat_yaw, retargeted["link_pos_local"].clone())
            link_pos[:, :2] += base_pos[None, :2]
            link_quat = quat_mul(quat_yaw, retargeted["link_quat_local"].clone())
            if show_viewer:
                for link_name in env.scene.objects.keys():  # type: ignore
                    if link_name in link_name_to_idx:
                        link_idx = link_name_to_idx[link_name]
                        env.scene.set_obj_pose(
                            link_name,
                            pos=link_pos[None, link_idx, :],
                            quat=link_quat[None, link_idx, :],
                        )

            env.update_buffers()

            # Extract state from environment
            pos_list.append(retargeted["base_pos"].clone())
            quat_list.append(retargeted["base_quat"].clone())
            dof_pos_list.append(env.dof_pos[0].clone())  # Store zeros as requested
            link_pos_list.append(link_pos.clone())
            link_quat_list.append(link_quat.clone())
            foot_contact_list.append(frame_foot_contact.clone())

            foot_pos = link_pos[[0, 1], :]
            if show_viewer:
                env.scene.scene.clear_debug_objects()
                for j in range(len(foot_links_idx)):
                    env.scene.scene.draw_debug_arrow(
                        foot_pos[j],
                        frame_foot_contact[j] * torch.tensor([0.0, 0.0, 0.5]),
                        radius=0.01,
                        color=(0.0, 0.0, 1.0),
                    )

                env.scene.scene.step()
                while time.time() - last_update_time < 1 / motion_data["fps"]:
                    time.sleep(0.01)
                last_update_time = time.time()

        if len(pos_list) == 0:
            print("Warning: No frames were processed (possibly only calibration frame)")
            return None

        motion_data["pos"] = torch.stack(pos_list).numpy()
        motion_data["quat"] = torch.stack(quat_list).numpy()
        motion_data["dof_pos"] = torch.stack(dof_pos_list).numpy()
        motion_data["link_pos"] = torch.stack(link_pos_list).numpy()
        motion_data["link_quat"] = torch.stack(link_quat_list).numpy()
        motion_data["foot_contact"] = torch.stack(foot_contact_list).numpy()

        return motion_data

    try:
        return run()
    except KeyboardInterrupt:
        return None


if __name__ == "__main__":
    show_viewer = True

    # Find pickle files saved by optitrack_publisher.py in assets/OptiTrack
    # pkl_files = list(Path("./assets/OptiTrack").glob("*.pkl"))
    pkl_files = ["./assets/OptiTrack/walk_straight_0.pkl"]

    log_dir = Path("./assets/motion/optitrack")
    os.makedirs(log_dir, exist_ok=True)

    env_args = cast(MotionEnvArgs, EnvArgsRegistry["g1_motion"])
    env = gs_envs.MotionEnv(
        args=env_args,
        num_envs=1,
        show_viewer=show_viewer,
        device=torch.device("cpu"),
        eval_mode=True,
    )
    env.reset()

    dataset_yaml = {
        "root_path": str(log_dir),
        "motions": [],
    }

    for pkl_file in pkl_files:
        # Create a new retargeter for each file to ensure clean calibration state
        retargeter = G1Retargeter()
        print(f"Processing {pkl_file}...")
        try:
            # Load OptiTrack data saved by optitrack_publisher.py
            with open(pkl_file, "rb") as f:
                optitrack_data = pickle.load(f)

            # Verify expected keys
            required_keys = ["fps", "link_names", "pos", "quat", "frame_id"]
            if not all(key in optitrack_data for key in required_keys):
                print(f"Warning: {pkl_file} missing required keys. Expected: {required_keys}")
                print(f"Found keys: {list(optitrack_data.keys())}")
                continue

            motion_name = Path(pkl_file).stem
            motion_file = motion_name + ".pkl"
            motion_path = log_dir / motion_file

            motion_data = optitrack_to_motion_data(
                env=env,
                optitrack_data=optitrack_data,
                retargeter=retargeter,
                show_viewer=show_viewer,
            )

            if motion_data is not None:
                print(f"Saving motion data to {motion_path}")
                with open(motion_path, "wb") as f:
                    pickle.dump(motion_data, f)
                dataset_yaml["motions"].append(
                    {
                        "file": motion_file,
                        "weight": 1.0,
                    }
                )
            else:
                print(f"Skipping motion data for {pkl_file}")

        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
            import traceback

            traceback.print_exc()
            continue

    dataset_yaml["motions"].sort(key=lambda x: x["file"])
    yaml_path = log_dir / "optitrack.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f)
    print(f"\nDataset YAML saved to {yaml_path}")
