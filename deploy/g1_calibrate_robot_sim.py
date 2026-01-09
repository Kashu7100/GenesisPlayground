import argparse
import sys
import termios
import tty
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_kinematics as pk
import torch
import yaml
from g1_r2s_config import G1_CB1_LINK_NAMES, G1_CB2_LINK_NAMES, G1_FK_TABLES, G1_R_WEIGHT
from genesis.utils.geom import _tc_quat_to_R as torch_quat_to_R
from genesis.utils.geom import _tc_R_to_quat as torch_R_to_quat
from genesis.utils.geom import inv_quat, transform_by_quat
from gs_env.common.utils.math_utils import (
    np_pose_mul,
    pose_diff,
    pose_inv,
    pose_mul,
    rot6d_to_rotmat,
)
from gs_env.sim.envs.config.registry import EnvArgsRegistry as sim_env_registry
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.envs.locomotion.custom_env import CustomEnv
from gs_env.sim.robots.config.schema import HumanoidRobotArgs
from tqdm import tqdm


def getch() -> str:
    """Non-blocking single-key input (Linux/macOS)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    # Parse accurate offset data path
    acc_path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "optitrack_offset"
        / (args.acc_config + ".yaml")
    )

    collected_data: list[dict[str, Any]] = []

    env_args = sim_env_registry["custom_g1_mocap"]
    assert isinstance(env_args, LeggedRobotEnvArgs)
    sim_env = CustomEnv(args=env_args, num_envs=1, show_viewer=False, device=device)

    with open(acc_path) as f:
        acc_offset_raw = yaml.safe_load(f)
    print(f"[INFO] Loaded accurate offset data from {acc_path}.")
    acc_offset: dict[str, dict[str, torch.Tensor]] = {}
    for link_name in G1_CB1_LINK_NAMES:
        acc_offset[link_name] = {}
        acc_offset[link_name]["pos"] = torch.tensor(
            acc_offset_raw[link_name]["pos"], dtype=torch.float32
        )
        acc_offset[link_name]["rot"] = torch_quat_to_R(
            torch.tensor(acc_offset_raw[link_name]["quat"], dtype=torch.float32)
        )

    # Create fake error and inv_offsets
    ERROR_SCALE, NOISE_SCALE, N_SAMPLES = 0.02, 0.01, 100
    qpos_error = torch.randn(29, dtype=torch.float32) * ERROR_SCALE
    link_offsets = {}
    for name in G1_CB1_LINK_NAMES:
        link_offsets[name] = {
            "pos": np.array(acc_offset_raw[name]["pos"], dtype=np.float32),
            "quat": np.array(acc_offset_raw[name]["quat"], dtype=np.float32),
        }
        link_offsets[name]["quat"] = inv_quat(link_offsets[name]["quat"])
        link_offsets[name]["pos"] = -transform_by_quat(
            link_offsets[name]["pos"], link_offsets[name]["quat"]
        )
    for name in G1_CB2_LINK_NAMES:
        link_offsets[name] = {
            "pos": np.random.uniform(-0.1, 0.1, size=(3,)).astype(np.float32),
            "quat": np.random.uniform(-0.2, 0.2, size=(4,)).astype(np.float32),
        }
        link_offsets[name]["quat"][0] = 1.0
        link_offsets[name]["quat"] /= np.linalg.norm(link_offsets[name]["quat"])

    # Sample data from sim env
    print("[INFO] Starting collection...")
    for _ in range(N_SAMPLES):
        sampled_qpos = np.random.uniform(-0.5, 0.5, size=(29,)).astype(np.float32)
        sim_env.set_dof_pos(torch.tensor(sampled_qpos))
        link_idx_local = sim_env.get_link_idx_local_by_name("imu_in_torso")
        sim_env.set_link_pose(link_idx_local, quat=torch.tensor([1.0, 0.0, 0.0, 0.0]))
        noisy_qpos = (
            sim_env.dof_pos[0].cpu().numpy().astype(np.float32)
            + qpos_error.cpu().numpy().astype(np.float32)
            + np.random.randn(29).astype(np.float32) * NOISE_SCALE
        )
        link_poses = {}
        for link_name in G1_CB1_LINK_NAMES + G1_CB2_LINK_NAMES:
            link_idx_local = sim_env.get_link_idx_local_by_name(link_name)
            pos_tensor, quat_tensor = sim_env.get_link_pose(link_idx_local)
            pos = pos_tensor.cpu().numpy().astype(np.float32)
            quat = quat_tensor.cpu().numpy().astype(np.float32)
            quat, pos = np_pose_mul(
                pos,
                quat,
                link_offsets[link_name]["pos"],
                link_offsets[link_name]["quat"],
            )
            link_poses[link_name] = (pos, quat)
        qpos = noisy_qpos
        data = {
            "link_poses": link_poses,
            "qpos": qpos,
        }
        collected_data.append(data)
        # time.sleep(0.1)
        # sim_env.step_visualizer()
        print(f"[INFO] Sample #{len(collected_data)} collected.")

    urdf_path = (
        Path(__file__).resolve().parent.parent
        / "assets"
        / "robot"
        / "unitree_g1"
        / "g1_custom_collision_29dof.urdf"
    )
    chain = pk.build_chain_from_urdf(open(urdf_path).read())
    pk_joint_order = chain.get_joint_parameter_names()
    real_env_args = sim_env_registry["g1_walk"]
    assert isinstance(real_env_args.robot_args, HumanoidRobotArgs)
    gs_joint_order = real_env_args.robot_args.dof_names
    assert len(gs_joint_order) == len(pk_joint_order)
    N, Samples = len(pk_joint_order), len(collected_data)

    # Batch the data w.r.t. samples
    real_qpos: torch.Tensor = torch.stack(
        [
            torch.tensor(
                [data["qpos"][gs_joint_order.index(name)] for name in pk_joint_order],
                dtype=torch.float32,
            )
            for data in collected_data
        ],
        dim=0,
    )  # Samples x N
    mocap_pos: dict[str, torch.Tensor] = {}
    mocap_rot: dict[str, torch.Tensor] = {}
    for link_name in collected_data[0]["link_poses"].keys():
        mocap_pos[link_name] = torch.stack(
            [
                torch.tensor(
                    data["link_poses"][link_name][0],
                    dtype=torch.float32,
                )
                for data in collected_data
            ],
            dim=0,
        )  # Samples x 3
        quats = torch.stack(
            [
                torch.tensor(
                    data["link_poses"][link_name][1],
                    dtype=torch.float32,
                )
                for data in collected_data
            ],
            dim=0,
        )  # Samples x 4
        mocap_rot[link_name] = torch_quat_to_R(quats)  # Samples x 3 x 3

    qpos_offset = torch.zeros(N, dtype=torch.float32, device=device, requires_grad=True)
    # in pk order

    opt_offset: dict[str, dict[str, torch.Tensor]] = {}
    # pos: 3, rot: 6 (6D continuous representation)
    for link_name in G1_CB2_LINK_NAMES:
        opt_offset[link_name] = {
            "pos": torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True),
            "rot": torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),
        }

    opt1 = torch.optim.Adam([v for d in opt_offset.values() for v in d.values()], lr=0.001)
    opt2 = torch.optim.Adam(
        [qpos_offset] + [v for d in opt_offset.values() for v in d.values()], lr=0.001
    )
    rms_R_history, rms_T_history = [], []
    for step in tqdm(range(2000)):
        opt = opt1 if step < 1000 else opt2
        opt.zero_grad()
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        qpos_calibrated = real_qpos + qpos_offset.unsqueeze(0)  # Samples x N
        fk_results = chain.forward_kinematics(qpos_calibrated)

        rms_Rs, rms_Ts = [], []
        for fk_item in G1_FK_TABLES:
            b_link_name = fk_item["parent"]
            e_link_name = fk_item["son"]
            b_fk_pos = fk_results[b_link_name].get_matrix()[:, :3, 3]
            b_fk_rot = fk_results[b_link_name].get_matrix()[:, :3, :3]
            e_fk_pos = fk_results[e_link_name].get_matrix()[:, :3, 3]
            e_fk_rot = fk_results[e_link_name].get_matrix()[:, :3, :3]
            b_mocap_pos = mocap_pos[b_link_name]
            b_mocap_rot = mocap_rot[b_link_name]
            e_mocap_pos = mocap_pos[e_link_name]
            e_mocap_rot = mocap_rot[e_link_name]

            if b_link_name in G1_CB1_LINK_NAMES:
                b_offset_R = acc_offset[b_link_name]["rot"]
                b_offset_T = acc_offset[b_link_name]["pos"]
            else:
                b_offset_R = rot6d_to_rotmat(opt_offset[b_link_name]["rot"])
                b_offset_T = opt_offset[b_link_name]["pos"]
            b_offset_R = b_offset_R.unsqueeze(0).repeat(Samples, 1, 1)
            b_offset_T = b_offset_T.unsqueeze(0).repeat(Samples, 1)

            if e_link_name in G1_CB1_LINK_NAMES:
                raise ValueError("Calibrated links should not be as end links in FK chain.")
            else:
                e_offset_R = rot6d_to_rotmat(opt_offset[e_link_name]["rot"])
                e_offset_T = opt_offset[e_link_name]["pos"]
            e_offset_R = e_offset_R.unsqueeze(0).repeat(Samples, 1, 1)
            e_offset_T = e_offset_T.unsqueeze(0).repeat(Samples, 1)

            fk_offset_R, fk_offset_T = pose_diff(
                b_fk_pos,
                b_fk_rot,
                e_fk_pos,
                e_fk_rot,
            )  # FK local transform
            b1_rot, b1_pos = pose_mul(
                b_mocap_pos,
                b_mocap_rot,
                b_offset_T,
                b_offset_R,
            )  # GT root link pose
            e1_rot, e1_pos = pose_mul(
                b1_pos,
                b1_rot,
                fk_offset_T,
                fk_offset_R,
            )  # GT end link pose
            e_offset_R_inv, e_offset_T_inv = pose_inv(e_offset_T, e_offset_R)
            e2_rot, e2_pos = pose_mul(
                e1_pos,
                e1_rot,
                e_offset_T_inv,
                e_offset_R_inv,
            )  # expected mocap end link pose

            e2_quat = torch_R_to_quat(e2_rot)  # Samples x 4
            e_mocap_quat = torch_R_to_quat(e_mocap_rot)  # Samples x 4
            R_diff = 1.0 - torch.sum(e2_quat * e_mocap_quat, dim=-1) ** 2  # Samples
            T_diff = torch.sum(torch.square(e2_pos - e_mocap_pos), dim=-1)  # Samples

            total_loss += torch.mean(R_diff) * G1_R_WEIGHT
            total_loss += torch.mean(T_diff) * (1 - G1_R_WEIGHT)

            rms_R = torch.mean(R_diff)
            rms_T = torch.sqrt(torch.mean(T_diff))
            rms_Rs.append(rms_R.item())
            rms_Ts.append(rms_T.item())

        rms_R_history.append(np.mean(rms_Rs))
        rms_T_history.append(np.mean(rms_Ts))
        total_loss += 0.0 * torch.sum(torch.square(qpos_offset))
        total_loss.backward()

        opt.step()

    print("Fake link offsets (inv):")
    for name in G1_CB2_LINK_NAMES:
        print(f"Link: {name}")
        print(link_offsets[name]["pos"])
        print(link_offsets[name]["quat"])
    print("Optimized link offsets:")
    for name in G1_CB2_LINK_NAMES:
        rot6d = opt_offset[name]["rot"].detach().cpu().numpy()
        rotmat = (
            rot6d_to_rotmat(torch.tensor(rot6d, dtype=torch.float32).unsqueeze(0))
            .squeeze(0)
            .numpy()
        )
        quat = (
            torch_R_to_quat(torch.tensor(rotmat, dtype=torch.float32).unsqueeze(0))
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
        pos = opt_offset[name]["pos"].detach().cpu().numpy()
        print(f"Link: {name}")
        print(pos)
        print(quat)
    print("Fake error:")
    qpos_error_gs = qpos_error.detach().cpu().numpy()
    qpos_error_fk = np.array(
        [qpos_error_gs[gs_joint_order.index(name)] for name in pk_joint_order],
        dtype=np.float32,
    )
    print(-qpos_error_fk)
    print("Optimized error:")
    print(qpos_offset.detach().cpu().numpy())

    plt.figure(figsize=(6, 4))
    plt.plot(rms_R_history[900:], label="Rotation Error")
    plt.plot(rms_T_history[900:], label="Translation Error")
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("Calibration Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_config", type=str, default="calibrated")
    parser.add_argument("--acc_config", type=str, default="foot")
    # robot qpos calibration is irrelevant to mocap offsets
    # save to robot offset folder
    args = parser.parse_args()
    main(args)
