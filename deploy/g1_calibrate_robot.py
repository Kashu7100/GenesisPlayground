import argparse
import pickle
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
from gs_env.real.config.registry import EnvArgsRegistry as real_env_registry
from gs_env.real.config.schema import OptitrackEnvArgs
from gs_env.real.leggedrobot_env import UnitreeLeggedEnv
from gs_env.real.optitrack_env import OptitrackEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry as sim_env_registry
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.robots.config.schema import HumanoidRobotArgs
from tqdm import tqdm


def torch_rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrices.
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def torch_inverse_RT(R: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse local transform (R, T)
    R_inv = R^t
    T_inv = -R^t * T
    """
    R_out = R.transpose(-1, -2)
    T_out = -(R_out @ T.unsqueeze(-1)).squeeze(-1)
    return R_out, T_out


def torch_transform_RT_by(
    R1: torch.Tensor, T1: torch.Tensor, R2: torch.Tensor, T2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply local transform (R2, T2) in the frame of (R1, T1)
    R = R1 * R2
    T = R1 * T2 + T1
    """
    R_out = R1 @ R2
    T_out = (R1 @ T2.unsqueeze(-1)).squeeze(-1) + T1
    return R_out, T_out


def torch_get_RT_between(
    R1: torch.Tensor, T1: torch.Tensor, R2: torch.Tensor, T2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get local transform from (R1, T1) to (R2, T2)
    R = R1^t * R2
    T = R1^t * (T2 - T1)
    """
    R_out = R1.transpose(-1, -2) @ R2
    T_out = (R1.transpose(-1, -2) @ (T2 - T1).unsqueeze(-1)).squeeze(-1)
    return R_out, T_out


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
    # Parse save path
    save_path_optitrack = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "optitrack_offset"
        / (args.save_config + ".yaml")
    )
    save_path_robot = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "robot_offset"
        / (args.save_config + ".yaml")
    )

    # Parse presample data path
    data_path = (
        Path(__file__).resolve().parent.parent / "config" / "robot_offset" / "collected_data.pkl"
    )

    # Parse accurate offset data path
    acc_path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "optitrack_offset"
        / (args.acc_config + ".yaml")
    )

    collected_data: list[dict[str, Any]] = []
    if not args.presample:
        # Create OptiTrack env with zero offsets
        load_config_path = (
            Path(__file__).resolve().parent.parent / "config" / "optitrack_offset" / "default.yaml"
        )
        optitrack_env_args = real_env_registry["g1_links_tracking"].model_copy(
            update={"offset_config": load_config_path},
        )
        assert isinstance(optitrack_env_args, OptitrackEnvArgs)
        optitrack_env = OptitrackEnv(num_envs=1, args=optitrack_env_args)

        # Create low state handler
        real_env_args = sim_env_registry["g1_walk"]
        assert isinstance(real_env_args, LeggedRobotEnvArgs)
        real_env = UnitreeLeggedEnv(args=real_env_args, interactive=False)

        print("[INFO] Starting collection...")
        print("       'c' to capture a sample, 'q' to stop and calibrate,")
        print("       's' to save collected data and exit.")
        while True:
            key = getch()
            if key == "c":
                link_poses = optitrack_env.get_tracked_links()
                qpos = real_env.dof_pos[0].cpu().numpy().astype(np.float32)
                data = {
                    "link_poses": link_poses,
                    "qpos": qpos,
                }
                collected_data.append(data)
                print(f"[INFO] Sample #{len(collected_data)} collected.")
            elif key == "q":
                print(f"[INFO] Total samples: {len(collected_data)}, starting calibration...")
                break
            elif key == "s":
                print(f"[INFO] Saving {len(collected_data)} samples to {data_path} and exiting...")
                with open(data_path, "wb") as f:
                    pickle.dump(collected_data, f)
                return
    else:
        with open(data_path, "rb") as f:
            collected_data = pickle.load(f)
        print(f"[INFO] Loaded {len(collected_data)} samples, starting calibration...")

    if len(collected_data) <= 1:
        print("[ERROR] Not enough samples collected, exit.")
        return

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

    device = torch.device("cpu")
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
                b_offset_R = torch_rot6d_to_rotmat(opt_offset[b_link_name]["rot"])
                b_offset_T = opt_offset[b_link_name]["pos"]
            b_offset_R = b_offset_R.unsqueeze(0).repeat(Samples, 1, 1)
            b_offset_T = b_offset_T.unsqueeze(0).repeat(Samples, 1)

            if e_link_name in G1_CB1_LINK_NAMES:
                raise ValueError("Calibrated links should not be as end links in FK chain.")
            else:
                e_offset_R = torch_rot6d_to_rotmat(opt_offset[e_link_name]["rot"])
                e_offset_T = opt_offset[e_link_name]["pos"]
            e_offset_R = e_offset_R.unsqueeze(0).repeat(Samples, 1, 1)
            e_offset_T = e_offset_T.unsqueeze(0).repeat(Samples, 1)

            fk_offset_R, fk_offset_T = torch_get_RT_between(
                b_fk_rot,
                b_fk_pos,
                e_fk_rot,
                e_fk_pos,
            )  # FK local transform
            b1_rot, b1_pos = torch_transform_RT_by(
                b_mocap_rot,
                b_mocap_pos,
                b_offset_R,
                b_offset_T,
            )  # GT root link pose
            e1_rot, e1_pos = torch_transform_RT_by(
                b1_rot,
                b1_pos,
                fk_offset_R,
                fk_offset_T,
            )  # GT end link pose
            e_offset_R_inv, e_offset_T_inv = torch_inverse_RT(e_offset_R, e_offset_T)
            e2_rot, e2_pos = torch_transform_RT_by(
                e1_rot,
                e1_pos,
                e_offset_R_inv,
                e_offset_T_inv,
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

    # print(qpos_offset.detach().cpu().numpy())
    save_offset = acc_offset_raw.copy()
    for link_name in G1_CB2_LINK_NAMES:
        save_offset[link_name] = {}
        save_offset[link_name]["pos"] = opt_offset[link_name]["pos"].detach().cpu().numpy().tolist()
        rot_mat = torch_rot6d_to_rotmat(opt_offset[link_name]["rot"])
        quat = torch_R_to_quat(rot_mat)
        save_offset[link_name]["quat"] = quat.detach().cpu().numpy().tolist()

    def represent_list(dumper: yaml.Dumper, data: list[Any]) -> yaml.nodes.SequenceNode:
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    yaml.add_representer(list, represent_list)
    with open(save_path_optitrack, "w") as f:
        yaml.dump(save_offset, f, sort_keys=False)
    print(f"[INFO] Optitrack calibration offsets saved to {save_path_optitrack}.")

    save_q = {}
    for i, name in enumerate(pk_joint_order):
        save_q[name] = qpos_offset[i].detach().cpu().numpy().item()
    with open(save_path_robot, "w") as f:
        yaml.dump(save_q, f, sort_keys=False)
    print(f"[INFO] Robot calibration offsets saved to {save_path_robot}.")

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
    parser.add_argument("--presample", action="store_true", default=False)
    # robot qpos calibration is irrelevant to mocap offsets
    # save to robot offset folder
    args = parser.parse_args()
    main(args)
