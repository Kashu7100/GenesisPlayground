import argparse
import json
import pickle
import sys
import termios
import time
import tty

import redis
import torch
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_diff,
    quat_from_euler,
    quat_inv,
    quat_mul,
    quat_to_angle_axis,
    quat_to_euler,
)
from gs_env.real.config.registry import EnvArgsRegistry
from gs_env.real.config.schema import OptitrackEnvArgs
from gs_env.real.optitrack.NatNetClient import setup_optitrack
from gs_env.real.optitrack.optitrack_config import RIGID_BODY_ID_MAP, track_id_offset


def _to_list(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().flatten().tolist()


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


def calc_global(
    T1: torch.Tensor, R1: torch.Tensor, T2: torch.Tensor, R2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    R_out = quat_mul(R1, R2)
    T_out = quat_apply(R1, T2) + T1
    return T_out, R_out


def calc_local(
    T1: torch.Tensor, R1: torch.Tensor, T2: torch.Tensor, R2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    R_out = quat_mul(quat_inv(R1), R2)
    T_out = quat_apply(quat_inv(R1), T2 - T1)
    return T_out, R_out


class OptitrackPublisher:
    """
    Publishes OptiTrack skeleton poses into Redis.

    Redis keys:
        - {key}:motion:base_pos [3]
        - {key}:motion:base_quat [4] (w, x, y, z)
        - {key}:motion:base_lin_vel [3]
        - {key}:motion:base_ang_vel [3]
        - {key}:motion:base_ang_vel_local [3]
        - {key}:motion:link_pos_local [N*3] (filtered to tracking links if specified)
        - {key}:motion:link_quat_local [N*4] (filtered to tracking links if specified)
        - {key}:motion:link_lin_vel [N*3]
        - {key}:motion:link_ang_vel [N*3]
        - {key}:motion:foot_contact [F]
        - {key}:timestamp:base_pos [1]
        - {key}:timestamp:base_quat [1]
        - {key}:timestamp:base_lin_vel [1]
        - {key}:timestamp:base_ang_vel [1]
        - {key}:timestamp:base_ang_vel_local [1]
        - {key}:timestamp:link_pos_local [1]
        - {key}:timestamp:link_quat_local [1]
        - {key}:timestamp:link_lin_vel [1]
        - {key}:timestamp:link_ang_vel [1]
        - {key}:timestamp:foot_contact [1]
    """

    SKELETON_ORDER = [RIGID_BODY_ID_MAP[i + track_id_offset] for i in range(1, 52)]

    MOTION_NAMES = ["LeftFoot", "RightFoot", "LeftHand", "RightHand", "Spine1", "Hips"]

    def __init__(
        self,
        redis_url: str,
        key: str,
        server_ip: str,
        client_ip: str,
        use_multicast: bool,
        save: bool,
        freq_hz: float,
    ) -> None:
        self.r = redis.from_url(redis_url)

        self.key = key
        self.freq_hz = freq_hz
        self.frame_id = 0
        self.frame_rate = 120.0
        self.save = save
        self.save_data = {
            "fps": 120,
            "pos6": [],
            "quat6": [],
            "frame_id": [],
        }

        optitrack_env_args = EnvArgsRegistry["g1_links_tracking"]
        assert isinstance(optitrack_env_args, OptitrackEnvArgs)
        self.server_ip = server_ip if server_ip != "0.0.0.0" else optitrack_env_args.server_ip
        self.client_ip = client_ip if client_ip != "0.0.0.0" else optitrack_env_args.client_ip
        self.client = setup_optitrack(
            server_address=server_ip,
            client_address=client_ip,
            use_multicast=use_multicast,
        )

        self.zero_link_pos_local = torch.tensor(
            [
                [0.0, 0.1, 0.04],
                [0.0, -0.1, 0.04],
                [0.2, 0.2, 0.87],
                [0.2, -0.2, 0.87],
                [0.0, 0.0, 0.83],
                [0.0, 0.0, 0.79],
            ],
            dtype=torch.float32,
        )
        self.zero_link_quat_local = torch.zeros(6, 4)
        self.zero_link_quat_local[:, 0] = 1.0
        self.zero_link_lin_vel = torch.zeros(6, 3)
        self.zero_link_ang_vel = torch.zeros(6, 3)

        self.name_to_idx: dict[str, int] = {n: i for i, n in enumerate(self.SKELETON_ORDER)}
        self.motion_indices = [self.name_to_idx[n] for n in self.MOTION_NAMES]
        self.base_idx_6 = self.MOTION_NAMES.index("Hips")
        self.l_hand_idx_6 = self.MOTION_NAMES.index("LeftHand")
        self.r_hand_idx_6 = self.MOTION_NAMES.index("RightHand")
        self.torso_idx_6 = self.MOTION_NAMES.index("Spine1")
        self.l_foot_idx_6 = self.MOTION_NAMES.index("LeftFoot")
        self.r_foot_idx_6 = self.MOTION_NAMES.index("RightFoot")
        self.l_foot_idx_51 = self.name_to_idx["LeftFoot"]
        self.r_foot_idx_51 = self.name_to_idx["RightFoot"]

        self.motion_quat_inv = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(6, 1)
        self.global_yaw_inv = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.global_xy = torch.tensor([0.0, 0.0])
        z_90_inv = quat_from_euler(torch.tensor([0.0, 0.0, -1.0]) * torch.pi / 2.0)
        self.motion_quat_inv[self.base_idx_6] = z_90_inv
        self.motion_quat_inv[self.torso_idx_6] = z_90_inv
        # Manual
        self.g1_shoulder_y = 0.100
        self.g1_arm_length = 0.419 * 0.9
        self.g1_pelvis_shoulder_z = 1.082 - 0.793
        self.g1_pelvis_torso_z = 0.837 - 0.793
        self.g1_pelvis_z = 0.793 * 0.95
        self.foot_contact_thresh = 0.04
        self.foot_offset_x = 0.06
        # Calibrated
        self.aug_shoulder_y = self.g1_shoulder_y * 1.0
        self.aug_arm_length = self.g1_arm_length * 1.0
        self.aug_pelvis_shoulder_z = self.g1_pelvis_shoulder_z * 1.0
        self.aug_pelvis_z = self.g1_pelvis_z * 1.0
        self.foot_ground_z_left = 0.0
        self.foot_ground_z_right = 0.0

        self._calibrated = False

        self.vel_ema_alpha = 0.25

        self.prev_pos6: torch.Tensor
        self.prev_quat6: torch.Tensor
        self.prev_frame_id: int = -1

        self.ema_base_lin_vel = torch.zeros(3)
        self.ema_base_ang_vel = torch.zeros(3)
        self.ema_base_ang_vel_local = torch.zeros(3)
        self.ema_link_lin_vel = torch.zeros(6, 3)
        self.ema_link_ang_vel = torch.zeros(6, 3)

    def close(self) -> None:
        try:
            self.client.shutdown()
        except Exception:
            pass

    def _ema(self, prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        a = self.vel_ema_alpha
        return (1.0 - a) * prev + a * x

    def _parse_frame(
        self, frame: dict[int, list[list[float]]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.zeros((51, 3), dtype=torch.float32)
        quat = torch.zeros((51, 4), dtype=torch.float32)
        quat[:, 0] = 1.0
        for rb_id, (p, q) in frame.items():
            if rb_id not in RIGID_BODY_ID_MAP:
                raise ValueError(f"Unmapped RB ID {rb_id}!! Please check RIGID_BODY_ID_MAP.")
            if RIGID_BODY_ID_MAP[rb_id] not in self.name_to_idx:
                continue
            idx = self.name_to_idx[RIGID_BODY_ID_MAP[rb_id]]
            pos[idx] = torch.tensor(p, dtype=torch.float32)
            quat[idx] = torch.roll(torch.tensor(q, dtype=torch.float32), 1)

        return pos, quat

    def _reorient_quat(self, quat_local: torch.Tensor, idxs: list[int]) -> torch.Tensor:
        quat_local[idxs] = quat_mul(quat_local[idxs], self.motion_quat_inv[idxs])
        return quat_local

    def _apply_yaw_inv(
        self, pos: torch.Tensor, quat: torch.Tensor, yaw_inv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        yaw_inv = yaw_inv.view(1, 4).repeat(pos.shape[0], 1)
        link_pos_global = quat_apply(yaw_inv, pos)
        link_quat_global = quat_mul(yaw_inv, quat)
        return link_pos_global, link_quat_global

    def _localize(self, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        link_pos_global = pos
        link_quat_global = quat
        base_pos = link_pos_global[self.base_idx_6, :]
        base_quat = link_quat_global[self.base_idx_6, :]
        relative_link_pos_global = link_pos_global.clone()
        relative_link_pos_global[:, :2] -= base_pos[:2]
        base_euler = quat_to_euler(base_quat)
        base_euler[0] = 0.0
        base_euler[1] = 0.0
        inv_yaw = quat_from_euler(-base_euler)
        link_pos_local, link_quat_local = self._apply_yaw_inv(
            relative_link_pos_global, link_quat_global, inv_yaw
        )
        return link_pos_local, link_quat_local

    def _calibrate(
        self,
        pos: torch.Tensor,
        quat: torch.Tensor,
        pos51: torch.Tensor,
    ) -> None:
        ### Global
        z_90 = quat_from_euler(torch.tensor([0.0, 0.0, 1.0]) * torch.pi / 2.0)
        base_quat = quat[self.base_idx_6]
        base_euler = quat_to_euler(base_quat)
        base_euler[0] = 0.0
        base_euler[1] = 0.0
        yaw = quat_from_euler(base_euler)
        self.global_yaw_inv = quat_mul(z_90, quat_inv(yaw))
        self.global_xy = pos[self.base_idx_6, :2].clone()
        ### Local
        quat = self._reorient_quat(
            quat, [self.base_idx_6, self.torso_idx_6]
        )  # Z-90 on Pelvis & Torso
        pos, quat = self._localize(pos, quat)
        ee_idxs_6 = [self.l_foot_idx_6, self.r_foot_idx_6, self.l_hand_idx_6, self.r_hand_idx_6]
        self.motion_quat_inv[ee_idxs_6] = quat_inv(quat[ee_idxs_6])
        ### Scale
        left_pos = pos[self.l_hand_idx_6]
        right_pos = pos[self.r_hand_idx_6]
        self.aug_shoulder_y = (left_pos[1].item() - right_pos[1].item()) / 2.0
        self.aug_arm_length = (left_pos[0].item() + right_pos[0].item()) / 2.0
        aug_shoulder_z = (left_pos[2].item() + right_pos[2].item()) / 2.0
        self.aug_pelvis_z = pos[self.base_idx_6, 2].item()
        self.aug_pelvis_shoulder_z = aug_shoulder_z - self.aug_pelvis_z
        self._calibrated = True
        ### Foot
        self.foot_ground_z_left = pos51[self.l_foot_idx_51, 2].item()
        self.foot_ground_z_right = pos51[self.r_foot_idx_51, 2].item()
        print("[optitrack_publisher] Calibration result:")
        print(f"  - Shoulder Y: {self.aug_shoulder_y:.3f}")
        print(f"  - Arm Length: {self.aug_arm_length:.3f}")
        print(f"  - Pelvis Z: {self.aug_pelvis_z:.3f}")
        print(f"  - Pelvis Shoulder Z: {self.aug_pelvis_shoulder_z:.3f}")

    def run(self) -> None:
        print("=" * 80)
        print("[optitrack_publisher] Started")
        print(f"Redis key prefix: {self.key}")
        print(f"OptiTrack server IP: {self.server_ip}")
        print(f"OptiTrack client IP: {self.client_ip}")
        print("=" * 80)

        self.client.run()
        frame = self.client.get_frame()
        self._parse_frame(frame)
        print("[optitrack_publisher] Successfully received data from OptiTrack server.")
        print("[optitrack_publisher] Press any key to calibrate and start publishing...")
        getch()

        try:
            while True:
                frame = self.client.get_frame()
                self.frame_id = self.client.get_frame_number()
                start_time = time.time()

                pos51, quat51 = self._parse_frame(frame)
                pos6 = pos51[self.motion_indices, :]
                quat6 = quat51[self.motion_indices, :]

                if not self._calibrated:
                    self._calibrate(pos6, quat6, pos51)
                    continue

                # Save raw
                if self.save and self.prev_frame_id != self.frame_id:
                    self.save_data["pos6"].append(pos6.detach().cpu())
                    self.save_data["quat6"].append(quat6.detach().cpu())
                    self.save_data["frame_id"].append(self.frame_id)

                # Local re-orientation
                quat6 = self._reorient_quat(quat6, list(range(6)))
                # Global
                pos6[:, :2] = pos6[:, :2] - self.global_xy
                pos6, quat6 = self._apply_yaw_inv(pos6, quat6, self.global_yaw_inv)
                # Arm scaling (use base frame + torso rotation)
                l_hand_pos_local, l_hand_quat_local = calc_local(
                    pos6[self.base_idx_6],
                    quat6[self.torso_idx_6],
                    pos6[self.l_hand_idx_6],
                    quat6[self.l_hand_idx_6],
                )
                r_hand_pos_local, r_hand_quat_local = calc_local(
                    pos6[self.base_idx_6],
                    quat6[self.torso_idx_6],
                    pos6[self.r_hand_idx_6],
                    quat6[self.r_hand_idx_6],
                )
                l_aug_anchor = torch.tensor(
                    [0.0, self.aug_shoulder_y, self.aug_pelvis_shoulder_z],
                    dtype=torch.float32,
                )
                l_g1_anchor = torch.tensor(
                    [0.0, self.g1_shoulder_y, self.g1_pelvis_shoulder_z],
                    dtype=torch.float32,
                )
                r_aug_anchor = torch.tensor(
                    [0.0, -self.aug_shoulder_y, self.aug_pelvis_shoulder_z],
                    dtype=torch.float32,
                )
                r_g1_anchor = torch.tensor(
                    [0.0, -self.g1_shoulder_y, self.g1_pelvis_shoulder_z],
                    dtype=torch.float32,
                )
                l_hand_pos_local = l_g1_anchor + (l_hand_pos_local - l_aug_anchor) * (
                    self.g1_arm_length / self.aug_arm_length
                )
                r_hand_pos_local = r_g1_anchor + (r_hand_pos_local - r_aug_anchor) * (
                    self.g1_arm_length / self.aug_arm_length
                )
                # Leg scaling
                l_foot_pos_local, l_foot_quat_local = calc_local(
                    pos6[self.base_idx_6],
                    quat6[self.base_idx_6],
                    pos6[self.l_foot_idx_6],
                    quat6[self.l_foot_idx_6],
                )
                r_foot_pos_local, r_foot_quat_local = calc_local(
                    pos6[self.base_idx_6],
                    quat6[self.base_idx_6],
                    pos6[self.r_foot_idx_6],
                    quat6[self.r_foot_idx_6],
                )
                l_foot_pos_local = l_foot_pos_local * (self.g1_pelvis_z / self.aug_pelvis_z)
                r_foot_pos_local = r_foot_pos_local * (self.g1_pelvis_z / self.aug_pelvis_z)
                # Base scaling
                pos6[self.base_idx_6] = pos6[self.base_idx_6] * (
                    self.g1_pelvis_z / self.aug_pelvis_z
                )
                # Update back
                pos6[self.l_foot_idx_6], quat6[self.l_foot_idx_6] = calc_global(
                    pos6[self.base_idx_6],
                    quat6[self.base_idx_6],
                    l_foot_pos_local,
                    l_foot_quat_local,
                )
                pos6[self.r_foot_idx_6], quat6[self.r_foot_idx_6] = calc_global(
                    pos6[self.base_idx_6],
                    quat6[self.base_idx_6],
                    r_foot_pos_local,
                    r_foot_quat_local,
                )
                pos6[self.l_foot_idx_6] = pos6[self.l_foot_idx_6] + quat_apply(
                    quat6[self.l_foot_idx_6],
                    torch.tensor([self.foot_offset_x, 0.0, 0.0]),
                )
                pos6[self.r_foot_idx_6] = pos6[self.r_foot_idx_6] + quat_apply(
                    quat6[self.r_foot_idx_6],
                    torch.tensor([self.foot_offset_x, 0.0, 0.0]),
                )
                pos6[self.torso_idx_6], _ = calc_global(  # Quat kept original
                    pos6[self.base_idx_6],
                    quat6[self.base_idx_6],
                    torch.tensor(
                        [0.0, 0.0, self.g1_pelvis_torso_z],
                        dtype=torch.float32,
                    ),
                    quat_from_euler(torch.tensor([0.0, 0.0, 0.0])),
                )
                pos6[self.l_hand_idx_6], quat6[self.l_hand_idx_6] = calc_global(
                    pos6[self.base_idx_6],
                    quat6[self.torso_idx_6],
                    l_hand_pos_local,
                    l_hand_quat_local,
                )
                pos6[self.r_hand_idx_6], quat6[self.r_hand_idx_6] = calc_global(
                    pos6[self.base_idx_6],
                    quat6[self.torso_idx_6],
                    r_hand_pos_local,
                    r_hand_quat_local,
                )

                # Localize
                pos6_local, quat6_local = self._localize(pos6, quat6)

                # Foot contact
                lz = pos51[self.l_foot_idx_51, 2].item()
                rz = pos51[self.r_foot_idx_51, 2].item()
                l_contact = 1.0 - min(
                    max((lz - self.foot_ground_z_left) / self.foot_contact_thresh, 0.0), 1.0
                )
                r_contact = 1.0 - min(
                    max((rz - self.foot_ground_z_right) / self.foot_contact_thresh, 0.0), 1.0
                )
                foot_contact = torch.tensor([l_contact, r_contact], dtype=torch.float32)

                if self.prev_frame_id != -1:
                    df = self.frame_id - self.prev_frame_id
                    if df > 0:
                        dt = df / self.frame_rate
                        link_lin_vel_raw = (pos6 - self.prev_pos6) / dt
                        base_lin_vel_raw = link_lin_vel_raw[self.base_idx_6]
                        q_delta = quat_diff(quat6, self.prev_quat6)
                        axis_angle = quat_to_angle_axis(q_delta)
                        link_ang_vel_raw = axis_angle / dt
                        base_ang_vel_raw = link_ang_vel_raw[self.base_idx_6]

                        base_q = quat6[self.base_idx_6]
                        base_ang_vel_local_raw = quat_apply(quat_inv(base_q), base_ang_vel_raw)

                        self.ema_link_lin_vel = self._ema(self.ema_link_lin_vel, link_lin_vel_raw)
                        self.ema_base_lin_vel = self._ema(self.ema_base_lin_vel, base_lin_vel_raw)
                        self.ema_link_ang_vel = self._ema(self.ema_link_ang_vel, link_ang_vel_raw)
                        self.ema_base_ang_vel = self._ema(self.ema_base_ang_vel, base_ang_vel_raw)
                        self.ema_base_ang_vel_local = self._ema(
                            self.ema_base_ang_vel_local, base_ang_vel_local_raw
                        )

                self.prev_pos6 = pos6.detach().clone()
                self.prev_quat6 = quat6.detach().clone()
                self.prev_frame_id = self.frame_id

                # Publish
                def rset(key: str, value: torch.Tensor) -> None:
                    self.r.set(f"{self.key}:motion:{key}", json.dumps(_to_list(value)))
                    self.r.set(f"{self.key}:timestamp:{key}", self.frame_id)

                rset("base_pos", pos6[self.base_idx_6])
                rset("base_quat", quat6[self.base_idx_6])
                rset("link_pos_local", pos6_local)
                rset("link_quat_local", quat6_local)
                rset("base_lin_vel", self.ema_base_lin_vel)
                rset("base_ang_vel", self.ema_base_ang_vel)
                rset("base_ang_vel_local", self.ema_base_ang_vel_local)
                rset("link_lin_vel", self.ema_link_lin_vel)
                rset("link_ang_vel", self.ema_link_ang_vel)
                rset("foot_contact", foot_contact)

                curr_time = time.time()
                if curr_time - start_time < 1.0 / self.freq_hz:
                    time.sleep(max(0.0, 1.0 / self.freq_hz - (curr_time - start_time)))

        except KeyboardInterrupt:
            print("\n[optitrack_publisher] Stopped by user.")
        finally:
            if self.save and len(self.save_data["frame_id"]) > 0:
                filename = (
                    f"optitrack_{self.save_data['frame_id'][0]}_{self.save_data['frame_id'][-1]}"
                )
                folder = "deploy/logs/recordings"
                import os

                self.save_data["pos6"] = torch.stack(self.save_data["pos6"], dim=0)
                self.save_data["quat6"] = torch.stack(self.save_data["quat6"], dim=0)
                self.save_data["frame_id"] = torch.tensor(
                    self.save_data["frame_id"], dtype=torch.int64
                )
                os.makedirs(folder, exist_ok=True)
                filename = os.path.join(folder, filename)
                with open(f"{filename}.pkl", "wb") as f:
                    pickle.dump(self.save_data, f)
                print(f"[optitrack_publisher] Saved recording to {filename}.pkl")
                self.save_data = {
                    "fps": 120,
                    "pos6": [],
                    "quat6": [],
                    "frame_id": [],
                }
            self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--key", type=str, default="optitrack:latest")
    parser.add_argument("--server_ip", type=str, default="0.0.0.0")
    parser.add_argument("--client_ip", type=str, default="0.0.0.0")
    parser.add_argument("--use_multicast", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    args = parser.parse_args()

    OptitrackPublisher(
        redis_url=args.redis_url,
        key=args.key,
        server_ip=args.server_ip,
        client_ip=args.client_ip,
        use_multicast=args.use_multicast,
        save=args.save,
        freq_hz=120.0,
    ).run()
