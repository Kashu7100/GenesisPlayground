import argparse
import json
import socket
import time

import redis
import torch
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_from_euler,
    quat_inv,
    quat_mul,
    quat_to_euler,
    quat_to_rotmat,
    rotmat_to_quat,
)


def _norm_label(label: str) -> str | None:
    """
    Normalize label strings to: 'hmd', 'left', 'right'.
    Returns None if label is unknown.
    """
    s = label.strip().lower()
    if s in ("hmd",):
        return "hmd"
    if s in ("left",):
        return "left"
    if s in ("right",):
        return "right"
    return None


def _parse_packet(msg: str) -> tuple[str, torch.Tensor, torch.Tensor] | None:
    """
    Expected message format (same as oculus_visualize.py):
      label,x,y,z,qx,qy,qz,qw
    Incoming quat is xyzw. We will convert to wxyz for Redis.
    """
    parts = msg.strip().split(",")
    if len(parts) != 8:
        return None

    label_raw, x, y, z, qx, qy, qz, qw = parts
    label = _norm_label(label_raw)
    if label is None:
        return None

    try:
        px, py, pz = float(x), float(y), float(z)
        qx, qy, qz, qw = float(qx), float(qy), float(qz), float(qw)
    except ValueError:
        return None

    pos = torch.tensor([px, py, pz], dtype=torch.float32)
    quat_wxyz = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
    return label, pos, quat_wxyz


def _to_list(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().flatten().tolist()


class OculusPublisher:
    """
    Receives UDP packets forever and publishes to Redis.

    Redis keys:
        Global Z-up:
        - {key}:global:hmd:pos        [3]
        - {key}:global:hmd:quat_wxyz  [4]
        - {key}:global:hmd:recv_time  float
        - {key}:global:left:pos
        - {key}:global:left:quat_wxyz
        - {key}:global:left:recv_time
        - {key}:global:right:pos
        - {key}:global:right:quat_wxyz
        - {key}:global:right:recv_time

        Motion References:
        - {key}:motion:link_pos_local       [N*3]
        - {key}:motion:link_quat_local      [N*4]
        - {key}:timestamp:link_pos_local    [1]
        - {key}:timestamp:link_quat_local   [1]
    """

    def __init__(
        self, redis_url: str, key: str, udp_host: str, udp_port: int, freq_hz: float
    ) -> None:
        self.r = redis.from_url(redis_url)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((udp_host, udp_port))

        self.key = key
        self.freq_hz = freq_hz
        self.last_publish_time = time.time()

        # x_t = z_o, y_t = -x_o, z_t = y_o
        # Oculus uses left-handed Y-up coordinate system
        # Target uses right-handed Z-up coordinate system
        self.A = torch.tensor(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.AT = self.A.t()

        self.hmd_pos_z: None | torch.Tensor = None
        self.hmd_quat_z: None | torch.Tensor = None
        self.left_pos_z: None | torch.Tensor = None
        self.left_quat_z: None | torch.Tensor = None
        self.right_pos_z: None | torch.Tensor = None
        self.right_quat_z: None | torch.Tensor = None

        self.hmd_recv_time: None | float = None
        self.left_recv_time: None | float = None
        self.right_recv_time: None | float = None

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
        self.timestamp = 0

    def _convert_to_target(
        self, pos_o: torch.Tensor, quat_o_wxyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        pos_t = A * pos_o
        R_t   = A * R_o * A^T   (A is orthogonal, even though det=-1)
        quat_t = rotmat_to_quat(R_t)
        """
        pos_t = self.A @ pos_o

        R_o = quat_to_rotmat(quat_o_wxyz[None, :])[0]
        R_t = self.A @ R_o @ self.AT
        quat_t = rotmat_to_quat(R_t[None, :, :])[0]
        return pos_t, quat_t

    def _head_yaw(self, head_quat_z: torch.Tensor) -> torch.Tensor:
        e = quat_to_euler(head_quat_z[None, :])[0]
        e[0] = 0.0
        e[1] = 0.0
        yaw_q = quat_from_euler(e[None, :])[0]
        return yaw_q

    def _publish_global(
        self, label: str, pos_z: torch.Tensor, quat_z: torch.Tensor, recv_time: float
    ) -> None:
        self.r.set(f"{self.key}:global:{label}:pos_zup", json.dumps(_to_list(pos_z)))
        self.r.set(f"{self.key}:global:{label}:quat_wxyz_zup", json.dumps(_to_list(quat_z)))
        self.r.set(f"{self.key}:global:{label}:recv_time", recv_time)

    def _publish_motion_refs(
        self,
        left_pos_local: torch.Tensor,
        left_quat_local: torch.Tensor,
        right_pos_local: torch.Tensor,
        right_quat_local: torch.Tensor,
    ) -> None:
        link_pos_local = self.zero_link_pos_local.clone()
        link_pos_local[2] = left_pos_local
        link_pos_local[3] = right_pos_local
        link_quat_local = self.zero_link_quat_local.clone()
        link_quat_local[2] = left_quat_local
        link_quat_local[3] = right_quat_local
        link_lin_vel = self.zero_link_lin_vel.clone()
        link_ang_vel = self.zero_link_ang_vel.clone()
        # Publish each field as a separate Redis key
        self.r.set(f"{self.key}:motion:link_pos_local", json.dumps(_to_list(link_pos_local)))
        self.r.set(f"{self.key}:motion:link_quat_local", json.dumps(_to_list(link_quat_local)))
        self.r.set(f"{self.key}:motion:link_lin_vel", json.dumps(_to_list(link_lin_vel)))
        self.r.set(f"{self.key}:motion:link_ang_vel", json.dumps(_to_list(link_ang_vel)))
        self.r.set(f"{self.key}:timestamp:link_pos_local", self.timestamp)
        self.r.set(f"{self.key}:timestamp:link_quat_local", self.timestamp)
        self.r.set(f"{self.key}:timestamp:link_lin_vel", self.timestamp)
        self.r.set(f"{self.key}:timestamp:link_ang_vel", self.timestamp)
        self.timestamp += 1

    def run(self) -> None:
        print("=" * 80)
        print("[oculus_publisher] Started (Z-up + head-relative local controller poses)")
        print(f"UDP bind: {self.sock.getsockname()[0]}:{self.sock.getsockname()[1]}")
        print(f"Redis key prefix: {self.key}")
        print("Incoming packet: label,x,y,z,qx,qy,qz,qw (xyzw incoming; wxyz stored)")
        print("=" * 80)

        try:
            while True:
                # Parse UDP packet
                data, _addr = self.sock.recvfrom(4096)
                recv_time = time.time()
                msg = data.decode("utf-8", errors="ignore")
                parsed = _parse_packet(msg)
                if parsed is None:
                    continue

                # Frame conversion
                label, pos_w, quat_w = parsed
                pos_z, quat_z = self._convert_to_target(pos_w, quat_w)
                if label == "hmd":
                    self.hmd_pos_z, self.hmd_quat_z, self.hmd_recv_time = pos_z, quat_z, recv_time
                elif label == "left":
                    self.left_pos_z, self.left_quat_z, self.left_recv_time = (
                        pos_z,
                        quat_z,
                        recv_time,
                    )
                elif label == "right":
                    self.right_pos_z, self.right_quat_z, self.right_recv_time = (
                        pos_z,
                        quat_z,
                        recv_time,
                    )

                # Publish global pose
                self._publish_global(label, pos_z, quat_z, recv_time)

                # Publish motion references if HMD + both controllers
                if self.hmd_pos_z is None or self.hmd_quat_z is None:
                    continue
                head_yaw_q = self._head_yaw(self.hmd_quat_z)
                inv_head_yaw = quat_inv(head_yaw_q[None, :])[0]

                def _localize(
                    ctrl_pos_z: torch.Tensor, ctrl_quat_z: torch.Tensor, inv_head_yaw: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    rel = ctrl_pos_z.clone()
                    assert self.hmd_pos_z is not None
                    rel[0:2] -= self.hmd_pos_z[0:2]
                    pos_local = quat_apply(inv_head_yaw[None, :], rel[None, :])[0]
                    quat_local = quat_mul(inv_head_yaw[None, :], ctrl_quat_z[None, :])[0]
                    return pos_local, quat_local

                if self.left_recv_time is not None and self.right_recv_time is not None:
                    curr_time = time.time()
                    if curr_time - self.last_publish_time >= 1.0 / self.freq_hz:
                        self.last_publish_time = curr_time
                        assert self.left_pos_z is not None
                        assert self.left_quat_z is not None
                        lp, lq = _localize(self.left_pos_z, self.left_quat_z, inv_head_yaw)
                        assert self.right_pos_z is not None
                        assert self.right_quat_z is not None
                        rp, rq = _localize(self.right_pos_z, self.right_quat_z, inv_head_yaw)
                        self._publish_motion_refs(lp, lq, rp, rq)
                    else:
                        pass

        except KeyboardInterrupt:
            print("\n[oculus_publisher] Stopped by user.")
        finally:
            self.sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--key", type=str, default="oculus:latest")
    parser.add_argument("--udp_host", type=str, default="0.0.0.0")
    parser.add_argument("--udp_port", type=int, default=5005)
    args = parser.parse_args()

    OculusPublisher(
        redis_url=args.redis_url,
        key=args.key,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        freq_hz=50.0,
    ).run()
