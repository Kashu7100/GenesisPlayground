import argparse
import json
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
from gs_env.real.oculus.OculusClient import OculusClient


def _to_list(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().flatten().tolist()


class OculusPublisher:
    """
    Receives UDP packets forever and publishes to Redis.

    Redis keys:
        Global Z-up:
        - {key}:global:hmd:pos        [3]
        - {key}:global:hmd:quat       [4]
        - {key}:global:left:pos       [3]
        - {key}:global:left:quat      [4]
        - {key}:global:right:pos      [3]
        - {key}:global:right:quat     [4]
        - {key}:global:recvtime       [1]

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

        self.client = OculusClient(udp_host=udp_host, udp_port=udp_port)

        self.key = key
        self.freq_hz = freq_hz

        # x_t = z_o, y_t = -x_o, z_t = y_o
        # Oculus uses left-handed Y-up coordinate system
        # Target uses right-handed Z-up coordinate system
        self.A = torch.tensor(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.AT = self.A.t()

        self.hmd_pos: torch.Tensor = torch.zeros(3)
        self.hmd_quat: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.left_pos: torch.Tensor = torch.zeros(3)
        self.left_quat: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.right_pos: torch.Tensor = torch.zeros(3)
        self.right_quat: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0])

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

    def _convert_to_target(
        self, pos_o: torch.Tensor, quat_o_xyzw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        pos_t = A * pos_o
        R_t   = A * R_o * A^T   (A is orthogonal, even though det=-1)
        quat_t = rotmat_to_quat(R_t)
        """
        quat_o_wxyz = quat_o_xyzw[[3, 0, 1, 2]]
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

    def _publish_global(self, label: str, pos: torch.Tensor, quat: torch.Tensor) -> None:
        self.r.set(f"{self.key}:global:{label}:pos", json.dumps(_to_list(pos)))
        self.r.set(f"{self.key}:global:{label}:quat", json.dumps(_to_list(quat)))

    def _publish_motion_refs(
        self,
        left_pos_local: torch.Tensor,
        left_quat_local: torch.Tensor,
        right_pos_local: torch.Tensor,
        right_quat_local: torch.Tensor,
        frame_id: int,
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
        self.r.set(f"{self.key}:timestamp:link_pos_local", frame_id)
        self.r.set(f"{self.key}:timestamp:link_quat_local", frame_id)
        self.r.set(f"{self.key}:timestamp:link_lin_vel", frame_id)
        self.r.set(f"{self.key}:timestamp:link_ang_vel", frame_id)
        # print(f"[oculus_publisher] Published motion refs for frame {frame_id}")

    def run(self) -> None:
        print("=" * 80)
        print("[oculus_publisher] Started")
        print(f"UDP bind: {self.client.udp_host}:{self.client.udp_port}")
        print(f"Redis key prefix: {self.key}")
        print("=" * 80)

        self.client.start()

        try:
            while True:
                data = self.client.get_frame()
                start_time = time.time()

                # Frame conversion
                self.hmd_pos, self.hmd_quat = self._convert_to_target(data.hmd_pos, data.hmd_quat)
                self.left_pos, self.left_quat = self._convert_to_target(
                    data.left_pos, data.left_quat
                )
                self.right_pos, self.right_quat = self._convert_to_target(
                    data.right_pos, data.right_quat
                )

                # Publish global pose
                self._publish_global("hmd", self.hmd_pos, self.hmd_quat)
                self._publish_global("left", self.left_pos, self.left_quat)
                self._publish_global("right", self.right_pos, self.right_quat)
                self.r.set(f"{self.key}:global:recvtime", data.recv_time)

                # Publish motion references if HMD + both controllers
                head_yaw_q = self._head_yaw(self.hmd_quat)
                inv_head_yaw = quat_inv(head_yaw_q[None, :])[0]

                def _localize(
                    ctrl_pos: torch.Tensor, ctrl_quat: torch.Tensor, inv_head_yaw: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    rel = ctrl_pos.clone()
                    assert self.hmd_pos is not None
                    rel[0:2] -= self.hmd_pos[0:2]
                    pos_local = quat_apply(inv_head_yaw[None, :], rel[None, :])[0]
                    quat_local = quat_mul(inv_head_yaw[None, :], ctrl_quat[None, :])[0]
                    return pos_local, quat_local

                lp, lq = _localize(self.left_pos, self.left_quat, inv_head_yaw)
                rp, rq = _localize(self.right_pos, self.right_quat, inv_head_yaw)
                self._publish_motion_refs(lp, lq, rp, rq, data.frame_id)

                curr_time = time.time()
                if curr_time - start_time >= 1.0 / self.freq_hz:
                    time.sleep(max(0.0, 1.0 / self.freq_hz - (curr_time - start_time)))

        except KeyboardInterrupt:
            print("\n[oculus_publisher] Stopped by user.")
        finally:
            self.client.close()


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
