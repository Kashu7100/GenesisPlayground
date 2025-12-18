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
        - {key}:global:h:pos        [3]
        - {key}:global:h:quat       [4]
        - {key}:global:l:pos        [3]
        - {key}:global:l:quat       [4]
        - {key}:global:l:buttons    [1]
        - {key}:global:r:buttons    [1]
        - {key}:global:r:pos        [3]
        - {key}:global:r:quat       [4]
        - {key}:global:recvtime     [1]

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

        self.h_pos: torch.Tensor = torch.zeros(3)
        self.h_quat: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.l_pos: torch.Tensor = torch.zeros(3)
        self.l_quat: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.r_pos: torch.Tensor = torch.zeros(3)
        self.r_quat: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.l_buttons: int = 0
        self.r_buttons: int = 0

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

        self.l_wrist_quat_inv = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.r_wrist_quat_inv = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.g1_shoulder_y = 0.100
        self.g1_arm_length = 0.378
        self.g1_shoulder_z = 1.082
        self.aug_shoulder_y = self.g1_shoulder_y * 1.0
        self.aug_arm_length = self.g1_arm_length * 1.0
        self.aug_shoulder_z = self.g1_shoulder_z * 1.0

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

    def _localize(
        self, ctrl_pos: torch.Tensor, ctrl_quat: torch.Tensor, inv_head_yaw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rel = ctrl_pos.clone()
        assert self.h_pos is not None
        rel[0:2] -= self.h_pos[0:2]
        pos_local = quat_apply(inv_head_yaw[None, :], rel[None, :])[0]
        quat_local = quat_mul(inv_head_yaw[None, :], ctrl_quat[None, :])[0]
        return pos_local, quat_local

    def _calibrate(
        self,
        left_pos_local: torch.Tensor,
        left_quat_local: torch.Tensor,
        right_pos_local: torch.Tensor,
        right_quat_local: torch.Tensor,
    ) -> None:
        self.l_wrist_quat_inv = quat_inv(left_quat_local)
        self.r_wrist_quat_inv = quat_inv(right_quat_local)
        self.aug_shoulder_y = (left_pos_local[1].item() - right_pos_local[1].item()) / 2.0
        self.aug_arm_length = (left_pos_local[0].item() + right_pos_local[0].item()) / 2.0
        self.aug_shoulder_z = (left_pos_local[2].item() + right_pos_local[2].item()) / 2.0

    def _rescale(
        self,
        pos_local: torch.Tensor,
        quat_local: torch.Tensor,
        ys: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ys = 1 for left (y+), -1 for right (y-)
        wrist_quat_inv = self.l_wrist_quat_inv if ys > 0 else self.r_wrist_quat_inv
        quat_local_rescaled = quat_mul(quat_local, wrist_quat_inv)
        pos_local_rescaled = pos_local.clone()
        pos_local_rescaled[0] = pos_local_rescaled[0] * (self.g1_arm_length / self.aug_arm_length)
        pos_local_rescaled[1] = (pos_local_rescaled[1] - ys * self.aug_shoulder_y) * (
            self.g1_arm_length / self.aug_arm_length
        ) + ys * self.g1_shoulder_y
        pos_local_rescaled[2] = pos_local_rescaled[2] * (self.g1_shoulder_z / self.aug_shoulder_z)
        return pos_local_rescaled, quat_local_rescaled

    def _on_button(self, button: str) -> bool:
        lb_map = {
            "LX": 1 << 0,
            "LY": 1 << 1,
            "LTrigger": 1 << 2,
            "LGrip": 1 << 3,
            "LClick": 1 << 4,
        }
        rb_map = {
            "RA": 1 << 0,
            "RB": 1 << 1,
            "RTrigger": 1 << 2,
            "RGrip": 1 << 3,
            "RClick": 1 << 4,
        }
        if button in lb_map:
            return (self.l_buttons & lb_map[button]) != 0
        elif button in rb_map:
            return (self.r_buttons & rb_map[button]) != 0
        else:
            return False

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
                self.h_pos, self.h_quat = self._convert_to_target(data.h_pos, data.h_quat)
                self.l_pos, self.l_quat = self._convert_to_target(data.l_pos, data.l_quat)
                self.r_pos, self.r_quat = self._convert_to_target(data.r_pos, data.r_quat)
                self.l_buttons = data.l_buttons
                self.r_buttons = data.r_buttons

                # Publish global info
                self._publish_global("h", self.h_pos, self.h_quat)
                self._publish_global("l", self.l_pos, self.l_quat)
                self._publish_global("r", self.r_pos, self.r_quat)
                self.r.set(f"{self.key}:global:l:buttons", self.l_buttons)
                self.r.set(f"{self.key}:global:r:buttons", self.r_buttons)
                self.r.set(f"{self.key}:global:recvtime", data.recv_time)

                # Compute local pose
                head_yaw_q = self._head_yaw(self.h_quat)
                inv_head_yaw = quat_inv(head_yaw_q[None, :])[0]
                lp, lq = self._localize(self.l_pos, self.l_quat, inv_head_yaw)
                rp, rq = self._localize(self.r_pos, self.r_quat, inv_head_yaw)

                if self._on_button("RB") and self._on_button("RTrigger"):
                    self._calibrate(lp, lq, rp, rq)

                lp, lq = self._rescale(lp, lq, 1)
                rp, rq = self._rescale(rp, rq, -1)

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
