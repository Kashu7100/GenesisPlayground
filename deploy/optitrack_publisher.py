import argparse
import json
import time

import redis
import torch
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_from_euler,
    quat_mul,
    quat_to_euler,
)
from gs_env.real.config.registry import EnvArgsRegistry
from gs_env.real.config.schema import OptitrackEnvArgs
from gs_env.real.optitrack.NatNetClient import setup_optitrack
from gs_env.real.optitrack.optitrack_config import RIGID_BODY_ID_MAP, track_id_offset


def _to_list(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().flatten().tolist()


class OptitrackPublisher:
    """
    Publishes OptiTrack skeleton poses into Redis.

    Redis keys:
        Global raw skeleton:
        - {key}:global:pos   [51*3]
        - {key}:global:quat  [51*4]

        Motion:
        - {key}:motion:link_pos_local        [6*3]
        - {key}:motion:link_quat_local       [6*4]
        - {key}:timestamp:link_pos_local     [1]
        - {key}:timestamp:link_quat_local    [1]
    """

    SKELETON_ORDER = [RIGID_BODY_ID_MAP[i + track_id_offset] for i in range(1, 52)]

    MOTION_NAMES = ["LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips", "Spine1"]

    def __init__(
        self,
        redis_url: str,
        key: str,
        server_ip: str,
        client_ip: str,
        use_multicast: bool,
        freq_hz: float,
    ) -> None:
        self.r = redis.from_url(redis_url)

        self.key = key
        self.freq_hz = freq_hz

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

        self.motion_quat_inv = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.g1_shoulder_y = 0.100
        self.g1_arm_length = 0.378
        self.g1_shoulder_z = 1.082
        self.aug_shoulder_y = self.g1_shoulder_y * 1.0
        self.aug_arm_length = self.g1_arm_length * 1.0
        self.aug_shoulder_z = self.g1_shoulder_z * 1.0

    def close(self) -> None:
        try:
            self.client.shutdown()
        except Exception:
            pass

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

    def _publish_global(self, pos: torch.Tensor, quat: torch.Tensor) -> None:
        self.r.set(f"{self.key}:global:pos", json.dumps(_to_list(pos)))
        self.r.set(f"{self.key}:global:quat", json.dumps(_to_list(quat)))

    def _localize(self, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        link_pos_global = pos
        link_quat_global = quat

        # Hips idx 4
        base_pos = link_pos_global[4:5, :]
        base_quat = link_quat_global[4:5, :]

        relative_link_pos_global = link_pos_global.clone()
        relative_link_pos_global[:, :2] -= base_pos[0, :2]
        base_euler = quat_to_euler(base_quat)
        base_euler[:, 0] = 0.0
        base_euler[:, 1] = 0.0
        inv_yaw = quat_from_euler(-base_euler)

        inv_yaw_b = inv_yaw.repeat(link_pos_global.shape[0], 1)
        link_pos_local = quat_apply(inv_yaw_b, relative_link_pos_global)
        link_quat_local = quat_mul(inv_yaw_b, link_quat_global)

        return link_pos_local, link_quat_local

    def _reorient_quat(self, quat_local: torch.Tensor) -> torch.Tensor:
        return quat_mul(quat_local, self.motion_quat_inv)

    def _rescale_all_heights(self, pos_local: torch.Tensor) -> torch.Tensor:
        pos_local[:, 2] = pos_local[:, 2] * (self.g1_shoulder_z / self.aug_shoulder_z)
        return pos_local

    def _rescale_hand_pos(self, hand_pos_local: torch.Tensor, ys: float) -> torch.Tensor:
        p = hand_pos_local.clone()
        p[0] = p[0] * (self.g1_arm_length / self.aug_arm_length)
        p[1] = (p[1] - ys * self.aug_shoulder_y) * (
            self.g1_arm_length / self.aug_arm_length
        ) + ys * self.g1_shoulder_y
        return p

    def _publish_motion_refs(self, pos: torch.Tensor, quat: torch.Tensor, frame_id: int) -> None:
        link_pos_local = pos
        link_quat_local = quat
        link_lin_vel = self.zero_link_lin_vel.clone()
        link_ang_vel = self.zero_link_ang_vel.clone()

        self.r.set(f"{self.key}:motion:link_pos_local", json.dumps(_to_list(link_pos_local)))
        self.r.set(f"{self.key}:motion:link_quat_local", json.dumps(_to_list(link_quat_local)))
        self.r.set(f"{self.key}:motion:link_lin_vel", json.dumps(_to_list(link_lin_vel)))
        self.r.set(f"{self.key}:motion:link_ang_vel", json.dumps(_to_list(link_ang_vel)))
        self.r.set(f"{self.key}:timestamp:link_pos_local", frame_id)
        self.r.set(f"{self.key}:timestamp:link_quat_local", frame_id)
        self.r.set(f"{self.key}:timestamp:link_lin_vel", frame_id)
        self.r.set(f"{self.key}:timestamp:link_ang_vel", frame_id)

    def run(self) -> None:
        print("=" * 80)
        print("[optitrack_publisher] Started")
        print(f"Redis key prefix: {self.key}")
        print(f"OptiTrack server IP: {self.server_ip}")
        print(f"OptiTrack client IP: {self.client_ip}")
        print("=" * 80)

        self.client.run()
        self.client.get_frame()
        print("[optitrack_publisher] Successfully received data from OptiTrack server.")

        try:
            while True:
                frame = self.client.get_frame()
                frame_id = self.client.get_frame_number()
                start_time = time.time()

                pos51, quat51 = self._parse_frame(frame)
                pos6 = pos51[self.motion_indices, :]
                quat6 = quat51[self.motion_indices, :]

                pos6, quat6 = self._localize(pos6, quat6)

                quat6 = self._reorient_quat(quat6)
                pos6 = self._rescale_all_heights(pos6)
                # TODO: hand pose is relative to torso, not pelvis
                pos6[2, :] = self._rescale_hand_pos(pos6[2, :], ys=1.0)
                pos6[3, :] = self._rescale_hand_pos(pos6[3, :], ys=-1.0)

                self._publish_global(pos51, quat51)
                self._publish_motion_refs(pos6, quat6, frame_id)

                curr_time = time.time()
                if curr_time - start_time >= 1.0 / self.freq_hz:
                    time.sleep(max(0.0, 1.0 / self.freq_hz - (curr_time - start_time)))

        except KeyboardInterrupt:
            print("\n[optitrack_publisher] Stopped by user.")
        finally:
            self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--key", type=str, default="optitrack:latest")
    parser.add_argument("--server_ip", type=str, default="0.0.0.0")
    parser.add_argument("--client_ip", type=str, default="0.0.0.0")
    parser.add_argument("--use_multicast", action="store_true", default=False)
    args = parser.parse_args()

    OptitrackPublisher(
        redis_url=args.redis_url,
        key=args.key,
        server_ip=args.server_ip,
        client_ip=args.client_ip,
        use_multicast=args.use_multicast,
        freq_hz=50.0,
    ).run()
