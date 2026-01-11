import argparse
import json
import sys
import termios
import time
import tty
from pathlib import Path

import redis
import torch
from gs_env.real.steamvr.SteamVRClient import SteamVRClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from deploy.utils import G1Retargeter


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


class SteamVRReceiver(SteamVRClient):
    pass


class RedisMotionPublisher:
    """
    Orchestrates OptiTrack receiving + retargeting, and publishes results into Redis.

    Redis keys:
        - {key}:motion:base_pos [3]
        - {key}:motion:base_quat [4] (w, x, y, z)
        - {key}:motion:base_lin_vel [3]
        - {key}:motion:base_ang_vel [3]
        - {key}:motion:base_ang_vel_local [3]
        - {key}:motion:link_pos_local [N*3]
        - {key}:motion:link_quat_local [N*4]
        - {key}:motion:link_lin_vel [N*3]
        - {key}:motion:link_ang_vel [N*3]
        - {key}:motion:foot_contact [F]
        - {key}:timestamp:* [1]
    """

    def __init__(
        self,
        redis_url: str,
        key_prefix: str,
        udp_host: str,
        udp_port: int,
        freq_hz: float,
        save: bool,
        save_dir: str,
    ) -> None:
        self._redis = redis.from_url(redis_url)
        self.key_prefix = key_prefix
        self.freq_hz = freq_hz
        self.save = save
        self.save_dir = save_dir

        self.receiver = SteamVRReceiver(
            udp_host=udp_host,
            udp_port=udp_port,
        )
        self.retargeter = G1Retargeter()

        self.save_data = {
            "fps": int(self.freq_hz),
            "pos": [],
            "quat": [],
            "frame_id": [],
            "foot_contact": [],
        }

        # Foot contact calibration/state (computed from raw 51-link positions)
        self._foot_contact_height_thresh = 0.04
        self._foot_contact_velocity_thresh = 0.5
        self._foot_initial_height = None
        self._foot_indices = [0, 1]
        self._foot_last_pos = torch.zeros((2, 3), dtype=torch.float32)

    def publish(self, key: str, value: torch.Tensor, frame_id: int) -> None:
        self._redis.set(f"{self.key_prefix}:motion:{key}", json.dumps(_to_list(value)))
        self._redis.set(f"{self.key_prefix}:timestamp:{key}", frame_id)

    def _get_foot_contact(self, all_link_pos: torch.Tensor) -> torch.Tensor:
        foot_pos = all_link_pos[self._foot_indices, :]
        if self._foot_initial_height is None:
            self._foot_initial_height = foot_pos[:, 2]
            self._foot_last_pos = foot_pos.clone()
        foot_height = foot_pos[:, 2]
        foot_not_contact_height = (
            (foot_height - self._foot_initial_height) / self._foot_contact_height_thresh
        ).clamp(0.0, 1.0)
        foot_velocity = (foot_pos - self._foot_last_pos) * self.freq_hz
        self._foot_last_pos = foot_pos.clone()
        foot_not_contact_velocity = (
            torch.norm(foot_velocity[..., :2], dim=-1) / self._foot_contact_velocity_thresh
        ).clamp(0.0, 1.0)
        foot_contact = 1 - (foot_not_contact_height + foot_not_contact_velocity).clamp(0.0, 1.0)
        return foot_contact

    def close(self) -> None:
        self.receiver.shutdown()

    def run(self) -> None:
        print("=" * 80)
        print("[steamvr_publisher] Started")
        print(f"Redis key prefix: {self.key_prefix}")
        print(f"UDP host: {self.receiver.udp_host}")
        print(f"UDP port: {self.receiver.udp_port}")
        print("=" * 80)

        self.receiver.start()
        self.receiver.get_links()
        print("[steamvr_publisher] Successfully received data from SteamVR server.")
        print("[steamvr_publisher] Press any key to calibrate and start publishing...")
        getch()

        try:
            next_publish_time = time.time() + 1.0 / self.freq_hz
            while True:
                tracked_pos, tracked_quat, frame_id = self.receiver.get_links()

                foot_contact = self._get_foot_contact(tracked_pos)

                if self.save:
                    self.save_data["pos"].append(tracked_pos.detach().cpu().clone())
                    self.save_data["quat"].append(tracked_quat.detach().cpu().clone())
                    self.save_data["foot_contact"].append(foot_contact.detach().cpu().clone())
                    self.save_data["frame_id"].append(frame_id)

                if not self.retargeter.calibrated:
                    self.retargeter.calibrate(
                        tracked_pos=tracked_pos,
                        tracked_quat=tracked_quat,
                    )
                    continue

                retargeted = self.retargeter.step(
                    tracked_pos=tracked_pos,
                    tracked_quat=tracked_quat,
                    frame_id=frame_id,
                )
                retargeted["foot_contact"] = foot_contact
                for k, v in retargeted.items():
                    self.publish(k, v, frame_id)

                if time.time() >= next_publish_time:
                    # print("SteamVR is lagging behind")
                    next_publish_time = time.time() + 1.0 / self.freq_hz
                    continue
                time.sleep(max(0.0, next_publish_time - time.time()))
                next_publish_time += 1.0 / self.freq_hz
        except KeyboardInterrupt:
            print("\n[steamvr_publisher] Stopped by user.")
        finally:
            if self.save:
                filename = (
                    f"steamvr_{self.save_data['frame_id'][0]}_{self.save_data['frame_id'][-1]}"
                )
                self.save_data["pos"] = torch.stack(self.save_data["pos"], dim=0)
                self.save_data["quat"] = torch.stack(self.save_data["quat"], dim=0)
                self.save_data["foot_contact"] = torch.stack(self.save_data["foot_contact"], dim=0)
                self.save_data["frame_id"] = torch.tensor(
                    self.save_data["frame_id"], dtype=torch.int64
                )
                import os
                import pickle

                os.makedirs(self.save_dir, exist_ok=True)
                with open(os.path.join(self.save_dir, filename + ".pkl"), "wb") as f:
                    pickle.dump(self.save_data, f)
                print(f"Saved data to {os.path.join(self.save_dir, filename + '.pkl')}")
            self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--key", type=str, default="steamvr:latest")
    parser.add_argument("--udp_host", type=str, default="0.0.0.0")
    parser.add_argument("--udp_port", type=int, default=5005)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="assets/steamvr")
    args = parser.parse_args()

    RedisMotionPublisher(
        redis_url=args.redis_url,
        key_prefix=args.key,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        freq_hz=60.0,
        save=args.save,
        save_dir=args.save_dir,
    ).run()
