import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class OculusFrame:
    # Left-handed coordinate system, Y-up, quat is xyzw
    frame_id: int
    recv_time: float
    hmd_pos: torch.Tensor
    left_pos: torch.Tensor
    right_pos: torch.Tensor
    hmd_quat: torch.Tensor
    left_quat: torch.Tensor
    right_quat: torch.Tensor


class OculusClient:
    """
    - Format:
        FRAME,<frame_id>,HMD,<8>,LEFT,<8>,RIGHT,<8>
        <8> = px,py,pz,qx,qy,qz,qw
    """

    def __init__(
        self, udp_host: str = "0.0.0.0", udp_port: int = 5005, device: str = "cpu"
    ) -> None:
        self.udp_host = udp_host
        self.udp_port = udp_port

        self._device_str = device

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((udp_host, udp_port))

        self._lock = threading.Lock()
        self._has_frame = threading.Event()
        self._latest_raw: dict[str, Any] = {}

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass

    def get_frame(self) -> OculusFrame:
        self._has_frame.wait()
        with self._lock:
            raw = self._latest_raw

        device = torch.device(self._device_str)

        def to_tensor(x: list[float]) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.float32, device=device)

        return OculusFrame(
            frame_id=raw["frame_id"],
            recv_time=raw["recv_time"],
            hmd_pos=to_tensor(raw["hmd"]["pos"]),
            left_pos=to_tensor(raw["left"]["pos"]),
            right_pos=to_tensor(raw["right"]["pos"]),
            hmd_quat=to_tensor(raw["hmd"]["quat"]),
            left_quat=to_tensor(raw["left"]["quat"]),
            right_quat=to_tensor(raw["right"]["quat"]),
        )

    # ---------------- internal ----------------

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(8192)  # blocking recv
            except OSError:
                break

            recv_time = time.time()
            msg = data.decode("utf-8", errors="ignore").strip()

            parsed = self._parse_frame_packet_to_lists(msg, recv_time)
            if parsed is None:
                continue

            with self._lock:
                self._latest_raw = parsed
            self._has_frame.set()

    def _parse_frame_packet_to_lists(self, msg: str, recv_time: float) -> dict[str, Any] | None:
        # parse frame id
        parts = msg.split(",")
        if len(parts) < 2 or parts[0] != "FRAME":
            return None

        try:
            frame_id = int(parts[1])
        except ValueError:
            return None

        # parse blocks
        hmd = self._parse_block(parts, "HMD")
        left = self._parse_block(parts, "LEFT")
        right = self._parse_block(parts, "RIGHT")
        if hmd is None or left is None or right is None:
            return None

        return {
            "frame_id": frame_id,
            "recv_time": recv_time,
            "hmd": hmd,
            "left": left,
            "right": right,
        }

    def _parse_block(self, parts: list[str], label: str) -> dict[str, list[float]] | None:
        try:
            i = parts.index(label)
        except ValueError:
            return None
        if i + 7 >= len(parts):
            return None
        try:
            px = float(parts[i + 1])
            py = float(parts[i + 2])
            pz = float(parts[i + 3])
            qx = float(parts[i + 4])
            qy = float(parts[i + 5])
            qz = float(parts[i + 6])
            qw = float(parts[i + 7])
        except ValueError:
            return None
        return {"pos": [px, py, pz], "quat": [qx, qy, qz, qw]}
