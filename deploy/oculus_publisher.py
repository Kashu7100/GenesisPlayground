import argparse
import json
import socket
import time

import redis


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


def _parse_packet(msg: str) -> tuple[str, list[float], list[float]] | None:
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

    pos = [px, py, pz]
    quat_wxyz = [qw, qx, qy, qz]
    return label, pos, quat_wxyz


def publish_oculus_poses(
    redis_url: str = "redis://localhost:6379/0",
    key: str = "oculus:",
    udp_host: str = "0.0.0.0",
    udp_port: int = 5005,
) -> None:
    """
    Receives UDP packets forever and publishes to Redis.

    Redis keys:
      - {key}:hmd:pos        [3]
      - {key}:hmd:quat_wxyz  [4]
      - {key}:hmd:recv_time  float (seconds since epoch)
      - {key}:left:pos
      - {key}:left:quat_wxyz
      - {key}:left:recv_time
      - {key}:right:pos
      - {key}:right:quat_wxyz
      - {key}:right:recv_time
    """
    r = redis.from_url(redis_url)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_host, udp_port))

    print("=" * 80)
    print("[oculus_pose_publisher] Started")
    print(f"UDP bind: {udp_host}:{udp_port}")
    print(f"Redis: {redis_url}")
    print(f"Key: {key}")
    print("=" * 80)

    last_seen: dict[str, float] = {"hmd": 0.0, "left": 0.0, "right": 0.0}

    try:
        while True:
            data, addr = sock.recvfrom(4096)
            recv_time = time.time()

            try:
                msg = data.decode("utf-8", errors="ignore")
            except Exception:
                continue

            parsed = _parse_packet(msg)
            if parsed is None:
                continue

            label, pos, quat_wxyz = parsed
            last_seen[label] = recv_time

            r.set(f"{key}:{label}:pos", json.dumps(pos))
            r.set(f"{key}:{label}:quat_wxyz", json.dumps(quat_wxyz))
            r.set(f"{key}:{label}:recv_time", recv_time)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--key", type=str, default="oculus:")
    parser.add_argument("--udp_host", type=str, default="0.0.0.0")
    parser.add_argument("--udp_port", type=int, default=5005)
    args = parser.parse_args()

    publish_oculus_poses(
        redis_url=args.redis_url,
        key=args.key,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
    )
