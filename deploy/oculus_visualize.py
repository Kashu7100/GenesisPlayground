import socket
import time

import matplotlib.pyplot as plt

UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.setblocking(False)  # non-blocking

print(f"Listening on UDP port {UDP_PORT}...")

hmd_x = []
hmd_z = []
MAX_POINTS = 100
REDRAW_INTERVAL = 0.05
last_redraw = time.time()

plt.ion()
fig, ax = plt.subplots()
(line,) = ax.plot([], [], marker="o", linestyle="-")

ax.set_title("Trajectory (top view: x vs z)")
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.grid(True)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

try:
    while True:
        while True:
            try:
                data, addr = sock.recvfrom(1024)
            except BlockingIOError:
                break
            except Exception as e:
                print("[Python] Socket error:", e)
                break

            msg = data.decode("utf-8").strip()
            parts = msg.split(",")
            if len(parts) != 8:
                continue

            label, x, y, z, qx, qy, qz, qw = parts
            if label != "Right":
                continue

            x = float(x)
            z = float(z)

            hmd_x.append(x)
            hmd_z.append(z)
            if len(hmd_x) > MAX_POINTS:
                hmd_x = hmd_x[-MAX_POINTS:]
                hmd_z = hmd_z[-MAX_POINTS:]

        now = time.time()
        if now - last_redraw >= REDRAW_INTERVAL and hmd_x:
            last_redraw = now
            line.set_data(hmd_x, hmd_z)

            # margin = 0.1
            # xmin = min(hmd_x) - margin
            # xmax = max(hmd_x) + margin
            # zmin = min(hmd_z) - margin
            # zmax = max(hmd_z) + margin
            # ax.set_xlim(xmin, xmax)
            # ax.set_ylim(zmin, zmax)

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(0.001)

except KeyboardInterrupt:
    print("\n[Python] Stopped by user.")

finally:
    sock.close()
    plt.ioff()
    plt.show()
