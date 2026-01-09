using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Valve.VR; // OpenVR

namespace SteamVRPoseStreamer
{
    class Program
    {
        // UDP settings
        static readonly string RemoteIp = "255.255.255.255"; // broadcast
        static readonly int RemotePort = 5005;

        // Target send rate
        static readonly int TargetHz = 120;

        static ulong frameId = 0;

        static void Main(string[] args)
        {
            // ---- UDP ----
            using var udp = new UdpClient();
            udp.EnableBroadcast = true;
            var remote = new IPEndPoint(IPAddress.Parse(RemoteIp), RemotePort);

            Console.WriteLine($"[SteamVRPoseStreamer] UDP broadcast to {RemoteIp}:{RemotePort} at ~{TargetHz} Hz");

            // ---- OpenVR Init ----
            EVRInitError initErr = EVRInitError.None;
            var vr = OpenVR.Init(ref initErr, EVRApplicationType.VRApplication_Other);
            if (initErr != EVRInitError.None || vr == null)
            {
                Console.WriteLine($"OpenVR init failed: {initErr}");
                Console.WriteLine("Make sure SteamVR is running and a VR system is available.");
                return;
            }

            Console.WriteLine("[SteamVRPoseStreamer] OpenVR initialized.");

            // ---- Main loop ----
            var poses = new TrackedDevicePose_t[OpenVR.k_unMaxTrackedDeviceCount];
            int sleepMs = Math.Max(1, (int)Math.Round(1000.0 / TargetHz));

            // Cache controller indices; they may change on reconnect, refresh periodically
            uint leftIndex = OpenVR.k_unTrackedDeviceIndexInvalid;
            uint rightIndex = OpenVR.k_unTrackedDeviceIndexInvalid;
            int refreshCounter = 0;

            while (true)
            {
                frameId++;
                refreshCounter++;

                // Occasionally refresh controller indices in case of reconnect
                if (refreshCounter % (TargetHz * 2) == 0) // every ~2 seconds
                {
                    leftIndex = vr.GetTrackedDeviceIndexForControllerRole(ETrackedControllerRole.LeftHand);
                    rightIndex = vr.GetTrackedDeviceIndexForControllerRole(ETrackedControllerRole.RightHand);
                }
                else if (frameId == 1)
                {
                    leftIndex = vr.GetTrackedDeviceIndexForControllerRole(ETrackedControllerRole.LeftHand);
                    rightIndex = vr.GetTrackedDeviceIndexForControllerRole(ETrackedControllerRole.RightHand);
                }

                var system = OpenVR.System;
                if (system == null)
                {
                    Console.WriteLine("OpenVR.System is null.");
                    break;
                }
                system.GetDeviceToAbsoluteTrackingPose(
                    ETrackingUniverseOrigin.TrackingUniverseStanding,
                    0f,
                    poses
                );


                // HMD is always index 0
                uint hmdIndex = OpenVR.k_unTrackedDeviceIndex_Hmd;

                bool okH = TryGetPose(poses, hmdIndex, out var hp, out var hq);
                bool okL = TryGetPose(poses, leftIndex, out var lp, out var lq);
                bool okR = TryGetPose(poses, rightIndex, out var rp, out var rq);

                // bit0 primary, bit1 secondary, bit2 triggerButton, bit3 gripButton, bit4 primary2DAxisClick
                int lb = TryGetButtonMask(vr, leftIndex);
                int rb = TryGetButtonMask(vr, rightIndex);

                if (!okH) { hp = (0, 0, 0); hq = (0, 0, 0, 1); }
                if (!okL) { lp = (0, 0, 0); lq = (0, 0, 0, 1); }
                if (!okR) { rp = (0, 0, 0); rq = (0, 0, 0, 1); }

                // SteamVR/OpenVR uses a right-handed coordinate system.

                string msg = string.Format(
                    System.Globalization.CultureInfo.InvariantCulture,
                    "FRAME,{0},HPOSE,{1},{2},{3},{4},{5},{6},{7},LPOSE,{8},{9},{10},{11},{12},{13},{14},LB,{15},RPOSE,{16},{17},{18},{19},{20},{21},{22},RB,{23}",
                    frameId,
                    hp.x, hp.y, hp.z, hq.x, hq.y, hq.z, hq.w,
                    lp.x, lp.y, lp.z, lq.x, lq.y, lq.z, lq.w, lb,
                    rp.x, rp.y, rp.z, rq.x, rq.y, rq.z, rq.w, rb
                );

                byte[] bytes = Encoding.UTF8.GetBytes(msg);
                udp.Send(bytes, bytes.Length, remote);

                Thread.Sleep(sleepMs);
            }

            OpenVR.Shutdown();
        }

        // ---------------- Pose ----------------

        static bool TryGetPose(TrackedDevicePose_t[] poses, uint deviceIndex,
            out (double x, double y, double z) pos,
            out (double x, double y, double z, double w) quat)
        {
            pos = (0, 0, 0);
            quat = (0, 0, 0, 1);

            if (deviceIndex == OpenVR.k_unTrackedDeviceIndexInvalid) return false;
            if (deviceIndex >= poses.Length) return false;

            var p = poses[deviceIndex];
            if (!p.bPoseIsValid) return false;

            var m = p.mDeviceToAbsoluteTracking; // 3x4 matrix

            // position
            double px = m.m3;
            double py = m.m7;
            double pz = m.m11;

            // rotation (3x3)
            double r00 = m.m0, r01 = m.m1, r02 = m.m2;
            double r10 = m.m4, r11 = m.m5, r12 = m.m6;
            double r20 = m.m8, r21 = m.m9, r22 = m.m10;

            var q = RotMatToQuat(r00, r01, r02, r10, r11, r12, r20, r21, r22);

            pos = (px, py, pz);
            quat = q;
            return true;
        }

        // Numerically stable 3x3 rotation matrix -> quaternion (xyzw)
        static (double x, double y, double z, double w) RotMatToQuat(
            double r00, double r01, double r02,
            double r10, double r11, double r12,
            double r20, double r21, double r22)
        {
            double trace = r00 + r11 + r22;
            double qw, qx, qy, qz;

            if (trace > 0)
            {
                double s = Math.Sqrt(trace + 1.0) * 2.0; // s=4*qw
                qw = 0.25 * s;
                qx = (r21 - r12) / s;
                qy = (r02 - r20) / s;
                qz = (r10 - r01) / s;
            }
            else if ((r00 > r11) && (r00 > r22))
            {
                double s = Math.Sqrt(1.0 + r00 - r11 - r22) * 2.0; // s=4*qx
                qw = (r21 - r12) / s;
                qx = 0.25 * s;
                qy = (r01 + r10) / s;
                qz = (r02 + r20) / s;
            }
            else if (r11 > r22)
            {
                double s = Math.Sqrt(1.0 + r11 - r00 - r22) * 2.0; // s=4*qy
                qw = (r02 - r20) / s;
                qx = (r01 + r10) / s;
                qy = 0.25 * s;
                qz = (r12 + r21) / s;
            }
            else
            {
                double s = Math.Sqrt(1.0 + r22 - r00 - r11) * 2.0; // s=4*qz
                qw = (r10 - r01) / s;
                qx = (r02 + r20) / s;
                qy = (r12 + r21) / s;
                qz = 0.25 * s;
            }

            // normalize
            double norm = Math.Sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
            if (norm > 1e-12)
            {
                qx /= norm; qy /= norm; qz /= norm; qw /= norm;
            }
            return (qx, qy, qz, qw);
        }

        // ---------------- Buttons ----------------

        static int TryGetButtonMask(CVRSystem vr, uint deviceIndex)
        {
            if (vr == null) return 0;
            if (deviceIndex == OpenVR.k_unTrackedDeviceIndexInvalid) return 0;

            VRControllerState_t state = new VRControllerState_t();
            uint stateSize = (uint)System.Runtime.InteropServices.Marshal.SizeOf(typeof(VRControllerState_t));
            bool ok = vr.GetControllerState(deviceIndex, ref state, stateSize);
            if (!ok) return 0;

            int mask = 0;

            // bit0: primaryButton    -> Button.A (right) / Button.X (left)
            // bit1: secondaryButton  -> Button.B (right) / Button.Y (left)
            // bit2: triggerButton    -> interpret as trigger "click" (if present)
            // bit3: gripButton       -> grip "click" (if present)
            // bit4: primary2DAxisClick -> joystick click

            bool primary = IsPressed(state, EVRButtonId.k_EButton_A); // often A/X depending on binding
            bool secondary = IsPressed(state, EVRButtonId.k_EButton_ApplicationMenu); // often B/Y or menu
            bool triggerClick = IsPressed(state, EVRButtonId.k_EButton_SteamVR_Trigger);
            bool gripClick = IsPressed(state, EVRButtonId.k_EButton_Grip);
            bool joystickClick = IsPressed(state, EVRButtonId.k_EButton_Axis0); // Axis0 click in many bindings

            if (primary) mask |= (1 << 0);
            if (secondary) mask |= (1 << 1);
            if (triggerClick) mask |= (1 << 2);
            if (gripClick) mask |= (1 << 3);
            if (joystickClick) mask |= (1 << 4);

            return mask;
        }

        static bool IsPressed(VRControllerState_t state, EVRButtonId button)
        {
            ulong mask = 1UL << (int)button;
            return (state.ulButtonPressed & mask) != 0;
        }
    }
}
