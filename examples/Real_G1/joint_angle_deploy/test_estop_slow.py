"""Low-speed real-G1 e-stop test; SPACE/Q latches a measured-position hold."""

import argparse
import math
import sys
import termios
import time
import tty

import numpy as np

from g1_joint_client import EStop, G1DDS, RIGHT_ARM_MOTORS, _read_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-interface", required=True)
    parser.add_argument(
        "--lower-body-mode", choices=("damping", "zero-torque"), default="damping"
    )
    parser.add_argument("--joint", type=int, choices=range(7), default=6)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--period", type=float, default=10.0)
    parser.add_argument("--lowstate-warning-age", type=float, default=0.2)
    parser.add_argument("--lowstate-timeout", type=float, default=0.5)
    parser.add_argument("--lowstate-warning-interval", type=float, default=2.0)
    parser.add_argument("--estop-damping-duration", type=float, default=10.0)
    args = parser.parse_args()
    if not sys.stdin.isatty():
        raise SystemExit("必须在交互终端运行")
    if min(
        args.lowstate_warning_age,
        args.lowstate_timeout,
        args.lowstate_warning_interval,
        args.estop_damping_duration,
    ) <= 0.0 or args.lowstate_warning_age >= args.lowstate_timeout:
        raise SystemExit("invalid LowState watchdog or e-stop damping parameters")
    robot, estop = G1DDS(args.network_interface, args.lower_body_mode), EStop()

    def read_robot_state() -> np.ndarray:
        return robot.state(
            args.lowstate_timeout,
            warning_age=args.lowstate_warning_age,
            warning_interval=args.lowstate_warning_interval,
        )

    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    print("低速急停测试：按 ENTER 解锁，SPACE/Q 急停并保持实测位置。")
    enabled, origin, start = False, None, time.monotonic()
    try:
        while not estop.latched:
            key = _read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard test")
                break
            if key in ("\n", "\r"):
                measured = read_robot_state()
                robot.enter_low_level(measured)
                enabled = True
                print("[ARMED] rt/lowcmd 测试运动已启用；机器人必须可靠吊挂")
            measured = read_robot_state()
            origin = measured[RIGHT_ARM_MOTORS].copy() if origin is None else origin
            target = origin.copy()
            target[args.joint] += args.amplitude * math.sin(2 * math.pi * (time.monotonic() - start) / args.period)
            if enabled:
                robot.send_right_arm(target, measured)
            time.sleep(0.01)
    except Exception as exc:
        estop.trigger(str(exc))
    finally:
        try:
            if enabled:
                state_is_stale = "lowstate missing or stale" in estop.reason.lower()
                measured = None
                if not state_is_stale:
                    try:
                        measured = read_robot_state()
                    except RuntimeError:
                        state_is_stale = True
                if state_is_stale:
                    robot.enter_emergency_damping()
                    deadline = time.monotonic() + args.estop_damping_duration
                    while time.monotonic() < deadline:
                        time.sleep(0.05)
                else:
                    for _ in range(20):
                        robot.send_right_arm(measured[RIGHT_ARM_MOTORS], measured)
                        time.sleep(0.01)
        finally:
            robot.stop_low_level_publisher()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)


if __name__ == "__main__":
    main()
