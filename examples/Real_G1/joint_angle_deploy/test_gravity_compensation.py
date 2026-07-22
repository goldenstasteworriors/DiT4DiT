"""Enable measured-pose gravity compensation so a suspended G1's arms can be moved by hand."""

from __future__ import annotations

import argparse
import signal
import sys
import termios
import time
import tty

from g1_joint_client import EStop, G1DDS, _read_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", default="enp7s0")
    parser.add_argument("--gravity-scale", type=float, default=0.3)
    parser.add_argument(
        "--gravity-hand-model", choices=("rh56dftp", "rubber"), default="rh56dftp"
    )
    parser.add_argument("--arm-damping", type=float, default=1.5, help="arm Kd in gravity-only mode")
    parser.add_argument("--lowstate-timeout", type=float, default=0.2)
    parser.add_argument("--shutdown-damping-duration", type=float, default=3.0)
    args = parser.parse_args()
    if not sys.stdin.isatty():
        raise SystemExit("必须在交互终端运行，确保 SPACE/Q 急停可用")
    if not 0.0 <= args.gravity_scale <= 1.0:
        raise SystemExit("--gravity-scale must be in [0, 1]")
    if args.arm_damping < 0.0 or args.lowstate_timeout <= 0.0 or args.shutdown_damping_duration <= 0.0:
        raise SystemExit("damping and watchdog parameters are invalid")

    robot = G1DDS(
        args.network_interface,
        lower_body_mode="damping",
        gravity_compensation=True,
        gravity_scale=args.gravity_scale,
        gravity_hand_model=args.gravity_hand_model,
    )
    estop = EStop()
    signal.signal(signal.SIGINT, lambda *_: estop.trigger("Ctrl-C"))
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    enabled = False
    print(
        "机器人必须由可靠吊架承重，双手先由操作人员托住。"
        "按 ENTER 启用纯重力补偿；SPACE/Q/Ctrl-C 切换阻尼并退出。"
    )
    try:
        while not estop.latched:
            key = _read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard")
                break
            measured = robot.state(args.lowstate_timeout)
            if key in ("\n", "\r") and not enabled:
                robot.enter_low_level(measured)
                robot.enter_gravity_only(args.arm_damping)
                enabled = True
                print(
                    f"[ARMED] RH56DFTP gravity-only scale={args.gravity_scale:.3f}；"
                    "现在可缓慢移动双臂比较用力程度"
                )
            time.sleep(0.01)
    except Exception as exc:
        estop.trigger(str(exc))
    finally:
        try:
            if enabled:
                robot.enter_emergency_damping()
                print(
                    f"[STOP] 重力前馈已清零，继续发送阻尼命令 "
                    f"{args.shutdown_damping_duration:.1f}s；请托住双臂"
                )
                deadline = time.monotonic() + args.shutdown_damping_duration
                while time.monotonic() < deadline:
                    time.sleep(0.05)
        finally:
            robot.stop_low_level_publisher()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
    print(f"[EXIT] {estop.reason or 'normal exit'}")


if __name__ == "__main__":
    main()
