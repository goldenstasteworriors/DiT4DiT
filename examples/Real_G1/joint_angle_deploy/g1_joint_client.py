"""Run the right-arm joint-angle DiT4DiT policy on a real Unitree G1.

The model runs on a remote ZMQ server.  This process must run on the computer
connected to the G1 DDS network.  Publishing is disabled unless --arm is set.
SPACE, Q, Ctrl-C, inference timeout, stale lowstate, or a joint safety violation
latches an e-stop which holds the latest measured right-arm pose.
"""

from __future__ import annotations

import argparse
import select
import signal
import socket
import sys
import termios
import threading
import time
import tty

import cv2
import msgpack
import numpy as np
import zmq

RIGHT_ARM_MOTORS = np.arange(22, 29)
LEFT_ARM_MOTORS = np.arange(15, 22)
LOWER_BODY_MOTORS = np.arange(0, 15)
POS_STOP_F = 2146000000.0
VEL_STOP_F = 16000.0
# SONICMJ g1_29dof_gear_wbc.yaml: soft arm gains and lower-body damping gains.
ARM_KP = np.array([50.0, 50.0, 20.0, 20.0, 10.0, 10.0, 10.0])
ARM_KD = np.array([5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0])
LOWER_BODY_KD = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 5.0, 5.0, 5.0])
RIGHT_ARM_LOWER = np.array([-3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144])
RIGHT_ARM_UPPER = np.array([2.6704, 1.5882, 2.618, 2.0944, 1.9722, 1.6144, 1.6144])
# Exact state stored in training episode 0 (right arm 7 + Inspire right hand 6).
DEFAULT_INITIAL_RIGHT_ARM = np.array(
    [0.01070191, -0.23347668, -0.07287607, -0.58485419, 0.36513537, 0.41992724, -0.25048229]
)
DEFAULT_INITIAL_RIGHT_HAND = np.array([0.99800003, 1.0, 0.99800003, 0.99800003, 0.99900001, 0.98400003])


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise TypeError(type(obj).__name__)


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class PolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{port}")

    def predict(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        data = {"examples": [{"image": [image], "lang": instruction, "state": state[None]}]}
        request = {"endpoint": "get_action", "data": data}
        try:
            self.socket.send(msgpack.packb(request, default=_pack_array))
            response = msgpack.unpackb(self.socket.recv(), object_hook=_unpack_array)
        except zmq.Again as exc:
            raise TimeoutError(
                "ZMQ inference timed out; check A800 server logs, SSH tunnel, and --timeout-ms"
            ) from exc
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error")))
        return np.asarray(response["data"]["unnormalized_actions"], dtype=np.float64)


class G1DDS:
    def __init__(self, network_interface: str, lower_body_mode: str = "damping"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        available_interfaces = {name for _, name in socket.if_nameindex()}
        if network_interface not in available_interfaces:
            available = ", ".join(sorted(available_interfaces))
            raise ValueError(
                f"network interface '{network_interface}' does not exist; available: {available}"
            )
        ChannelFactoryInitialize(0, network_interface)
        if lower_body_mode not in ("damping", "zero-torque"):
            raise ValueError(f"unsupported lower-body mode: {lower_body_mode}")
        self._lower_body_mode = lower_body_mode
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._q = None
        self._mode_machine = 0
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._low_level_enabled = False
        self._left_arm_hold = None
        self._right_arm_target = None
        self._publisher_stop = threading.Event()
        self._publisher_thread = None
        self._motion_switcher = MotionSwitcherClient()
        self._motion_switcher.SetTimeout(5.0)
        self._motion_switcher.Init()
        # Match SONICMJ BodyCommandSender: direct 29-DoF low-level command topic.
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self._hand_cmd = MotorCmds_([unitree_go_msg_dds__MotorCmd_() for _ in range(12)])
        self._hand_publisher = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self._hand_publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_state, 10)
        self._wait_for_first_state(timeout=5.0)

    def _on_state(self, msg):
        with self._lock:
            self._q = np.array([motor.q for motor in msg.motor_state[:29]], dtype=np.float64)
            self._mode_machine = int(msg.mode_machine)
            self._stamp = time.monotonic()

    def _wait_for_first_state(self, timeout: float):
        """Allow CycloneDDS discovery to finish before enabling the runtime watchdog."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._q is not None:
                    return
            time.sleep(0.01)
        raise RuntimeError(
            f"no rt/lowstate received within {timeout:.1f}s; "
            "check G1 power/mode, DDS domain, and robot network interface"
        )

    def state(self, max_age: float) -> np.ndarray:
        with self._lock:
            if self._q is None or time.monotonic() - self._stamp > max_age:
                raise RuntimeError("lowstate missing or stale")
            return self._q.copy()

    def enter_low_level(self, measured: np.ndarray):
        """Release Unitree's motion service, matching SONICMJ real-robot debug mode."""
        if self._low_level_enabled:
            return
        status, result = self._motion_switcher.CheckMode()
        if status != 0:
            raise RuntimeError(f"MotionSwitcher CheckMode failed: status={status}, result={result}")
        for _ in range(10):
            if not result.get("name"):
                break
            status, result = self._motion_switcher.ReleaseMode()
            if status != 0:
                raise RuntimeError(f"MotionSwitcher ReleaseMode failed: status={status}, result={result}")
            time.sleep(1.0)
            status, result = self._motion_switcher.CheckMode()
        if result.get("name"):
            raise RuntimeError(f"failed to release active motion mode: {result}")
        self._left_arm_hold = measured[LEFT_ARM_MOTORS].copy()
        self._right_arm_target = measured[RIGHT_ARM_MOTORS].copy()
        self._low_level_enabled = True
        self._publisher_stop.clear()
        self._publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publisher_thread.start()
        print(f"[LOW LEVEL] rt/lowcmd enabled; lower body={self._lower_body_mode}")

    def send_right_arm(self, target: np.ndarray, measured: np.ndarray):
        if not self._low_level_enabled or self._left_arm_hold is None:
            raise RuntimeError("low-level control is not enabled")
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("right-arm target must contain 7 finite values")
        with self._command_lock:
            self._right_arm_target = target.copy()

    def _publish_loop(self):
        """Keep rt/lowcmd alive at 100 Hz while inference runs asynchronously on the main thread."""
        period = 0.01
        while not self._publisher_stop.is_set():
            start = time.monotonic()
            try:
                self._write_lowcmd()
            except Exception as exc:
                print(f"\n[LOWCMD ERROR] {exc}", flush=True)
                self._publisher_stop.set()
                return
            self._publisher_stop.wait(max(0.0, period - (time.monotonic() - start)))

    def _write_lowcmd(self):
        with self._command_lock:
            if self._right_arm_target is None:
                return
            target = self._right_arm_target.copy()
        self._cmd.level_flag = 0xFF
        self._cmd.mode_pr = 0
        self._cmd.mode_machine = self._mode_machine
        for i in range(29):
            motor = self._cmd.motor_cmd[i]
            motor.mode = 0x01
            motor.dq = 0.0
            motor.tau = 0.0
            if i in LOWER_BODY_MOTORS:
                motor.q = POS_STOP_F
                motor.kp = 0.0
                if self._lower_body_mode == "damping":
                    motor.kd = float(LOWER_BODY_KD[i])
                else:
                    motor.dq = VEL_STOP_F
                    motor.kd = 0.0
            elif i in LEFT_ARM_MOTORS:
                arm_index = i - LEFT_ARM_MOTORS[0]
                motor.q = float(self._left_arm_hold[arm_index])
                motor.kp = float(ARM_KP[arm_index])
                motor.kd = float(ARM_KD[arm_index])
            else:
                arm_index = i - RIGHT_ARM_MOTORS[0]
                motor.q = float(target[arm_index])
                motor.kp = float(ARM_KP[arm_index])
                motor.kd = float(ARM_KD[arm_index])
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._publisher.Write(self._cmd)

    def stop_low_level_publisher(self):
        self._publisher_stop.set()
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=1.0)

    def send_inspire_hands(self, right_target: np.ndarray, left_target: np.ndarray):
        """Publish normalized Inspire commands; bridge order is right 6 then left 6."""
        right = np.clip(np.asarray(right_target, dtype=np.float64), 0.0, 1.0)
        left = np.clip(np.asarray(left_target, dtype=np.float64), 0.0, 1.0)
        if right.shape != (6,) or left.shape != (6,):
            raise ValueError("Inspire hand targets must each contain 6 values")
        for i, value in enumerate(np.concatenate((right, left))):
            self._hand_cmd.cmds[i].q = float(value)
        self._hand_publisher.Write(self._hand_cmd)


class EStop:
    def __init__(self):
        self.latched = False
        self.reason = ""

    def trigger(self, reason: str):
        if not self.latched:
            self.latched, self.reason = True, reason
            print(f"\n[E-STOP] 已锁存：{reason}；保持当前实测关节角", flush=True)


def _read_key() -> str | None:
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1).lower()
    return None


def _minimum_jerk(start: np.ndarray, goal: np.ndarray, progress: float) -> np.ndarray:
    """Zero-velocity/acceleration interpolation at both endpoints."""
    x = float(np.clip(progress, 0.0, 1.0))
    blend = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
    return start + blend * (goal - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument(
        "--lower-body-mode",
        choices=("damping", "zero-torque"),
        default="damping",
        help="legs/waist receive no position target; damping is the safer default",
    )
    parser.add_argument("--camera", default="0", help="OpenCV camera index or URL")
    parser.add_argument("--instruction", default="pick up the pipette")
    parser.add_argument("--frequency", type=float, default=10.0)
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--max-speed", type=float, default=0.25, help="rad/s")
    parser.add_argument("--max-hand-speed", type=float, default=0.5, help="normalized Inspire units/s")
    parser.add_argument("--initial-speed", type=float, default=0.15, help="initialization speed limit in rad/s")
    parser.add_argument("--initial-duration", type=float, default=5.0, help="minimum initialization duration in seconds")
    parser.add_argument("--initial-tolerance", type=float, default=0.04, help="ready tolerance in rad")
    parser.add_argument(
        "--initial-right-arm",
        type=float,
        nargs=7,
        default=DEFAULT_INITIAL_RIGHT_ARM.tolist(),
        metavar=("SP", "SR", "SY", "E", "WR", "WP", "WY"),
        help="right-arm deployment pose; default is the exact episode-0 initial state",
    )
    parser.add_argument(
        "--initial-right-hand",
        "--right-hand-state",
        dest="initial_right_hand",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_RIGHT_HAND.tolist(),
        help="episode-0 Inspire-hand initial state and 13-D model input state",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000,
        help="ZMQ send/receive timeout; includes first-inference CUDA warm-up",
    )
    parser.add_argument("--arm", action="store_true", help="actually publish rt/lowcmd")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        raise SystemExit("必须在交互终端运行，确保键盘急停可用")

    robot = G1DDS(args.network_interface, args.lower_body_mode)
    policy = PolicyClient(args.server, args.port, args.timeout_ms)
    camera = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    if not camera.isOpened():
        raise SystemExit(f"无法打开相机 {args.camera}")
    estop, period = EStop(), 1.0 / args.frequency
    signal.signal(signal.SIGINT, lambda *_: estop.trigger("Ctrl-C"))
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    initial_pose = np.asarray(args.initial_right_arm, dtype=np.float64)
    right_hand_state = np.asarray(args.initial_right_hand, dtype=np.float64)
    if np.any(initial_pose < RIGHT_ARM_LOWER) or np.any(initial_pose > RIGHT_ARM_UPPER):
        raise SystemExit("initial-right-arm exceeds URDF joint limits")
    if np.any(right_hand_state < 0.0) or np.any(right_hand_state > 1.0):
        raise SystemExit("initial-right-hand must be normalized to [0, 1]")
    print(f"初始化目标关节角: {np.round(initial_pose, 4)}")
    print(f"episode 0 Inspire 手初态: {np.round(right_hand_state, 4)}")
    print("SPACE/Q=急停；ENTER=开始抬臂初始化；到达 READY 后按 L 才启动模型。")
    enabled = False
    phase = "DRY_RUN" if not args.arm else "DISARMED"
    init_start_q = None
    init_start_time = 0.0
    init_duration = args.initial_duration
    cache = np.empty((0, 13))
    left_hand_hold = right_hand_state.copy()
    try:
        while not estop.latched:
            start = time.monotonic()
            key = _read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard")
                break
            if key in ("\n", "\r") and args.arm and phase == "DISARMED":
                measured = robot.state(0.2)
                robot.enter_low_level(measured)
                enabled = True
                phase = "START_INITIALIZATION"
                print("[ARMED] 已启用 rt/lowcmd，开始移动到初始化姿态；机器人必须可靠吊挂")
            if key == "l" and phase == "READY":
                phase = "INFERENCE"
                cache = np.empty((0, 13))
                print("[INFERENCE] 模型已接管右臂")
            measured = robot.state(0.2)
            arm_q = measured[RIGHT_ARM_MOTORS]
            if phase == "START_INITIALIZATION":
                init_start_q = arm_q.copy()
                init_start_time = time.monotonic()
                # Minimum-jerk peak velocity is 1.875 * distance / duration.
                speed_duration = 1.875 * float(np.max(np.abs(initial_pose - init_start_q))) / args.initial_speed
                init_duration = max(args.initial_duration, speed_duration)
                phase = "INITIALIZING"
                print(f"[INITIALIZING] 预计 {init_duration:.1f}s；插值期间 SPACE/Q 同样有效")
            if phase == "INITIALIZING":
                progress = (time.monotonic() - init_start_time) / init_duration
                target = _minimum_jerk(init_start_q, initial_pose, progress)
                robot.send_right_arm(target, measured)
                robot.send_inspire_hands(right_hand_state, left_hand_hold)
                if progress >= 1.0 and np.max(np.abs(arm_q - initial_pose)) <= args.initial_tolerance:
                    phase = "READY"
                    print("[READY] 初始化姿态已到位。检查现场安全后按 L 启动模型")
                time.sleep(max(0.0, period - (time.monotonic() - start)))
                continue
            if phase in ("DISARMED", "READY"):
                if phase == "READY":
                    robot.send_right_arm(initial_pose, measured)
                    robot.send_inspire_hands(right_hand_state, left_hand_hold)
                time.sleep(max(0.0, period - (time.monotonic() - start)))
                continue
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("camera frame unavailable")
            if len(cache) == 0:
                image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
                model_state = np.concatenate((arm_q, right_hand_state))
                predicted = policy.predict(image, model_state, args.instruction)
                if predicted.ndim != 2 or predicted.shape[1] != 13 or not np.isfinite(predicted).all():
                    raise RuntimeError(f"invalid policy output {predicted.shape}")
                cache = predicted[: args.execution_horizon]
            desired_arm, desired_hand, cache = cache[0, :7], cache[0, 7:13], cache[1:]
            if np.any(desired_arm < RIGHT_ARM_LOWER) or np.any(desired_arm > RIGHT_ARM_UPPER):
                raise RuntimeError("policy target exceeds URDF joint limits")
            if np.any(desired_hand < 0.0) or np.any(desired_hand > 1.0):
                raise RuntimeError("policy Inspire target exceeds normalized limits")
            max_step = args.max_speed * period
            target = arm_q + np.clip(desired_arm - arm_q, -max_step, max_step)
            hand_max_step = args.max_hand_speed * period
            right_hand_state += np.clip(desired_hand - right_hand_state, -hand_max_step, hand_max_step)
            if enabled:
                robot.send_right_arm(target, measured)
                robot.send_inspire_hands(right_hand_state, left_hand_hold)
            else:
                print(f"\rdry-run q_target={np.round(target, 3)}", end="", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - start)))
    except Exception as exc:
        estop.trigger(str(exc))
    finally:
        try:
            if enabled:
                measured = robot.state(0.2)
                for _ in range(20):
                    robot.send_right_arm(measured[RIGHT_ARM_MOTORS], measured)
                    robot.send_inspire_hands(right_hand_state, left_hand_hold)
                    time.sleep(0.01)
        finally:
            robot.stop_low_level_publisher()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
            camera.release()


if __name__ == "__main__":
    main()
