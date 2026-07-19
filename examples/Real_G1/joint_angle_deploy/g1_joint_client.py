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
RIGHT_ARM_LOWER = np.array([-3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144])
RIGHT_ARM_UPPER = np.array([2.6704, 1.5882, 2.618, 2.0944, 1.9722, 1.6144, 1.6144])
# Median of frame 0 from the 19 pipette-right-joints training episodes.
DEFAULT_INITIAL_RIGHT_ARM = np.array([-0.060281, -0.251992, -0.072517, -0.577184, 0.402035, 0.493582, -0.250482])
DEFAULT_RIGHT_HAND_STATE = np.array([0.998, 1.0, 0.998, 0.998, 0.999, 0.984])


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
        self.socket.send(msgpack.packb(request, default=_pack_array))
        response = msgpack.unpackb(self.socket.recv(), object_hook=_unpack_array)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error")))
        return np.asarray(response["data"]["unnormalized_actions"], dtype=np.float64)


class G1DDS:
    def __init__(self, network_interface: str):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0, network_interface)
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._q = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        # Unitree's arm SDK overlay leaves the stock lower-body controller active.
        # Publishing rt/lowcmd directly here could take over the legs as well.
        self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_state, 10)

    def _on_state(self, msg):
        with self._lock:
            self._q = np.array([motor.q for motor in msg.motor_state[:29]], dtype=np.float64)
            self._stamp = time.monotonic()

    def state(self, max_age: float) -> np.ndarray:
        with self._lock:
            if self._q is None or time.monotonic() - self._stamp > max_age:
                raise RuntimeError("lowstate missing or stale")
            return self._q.copy()

    def send_right_arm(self, target: np.ndarray, measured: np.ndarray):
        self._cmd.motor_cmd[29].q = 1.0  # Unitree arm_sdk enable weight
        for i in np.concatenate((LEFT_ARM_MOTORS, RIGHT_ARM_MOTORS)):
            motor = self._cmd.motor_cmd[i]
            motor.q = float(target[np.where(RIGHT_ARM_MOTORS == i)[0][0]]) if i in RIGHT_ARM_MOTORS else float(measured[i])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 40.0 if i in np.concatenate((LEFT_ARM_MOTORS[:4], RIGHT_ARM_MOTORS[:4])) else 15.0
            motor.kd = 1.0 if i in np.concatenate((LEFT_ARM_MOTORS[:4], RIGHT_ARM_MOTORS[:4])) else 0.5
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._publisher.Write(self._cmd)


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
    parser.add_argument("--camera", default="0", help="OpenCV camera index or URL")
    parser.add_argument("--instruction", default="pick up the pipette")
    parser.add_argument("--frequency", type=float, default=10.0)
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--max-speed", type=float, default=0.25, help="rad/s")
    parser.add_argument("--initial-speed", type=float, default=0.15, help="initialization speed limit in rad/s")
    parser.add_argument("--initial-duration", type=float, default=5.0, help="minimum initialization duration in seconds")
    parser.add_argument("--initial-tolerance", type=float, default=0.04, help="ready tolerance in rad")
    parser.add_argument(
        "--initial-right-arm",
        type=float,
        nargs=7,
        default=DEFAULT_INITIAL_RIGHT_ARM.tolist(),
        metavar=("SP", "SR", "SY", "E", "WR", "WP", "WY"),
        help="right-arm deployment pose; default is the training-episode frame-0 median",
    )
    parser.add_argument(
        "--right-hand-state",
        type=float,
        nargs=6,
        default=DEFAULT_RIGHT_HAND_STATE.tolist(),
        help="right Inspire-hand state appended to the 7 arm joints for the 13-D model input",
    )
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--arm", action="store_true", help="actually publish rt/lowcmd")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        raise SystemExit("必须在交互终端运行，确保键盘急停可用")

    robot = G1DDS(args.network_interface)
    policy = PolicyClient(args.server, args.port, args.timeout_ms)
    camera = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    if not camera.isOpened():
        raise SystemExit(f"无法打开相机 {args.camera}")
    estop, period = EStop(), 1.0 / args.frequency
    signal.signal(signal.SIGINT, lambda *_: estop.trigger("Ctrl-C"))
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    initial_pose = np.asarray(args.initial_right_arm, dtype=np.float64)
    right_hand_state = np.asarray(args.right_hand_state, dtype=np.float64)
    if np.any(initial_pose < RIGHT_ARM_LOWER) or np.any(initial_pose > RIGHT_ARM_UPPER):
        raise SystemExit("initial-right-arm exceeds URDF joint limits")
    print(f"初始化目标关节角: {np.round(initial_pose, 4)}")
    print("SPACE/Q=急停；ENTER=开始抬臂初始化；到达 READY 后按 L 才启动模型。")
    enabled = False
    phase = "DRY_RUN" if not args.arm else "DISARMED"
    init_start_q = None
    init_start_time = 0.0
    init_duration = args.initial_duration
    cache = np.empty((0, 13))
    try:
        while not estop.latched:
            start = time.monotonic()
            key = _read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard")
                break
            if key in ("\n", "\r") and args.arm and phase == "DISARMED":
                enabled = True
                phase = "START_INITIALIZATION"
                print("[ARMED] 已启用 arm_sdk，开始移动到初始化姿态")
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
                if progress >= 1.0 and np.max(np.abs(arm_q - initial_pose)) <= args.initial_tolerance:
                    phase = "READY"
                    print("[READY] 初始化姿态已到位。检查现场安全后按 L 启动模型")
                time.sleep(max(0.0, period - (time.monotonic() - start)))
                continue
            if phase in ("DISARMED", "READY"):
                if phase == "READY":
                    robot.send_right_arm(initial_pose, measured)
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
            desired, cache = cache[0, :7], cache[1:]
            if np.any(desired < RIGHT_ARM_LOWER) or np.any(desired > RIGHT_ARM_UPPER):
                raise RuntimeError("policy target exceeds URDF joint limits")
            max_step = args.max_speed * period
            target = arm_q + np.clip(desired - arm_q, -max_step, max_step)
            if enabled:
                robot.send_right_arm(target, measured)
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
                    time.sleep(0.01)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
            camera.release()


if __name__ == "__main__":
    main()
