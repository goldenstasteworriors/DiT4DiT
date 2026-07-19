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
    print("SPACE/Q=急停。默认 dry-run；--arm 模式下按 ENTER 二次解锁。")
    enabled = False
    cache = np.empty((0, 13))
    try:
        while not estop.latched:
            start = time.monotonic()
            key = _read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard")
                break
            if key in ("\n", "\r") and args.arm:
                enabled = True
                print("[ARMED] 已启用真机下发")
            measured = robot.state(0.2)
            arm_q = measured[RIGHT_ARM_MOTORS]
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("camera frame unavailable")
            if len(cache) == 0:
                image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
                predicted = policy.predict(image, arm_q, args.instruction)
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
