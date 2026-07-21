"""MuJoCo viewer process for the ideal, zero-tracking-error deployment shadow."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
from pathlib import Path

import numpy as np


def _viewer_main(model_path: str, command_queue, ready, startup_error) -> None:
    try:
        if os.environ.get("MUJOCO_GL", "").lower() in {"egl", "osmesa"}:
            os.environ.pop("MUJOCO_GL", None)
        import mujoco
        from mujoco import viewer as mj_viewer

        from examples.PipetteRightOnly.convert_wrist_delta_to_joint_chunks import BODY_NAMES

        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        qpos = []
        for name in BODY_NAMES:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError(f"joint missing from simulation model: {name}")
            qpos.append(model.jnt_qposadr[joint])
        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
            viewer.cam.distance = 2.15
            viewer.cam.azimuth = 150
            viewer.cam.elevation = -12
            ready.set()
            while viewer.is_running():
                try:
                    state = command_queue.get(timeout=0.05)
                except queue.Empty:
                    viewer.sync()
                    continue
                if state is None:
                    return
                data.qpos[qpos] = state
                mujoco.mj_forward(model, data)
                viewer.sync()
    except Exception as exc:
        startup_error.put(str(exc))
        ready.set()


class LiveShadowMujoco:
    """Display nominal commands without reading back or influencing the robot."""

    def __init__(self, model_path: Path):
        context = mp.get_context("spawn")
        self._queue = context.Queue()
        self._ready = context.Event()
        self._errors = context.Queue()
        self._process = context.Process(
            target=_viewer_main,
            args=(str(model_path.expanduser().resolve()), self._queue, self._ready, self._errors),
            daemon=True,
        )
        self._process.start()
        if not self._ready.wait(timeout=10.0):
            self.close()
            raise RuntimeError("MuJoCo shadow viewer startup timed out")
        if not self._errors.empty():
            error = self._errors.get()
            self.close()
            raise RuntimeError(f"MuJoCo shadow viewer failed: {error}")
        print("[SIMULATION] 在线理想执行影子仿真已打开；仿真不会向机器人发送任何数据")

    def update(self, measured_body: np.ndarray, left_arm: np.ndarray, right_arm: np.ndarray) -> None:
        if not self._process.is_alive():
            error = self._errors.get() if not self._errors.empty() else "viewer window closed"
            raise RuntimeError(f"MuJoCo shadow viewer stopped: {error}")
        state = np.asarray(measured_body[:29], dtype=np.float64).copy()
        state[15:22] = left_arm
        state[22:29] = right_arm
        self._queue.put(state)

    def close(self) -> None:
        if getattr(self, "_process", None) is None:
            return
        if self._process.is_alive():
            self._queue.put(None)
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)

