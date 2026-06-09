# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import Any, Callable, Dict

import msgpack
import numpy as np
import torch
import zmq

# Optional import for WebSocket (legacy code path; not used by InferenceClient)
try:
    import websockets.sync.client
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


class TorchSerializer:
    """Serializer using torch.save/torch.load - used by local WBC servers."""

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


class MsgSerializer:
    """Serializer using msgpack with numpy support - compatible with gr00t inference server."""

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if "__ndarray_class__" in obj:
            obj = np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "*", port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)
                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(b"ERROR")


class BaseInferenceClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return TorchSerializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def set_observation(self, observation: dict[str, Any]):
        self.call_endpoint("set_observation", data=observation)

    def get_action(self, time: float | None = None) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("get_action", data={"time": time})

    def get_modality_config(self) -> dict[str, Any]:
        return self.call_endpoint("get_modality_config")


class RobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with RobotInferenceServer (uses TorchSerializer).
    For gr00t inference server, use Gr00tInferenceClient instead.
    """

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the server given observations.

        Args:
            observations: Dict containing video, state, and annotation data

        Returns:
            Dict containing predicted actions
        """
        return self.call_endpoint("get_action", data=observations)

    def get_modality_config(self) -> dict[str, Any]:
        return self.call_endpoint("get_modality_config", requires_input=False)


class Gr00tInferenceClient:
    """
    Client for communicating with gr00t inference server.
    Uses MsgSerializer (msgpack) for compatibility with gr00t.eval.service.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server using MsgSerializer.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)

        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the gr00t inference server given observations.

        Args:
            observations: Dict containing video, state, and annotation data

        Returns:
            Dict containing predicted actions
        """
        return self.call_endpoint("get_action", data=observations)

    def get_modality_config(self) -> dict[str, Any]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def __del__(self):
        """Cleanup resources on destruction"""
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()


# =============================================================================
# Inference Client (ZMQ + msgpack)
# =============================================================================

def _pack_numpy_array(obj):
    """Pack numpy arrays for msgpack serialization (uses tobytes)."""
    if isinstance(obj, np.ndarray) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_numpy_array(obj):
    """Unpack numpy arrays from msgpack serialization."""
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"]
        )

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


_packb = partial(msgpack.packb, default=_pack_numpy_array)
_unpackb = partial(msgpack.unpackb, object_hook=_unpack_numpy_array)


class InferenceClient:
    """
    Client for communicating with a model inference server via ZMQ REQ/REP
    with msgpack (tobytes) serialization for numpy arrays.

    Example usage:
        client = InferenceClient(host="localhost", port=5556)
        query = {
            "examples": [{
                "image": [image_array],  # [H, W, 3] uint8
                "lang": "pick the object",
            }],
            "use_ddim": True,
            "num_ddim_steps": 10,
        }
        response = client.predict_action(query)
        normalized_actions = response["data"]["normalized_actions"]  # [B, T, D]
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        timeout_ms: int = 15000,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the ZMQ socket."""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        """Check if server is alive."""
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint (e.g., "get_action", "ping").
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.

        Returns:
            Response dict from server.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(_packb(request))
        message = self.socket.recv()
        return _unpackb(message)

    def predict_action(self, query_info: Dict) -> Dict:
        """
        Query inference server for action prediction.

        Args:
            query_info: Dict containing:
                - examples: List of observation dicts, each with:
                    - image: List of image arrays [H, W, 3] uint8
                    - lang: Task description string
                    - state: Optional state array [1, state_dim]
                - use_ddim: bool (default True)
                - num_ddim_steps: int (default 10)

        Returns:
            Dict containing:
                - status: "ok" or "error"
                - ok: bool
                - type: "inference_result"
                - data: {"normalized_actions": np.ndarray [B, T, action_dim]}
        """
        response = self.call_endpoint("get_action", data=query_info)

        if response.get("status") == "error" or not response.get("ok", True):
            error_info = response.get("error", {})
            raise RuntimeError(f"Inference server error: {error_info}")

        return response

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get action in format compatible with policy loop.
        This is a convenience wrapper around predict_action.

        Args:
            observations: Dict containing image, lang, and optionally state

        Returns:
            Dict with normalized_actions
        """
        query = {
            "examples": [observations],
            "use_ddim": observations.get("use_ddim", True),
            "num_ddim_steps": observations.get("num_ddim_steps", 10),
        }
        response = self.predict_action(query)
        return response.get("data", {})

    def close(self) -> None:
        """Close ZMQ connection."""
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()

    def __del__(self):
        """Cleanup resources on destruction."""
        self.close()
