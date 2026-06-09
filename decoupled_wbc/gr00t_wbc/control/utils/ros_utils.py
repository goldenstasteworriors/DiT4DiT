import base64
import signal
import threading
from collections import deque
from typing import Optional

import msgpack
import msgpack_numpy as mnp
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
from std_srvs.srv import Trigger

_signal_registered = False


def register_keyboard_interrupt_handler():
    """
    Register a signal handler for SIGINT (Ctrl+C) and SIGTERM that raises KeyboardInterrupt.
    This ensures consistent exception handling across different termination signals.
    """
    global _signal_registered
    if not _signal_registered:

        def signal_handler(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        _signal_registered = True


class ROSManager:
    """
    Manages the ROS2 node and executor.

    Usage example:
    ```python
    def main():
        ros_manager = ROSManager()
        node = ros_manager.node

        try:
            while ros_manager.ok():
                time.sleep(0.1)
        except ros_manager.exceptions() as e:
            print(f"ROSManager interrupted by user: {e}")
        finally:
            ros_manager.shutdown()
    ```
    """

    def __init__(self, node_name: str = "ros_manager"):
        if not rclpy.ok():
            rclpy.init()
            self.node = rclpy.create_node(node_name)
            self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            self.thread.start()
        else:
            executor = rclpy.get_global_executor()
            if len(executor.get_nodes()) > 0:
                self.node = executor.get_nodes()[0]
            else:
                self.node = rclpy.create_node(node_name)

        register_keyboard_interrupt_handler()

    @staticmethod
    def ok():
        return rclpy.ok()

    @staticmethod
    def shutdown():
        if rclpy.ok():
            rclpy.shutdown()

    @staticmethod
    def exceptions():
        return (rclpy.exceptions.ROSInterruptException, KeyboardInterrupt)


class ROSMsgPublisher:
    """
    Publishes any serializable dict to a topic.
    """

    def __init__(self, topic_name: str):
        ros_manager = ROSManager()
        self.node = ros_manager.node
        self.publisher = self.node.create_publisher(ByteMultiArray, topic_name, 1)

    def publish(self, msg: dict):
        payload = msgpack.packb(msg, default=mnp.encode)
        payload = tuple(bytes([a]) for a in payload)
        msg = ByteMultiArray()
        msg.data = payload
        self.publisher.publish(msg)


class ROSMsgSubscriber:
    """
    Subscribes to any topics published by a ROSMsgPublisher.
    Uses a queue to ensure messages are not lost during control loop delays.
    """

    def __init__(self, topic_name: str, max_queue_size: int = 10):
        ros_manager = ROSManager()
        self.node = ros_manager.node
        self._queue = deque(maxlen=max_queue_size)
        self.subscription = self.node.create_subscription(
            ByteMultiArray, topic_name, self._callback, 10
        )

    def _callback(self, msg: ByteMultiArray):
        self._queue.append(msg)

    def get_msg(self) -> Optional[dict]:
        if not self._queue:
            return None
        msg = self._queue.popleft()
        return msgpack.unpackb(bytes([ab for a in msg.data for ab in a]), object_hook=mnp.decode)

    def clear(self):
        """Clear all queued messages."""
        self._queue.clear()


class ROSServiceServer:
    """
    Generic ROS2 Service server that stores and serves a config dict.
    """

    def __init__(self, service_name: str, config: dict):
        ros_manager = ROSManager()
        self.node = ros_manager.node
        packed = msgpack.packb(config, default=mnp.encode)
        self.message = base64.b64encode(packed).decode("ascii")
        self.server = self.node.create_service(Trigger, service_name, self._callback)

    def _callback(self, request, response):
        try:
            response.success = True
            response.message = self.message
            print("Sending encoded message of length:", len(response.message))
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response


class ROSServiceClient(Node):

    def __init__(self, service_name: str, node_name: str = "service_client"):
        super().__init__(node_name)
        self.cli = self.create_client(Trigger, service_name)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.req = Trigger.Request()

    def get_config(self):
        future = self.cli.call_async(self.req)
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        executor.spin_until_future_complete(future, timeout_sec=1.0)
        executor.remove_node(self)
        executor.shutdown()
        result = future.result()
        if result.success:
            decoded = base64.b64decode(result.message.encode("ascii"))
            return msgpack.unpackb(decoded, object_hook=mnp.decode)
        else:
            raise RuntimeError(f"Service call failed: {result.message}")
