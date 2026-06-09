import os
import subprocess
import sys
import threading
from collections import deque

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sshkeyboard import listen_keyboard, stop_listening
from std_msgs.msg import String as RosStringMsg

from gr00t_wbc.control.main.constants import KEYBOARD_INPUT_TOPIC

# Global variable to store original terminal attributes
_original_terminal_attrs = None


def save_terminal_state():
    """Save the current terminal state."""
    global _original_terminal_attrs
    try:
        import termios

        fd = sys.stdin.fileno()
        _original_terminal_attrs = termios.tcgetattr(fd)
    except (ImportError, OSError, termios.error):
        _original_terminal_attrs = None


def restore_terminal():
    """Restore terminal to original state."""
    global _original_terminal_attrs
    try:
        import termios

        if _original_terminal_attrs is not None:
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSANOW, _original_terminal_attrs)
            return
    except (ImportError, OSError, termios.error):
        pass

    # Fallback for non-Unix systems or if termios fails
    try:
        if os.name == "posix":
            os.system("stty sane")
    except OSError:
        pass


class ROSKeyboardDispatcher:
    """ROS-based keyboard dispatcher that receives keyboard events via ROS topics."""

    def __init__(self):
        self.listeners = []
        self._active = False
        assert rclpy.ok(), "Expected ROS2 to be initialized in this process..."
        executor = rclpy.get_global_executor()
        self.node = executor.get_nodes()[0]
        print("creating keyboard input subscriber...")
        self.subscription = self.node.create_subscription(
            RosStringMsg, KEYBOARD_INPUT_TOPIC, self._callback, 10
        )

    def register(self, listener):
        if not hasattr(listener, "handle_keyboard_button"):
            raise NotImplementedError("handle_keyboard_button is not implemented")
        self.listeners.append(listener)

    def start(self):
        """Start the ROS keyboard dispatcher."""
        self._active = True
        print("ROS keyboard dispatcher started")

    def stop(self):
        """Stop the ROS keyboard dispatcher and cleanup."""
        if self._active:
            self._active = False
            # Clean up subscription
            if hasattr(self, "subscription"):
                self.node.destroy_subscription(self.subscription)
            print("ROS keyboard dispatcher stopped")

    def handle_key(self, key):
        """Programmatically dispatch a key press to all registered listeners.

        This allows non-keyboard sources (e.g., Pico VR buttons) to trigger
        the same handlers as keyboard input.
        """
        print(f"[ROSKeyboardDispatcher] handle_key('{key}') called, active={self._active}, listeners={len(self.listeners)}")
        if self._active:
            for listener in self.listeners:
                print(f"[ROSKeyboardDispatcher] Dispatching '{key}' to {listener.__class__.__name__}")
                listener.handle_keyboard_button(key)

    def _callback(self, msg: RosStringMsg):
        if self._active:
            for listener in self.listeners:
                listener.handle_keyboard_button(msg.data)

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()


class KeyboardDispatcher:
    def __init__(self):
        self.listeners = []
        self._listening_thread = None
        self._stop_event = threading.Event()
        self._key = None

    def register(self, listener):
        # raise if handle_keyboard_button is not implemented
        # TODO(YL): let listener be a Callable instead of a class
        if not hasattr(listener, "handle_keyboard_button"):
            raise NotImplementedError("handle_keyboard_button is not implemented")
        self.listeners.append(listener)

    def handle_key(self, key):
        # Check if we should stop
        if self._stop_event.is_set():
            stop_listening()
            return

        for listener in self.listeners:
            listener.handle_keyboard_button(key)

    def start_listening(self):
        try:
            save_terminal_state()  # Save original terminal state before listening
            listen_keyboard(
                on_press=self.handle_key,
                delay_second_char=0.1,
                delay_other_chars=0.05,
                sleep=0.01,
            )
        except Exception as e:
            print(f"Keyboard listener stopped: {e}")
        finally:
            # Ensure terminal is restored even if an exception occurs
            self._restore_terminal()

    def start(self):
        self._listening_thread = threading.Thread(target=self.start_listening, daemon=True)
        self._listening_thread.start()

    def stop(self):
        """Stop the keyboard listener and restore terminal settings."""
        if self._listening_thread and self._listening_thread.is_alive():
            self._stop_event.set()
            # Force stop_listening to be called
            try:
                stop_listening()
            except Exception:
                pass
            # Wait a bit for the thread to finish
            self._listening_thread.join(timeout=0.5)
            # Restore terminal settings
            self._restore_terminal()

    def _restore_terminal(self):
        """Restore terminal to a sane state."""
        restore_terminal()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()


KEYBOARD_LISTENER_TOPIC_NAME = "/Gr00tKeyboardListener"

# Reliable QoS for keyboard commands - ensures delivery of start/stop recording
KEYBOARD_RELIABLE_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


class KeyboardListener:
    def __init__(self):
        self.key = None

    def handle_keyboard_button(self, key):
        self.key = key

    def pop_key(self):
        key = self.key
        self.key = None
        return key


class KeyboardListenerPublisher:
    def __init__(self, topic_name: str = KEYBOARD_LISTENER_TOPIC_NAME):
        """
        Initialize keyboard listener for remote teleop with simplified interface.

        Args:
            topic_name: ROS topic name for keyboard commands
        """
        assert rclpy.ok(), "Expected ROS2 to be initialized in this process..."
        executor = rclpy.get_global_executor()
        self.node = executor.get_nodes()[0]
        self.publisher = self.node.create_publisher(
            RosStringMsg, topic_name, KEYBOARD_RELIABLE_QOS
        )

    def handle_keyboard_button(self, key):
        print(f"[KeyboardListenerPublisher] Publishing key '{key}' to {KEYBOARD_LISTENER_TOPIC_NAME}")
        self.publisher.publish(RosStringMsg(data=key))


class KeyboardListenerSubscriber:
    def __init__(
        self,
        topic_name: str = KEYBOARD_LISTENER_TOPIC_NAME,
        node_name: str = "keyboard_listener_subscriber",
        max_queue_size: int = 10,
    ):
        """Subscribe to keyboard events via ROS topic.

        Args:
            topic_name: ROS topic name
            node_name: ROS node name (unused if node already exists)
            max_queue_size: Maximum number of key presses to buffer
        """
        assert rclpy.ok(), "Expected ROS2 to be initialized in this process..."
        executor = rclpy.get_global_executor()
        nodes = executor.get_nodes()
        if nodes:
            self.node = nodes[0]
            self._create_node = False
        else:
            self.node = rclpy.create_node("KeyboardListenerSubscriber")
            executor.add_node(self.node)
            self._create_node = True
        self.subscriber = self.node.create_subscription(
            RosStringMsg, topic_name, self._callback, KEYBOARD_RELIABLE_QOS
        )
        # Use deque to buffer multiple key presses instead of single variable
        self._data_queue = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()

    def _callback(self, msg: RosStringMsg):
        """Callback for ROS messages - adds to queue instead of overwriting."""
        with self._lock:
            self._data_queue.append(msg.data)
            print(f"[KeyboardListenerSubscriber] Received and queued key '{msg.data}' (queue size: {len(self._data_queue)})")

    def read_msg(self):
        """Read the oldest unprocessed key press from the queue.

        Returns:
            The oldest key press, or None if queue is empty.
        """
        with self._lock:
            if self._data_queue:
                return self._data_queue.popleft()
            return None


class KeyboardEStop:
    def __init__(self):
        """Initialize KeyboardEStop with automatic tmux cleanup detection."""
        # Automatically create tmux cleanup if in deployment mode
        self.cleanup_callback = self._create_tmux_cleanup_callback()

    def _create_tmux_cleanup_callback(self):
        """Create a cleanup callback that kills the tmux session if running in deployment mode."""
        tmux_session = os.environ.get("GR00T_WBC_TMUX_SESSION")

        def cleanup_callback():
            if tmux_session:
                print(f"Emergency stop: Killing tmux session '{tmux_session}'...")
                try:
                    subprocess.run(["tmux", "kill-session", "-t", tmux_session], timeout=5)
                    print("Tmux session terminated successfully.")
                except subprocess.TimeoutExpired:
                    print("Warning: Tmux session termination timed out, forcing kill...")
                    try:
                        subprocess.run(["tmux", "kill-session", "-t", tmux_session, "-9"])
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Warning: Error during tmux cleanup: {e}")
                    # If tmux cleanup fails, fallback to immediate exit
                    restore_terminal()
                    os._exit(1)
            else:
                print("Emergency stop: No tmux session, exiting normally...")
                sys.exit(1)

        return cleanup_callback

    def handle_keyboard_button(self, key):
        if key == "`":
            print("Emergency stop triggered - running cleanup...")
            self.cleanup_callback()
