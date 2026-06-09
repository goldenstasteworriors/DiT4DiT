from copy import deepcopy
from datetime import datetime
import glob
import logging
import os
import sys
import time
from typing import Optional

import rclpy
import subprocess
import tyro

import pdb


# Install stdout/stderr filters FIRST, before any other imports
# This ensures the unitree SDK (imported later) uses the filtered streams
class _FilteredStdout:
    def __init__(self, original):
        self.original = original
        self.filtered_last = False  # Track if we just filtered a message

    def write(self, message):
        # Filter the error message itself
        if "[Reader] take sample error" in message:
            self.filtered_last = True
            return

        # Also filter the newline that follows a filtered message
        if self.filtered_last and message in ("\n", "\r\n", "\r"):
            self.filtered_last = False
            return

        self.filtered_last = False
        self.original.write(message)

    def flush(self):
        self.original.flush()


class _FilteredStderr:
    def __init__(self, original):
        self.original = original
        self.filtered_last = False

    def write(self, message):
        # Filter the error message itself
        if "[Reader] take sample error" in message:
            self.filtered_last = True
            return

        # Also filter the newline that follows a filtered message
        if self.filtered_last and message in ("\n", "\r\n", "\r"):
            self.filtered_last = False
            return

        self.filtered_last = False
        self.original.write(message)

    def flush(self):
        self.original.flush()


# Replace stdout/stderr BEFORE importing modules that use unitree SDK
sys.stdout = _FilteredStdout(sys.stdout)
sys.stderr = _FilteredStderr(sys.stderr)


def setup_logging(log_dir: Optional[str] = None) -> str:
    """Setup Python logging to file with console output.

    Returns the path to the log file.
    """
    if log_dir is None:
        # Find project root by looking for pyproject.toml or .git
        current = os.path.dirname(os.path.abspath(__file__))
        while current != "/":
            if os.path.exists(os.path.join(current, "pyproject.toml")) or os.path.exists(
                os.path.join(current, ".git")
            ):
                log_dir = os.path.join(current, "logs")
                break
            current = os.path.dirname(current)
        else:
            log_dir = os.path.expanduser("~/logs")

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"g1_control_loop_{timestamp}.log")

    # Configure Python logging - simple file + console output
    # Note: stdout/stderr are already filtered at module level (see top of file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
    )

    print("=" * 70)
    print("G1 Control Loop - Log Started")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Log file: {log_file}")
    print("=" * 70)

    return log_file


from gr00t_wbc.control.envs.g1.g1_env import G1Env
from gr00t_wbc.control.utils.signal_handler import (
    SignalHandler,
    is_shutdown_requested,
    cleanup_dds_shared_memory,
)
from gr00t_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
    JOINT_SAFETY_STATUS_TOPIC,
    LOWER_BODY_POLICY_STATUS_TOPIC,
    ROBOT_CONFIG_TOPIC,
    STATE_TOPIC_NAME,
)
from gr00t_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from gr00t_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from gr00t_wbc.control.robot_model.instantiation import get_robot_type_and_model
from gr00t_wbc.control.utils.keyboard_dispatcher import (
    KeyboardDispatcher,
    KeyboardEStop,
    KeyboardListenerPublisher,
    ROSKeyboardDispatcher,
)
from gr00t_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
    ROSMsgSubscriber,
    ROSServiceServer,
)
from gr00t_wbc.control.utils.telemetry import Telemetry

CONTROL_NODE_NAME = "ControlPolicy"

# Process patterns to kill during cleanup (G1-related processes)
G1_PROCESS_PATTERNS = [
    "run_g1_control_loop",
    "g1_control",
    "G1Env",
    "ControlPolicy",
    "WBCPolicy",
    "teleop_policy_loop",
    "Gr00tKeyboard",
    "ros2_daemon",
    "ros2cli.daemon",
]


def kill_g1_processes(exclude_current: bool = True, verbose: bool = True):
    """Kill all G1-related processes.

    Args:
        exclude_current: If True, don't kill the current process
        verbose: If True, print status messages
    """
    current_pid = str(os.getpid())
    killed_count = 0

    for pattern in G1_PROCESS_PATTERNS:
        try:
            # Find processes matching pattern
            result = subprocess.run(
                ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=2
            )

            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                pids_to_kill = [p for p in pids if not exclude_current or p != current_pid]

                for pid in pids_to_kill:
                    try:
                        subprocess.run(["kill", "-9", pid], capture_output=True, timeout=1)
                        killed_count += 1
                    except:
                        pass
        except:
            pass

    # Also stop ROS2 daemon gracefully first
    try:
        subprocess.run(["ros2", "daemon", "stop"], capture_output=True, timeout=5)
    except:
        pass

    if verbose and killed_count > 0:
        print(f"[Cleanup] Killed {killed_count} stale G1-related process(es)")

    return killed_count


def cleanup_dds_files():
    """Clean up DDS shared memory files."""
    try:
        cleanup_dds_shared_memory()
    except:
        pass

    # Also clean cyclonedds files
    try:
        import glob as g

        for f in g.glob("/dev/shm/cyclonedds_*") + g.glob("/dev/shm/fastrtps_*"):
            try:
                os.remove(f)
            except:
                pass
    except:
        pass


def cleanup_startup_state():
    """Clean up all startup state: processes only.

    This ensures a clean environment before initializing the control loop by:
    1. Killing existing G1-related processes that may conflict

    NOTE: We do NOT clean up DDS shared memory files on startup because:
    - The robot's DDS participants are already using these files
    - Deleting them breaks communication with the robot
    - DDS cleanup should only happen on shutdown for files we created
    """
    print("[Startup] Cleaning up stale processes...")

    # Only kill existing G1 processes (except current)
    # Do NOT clean DDS files - they are shared with the robot
    kill_g1_processes(exclude_current=True, verbose=True)

    print("[Startup] Cleanup complete")


def main(config: ControlLoopConfig):
    # Install signal handlers for graceful shutdown (Ctrl+C)
    signal_handler = SignalHandler()

    # Register callback to shutdown ROS when signal is received
    def shutdown_ros_on_signal():
        try:
            if rclpy.ok():
                print("[G1ControlLoop] Shutting down ROS due to signal...")
                rclpy.shutdown()
        except Exception as e:
            print(f"[G1ControlLoop] Error shutting down ROS in signal handler: {e}")

    signal_handler.register_cleanup(shutdown_ros_on_signal)

    # Clean up all startup state before initializing
    cleanup_startup_state()

    # Track resources for cleanup
    env = None
    dispatcher = None
    ros_manager = None

    try:
        ros_manager = ROSManager(node_name=CONTROL_NODE_NAME)
        node = ros_manager.node
    except KeyboardInterrupt:
        print("[G1ControlLoop] Interrupted during ROS initialization")
        return

    # start the robot config server
    ROSServiceServer(ROBOT_CONFIG_TOPIC, config.to_dict())

    wbc_config = config.load_wbc_yaml()

    data_exp_pub = ROSMsgPublisher(STATE_TOPIC_NAME)
    lower_body_policy_status_pub = ROSMsgPublisher(LOWER_BODY_POLICY_STATUS_TOPIC)
    joint_safety_status_pub = ROSMsgPublisher(JOINT_SAFETY_STATUS_TOPIC)

    # Initialize telemetry
    telemetry = Telemetry(window_size=100)

    # Select robot model based on hand_type configuration
    # Only use hand-specific model if hands are actually enabled
    if config.with_hands and config.hand_type == "aloha":
        robot_name = "g1_aloha"
        print(f"[G1ControlLoop] Using ALOHA robot model (31 DOF)")
    else:
        robot_name = "g1"
        if not config.with_hands:
            print(f"[G1ControlLoop] Hands disabled, using G1 model (29 DOF body only)")
        else:
            print(f"[G1ControlLoop] Using G1 model with three-finger hands (43 DOF)")

    robot_type, robot_model = get_robot_type_and_model(
        robot=robot_name,
        high_elbow_pose=config.high_elbow_pose,
    )

    try:
        env = G1Env(
            env_name=config.env_name,
            robot_model=robot_model,
            config=wbc_config,
            wbc_version=config.wbc_version,
        )
    except KeyboardInterrupt:
        print("[G1ControlLoop] Interrupted during environment initialization")
        if ros_manager:
            ros_manager.shutdown()
        return

    if env.sim and not config.sim_sync_mode:
        env.start_simulator()

    wbc_policy = get_wbc_policy("g1", robot_model, wbc_config, config.upper_body_joint_speed)

    keyboard_listener_pub = KeyboardListenerPublisher()
    keyboard_estop = KeyboardEStop()
    if config.keyboard_dispatcher_type == "raw":
        dispatcher = KeyboardDispatcher()
    elif config.keyboard_dispatcher_type == "ros":
        dispatcher = ROSKeyboardDispatcher()
    else:
        raise ValueError(
            f"Invalid keyboard dispatcher: {config.keyboard_dispatcher_type}, please use 'raw' or 'ros'"
        )

    dispatcher.register(env)
    dispatcher.register(wbc_policy)
    dispatcher.register(keyboard_listener_pub)
    dispatcher.register(keyboard_estop)
    dispatcher.start()

    rate = node.create_rate(config.control_frequency)

    upper_body_policy_subscriber = ROSMsgSubscriber(CONTROL_GOAL_TOPIC)

    last_teleop_cmd = None

    try:
        print("Starting control loop (Ctrl+C to stop)")
        print("ROS_Manager is running... : ", ros_manager.ok())

        # Main control loop - check both ROS and signal handler
        while ros_manager.ok() and not is_shutdown_requested():
            # Check shutdown request at start of loop (for responsive Ctrl+C)
            if is_shutdown_requested():
                print("[G1ControlLoop] Shutdown requested, exiting loop")
                break

            t_start = time.monotonic()
            with telemetry.timer("total_loop"):
                # Step simulator if in sync mode
                with telemetry.timer("step_simulator"):
                    if env.sim and config.sim_sync_mode:
                        env.step_simulator()

                # Measure observation time
                with telemetry.timer("observe"):
                    obs = env.observe()
                    wbc_policy.set_observation(obs)

                # Measure policy setup time
                with telemetry.timer("policy_setup"):
                    # Process ALL queued messages to ensure no button events are lost
                    # Use latest message for teleop commands, but accumulate button events
                    upper_body_cmd = None
                    has_toggle_data_collection = False
                    has_toggle_data_abort = False
                    has_reset_env_and_policy = False

                    while True:
                        msg = upper_body_policy_subscriber.get_msg()
                        if msg is None:
                            break
                        upper_body_cmd = msg  # Keep latest for teleop commands
                        # Accumulate button events (OR logic - if any message has it, trigger it)
                        if msg.get("toggle_data_collection", False):
                            has_toggle_data_collection = True
                        if msg.get("toggle_data_abort", False):
                            has_toggle_data_abort = True
                        if msg.get("reset_env_and_policy", False):
                            has_reset_env_and_policy = True

                    t_now = time.monotonic()

                    wbc_goal = {}
                    if upper_body_cmd:
                        wbc_goal = upper_body_cmd.copy()
                        last_teleop_cmd = upper_body_cmd.copy()
                        # Merge accumulated button events into wbc_goal
                        wbc_goal["toggle_data_collection"] = has_toggle_data_collection
                        wbc_goal["toggle_data_abort"] = has_toggle_data_abort
                        wbc_goal["reset_env_and_policy"] = has_reset_env_and_policy

                        if has_toggle_data_collection:
                            print(f"[G1ControlLoop] toggle_data_collection detected in queue")
                        if config.ik_indicator:
                            env.set_ik_indicator(upper_body_cmd)
                    # Send goal to policy
                    if wbc_goal:
                        wbc_goal["interpolation_garbage_collection_time"] = t_now - 2 * (
                            1 / config.control_frequency
                        )
                        wbc_policy.set_goal(wbc_goal)

                # Measure policy action calculation time
                with telemetry.timer("policy_action"):
                    wbc_action = wbc_policy.get_action(time=t_now)

                # Measure action queue time
                with telemetry.timer("queue_action"):
                    env.queue_action(wbc_action)

                # Publish status information for InteractiveModeController
                with telemetry.timer("publish_status"):
                    # Get policy status - check if the lower body policy has use_policy_action enabled
                    policy_use_action = False
                    try:
                        # Access the lower body policy through the decoupled whole body policy
                        if hasattr(wbc_policy, "lower_body_policy"):
                            policy_use_action = getattr(
                                wbc_policy.lower_body_policy, "use_policy_action", False
                            )
                    except (AttributeError, TypeError):
                        policy_use_action = False

                    policy_status_msg = {"use_policy_action": policy_use_action, "timestamp": t_now}
                    lower_body_policy_status_pub.publish(policy_status_msg)

                    # Get joint safety status from G1Env (which already runs the safety monitor)
                    joint_safety_ok = env.get_joint_safety_status()

                    joint_safety_status_msg = {
                        "joint_safety_ok": joint_safety_ok,
                        "timestamp": t_now,
                    }
                    joint_safety_status_pub.publish(joint_safety_status_msg)

                # Start or Stop data collection
                if wbc_goal.get("toggle_data_collection", False):
                    print(
                        "[G1ControlLoop] toggle_data_collection=True received, dispatching 'c' key"
                    )
                    dispatcher.handle_key("c")

                # Abort the current episode
                if wbc_goal.get("toggle_data_abort", False):
                    print("[G1ControlLoop] toggle_data_abort=True received, dispatching 'x' key")
                    dispatcher.handle_key("x")

                # Delete the just-saved episode
                if wbc_goal.get("toggle_delete_episode", False):
                    print(
                        "[G1ControlLoop] toggle_delete_episode=True received, dispatching 'd' key"
                    )
                    dispatcher.handle_key("d")

                if env.use_sim and wbc_goal.get("reset_env_and_policy", False):
                    print("Resetting sim environment and policy")
                    # Reset teleop policy & sim env
                    dispatcher.handle_key("k")

                    # Clear upper body commands
                    upper_body_policy_subscriber.clear()
                    upper_body_cmd = {
                        "target_upper_body_pose": obs["q"][
                            robot_model.get_joint_group_indices("upper_body")
                        ],
                        "wrist_pose": DEFAULT_WRIST_POSE,
                        "base_height_command": DEFAULT_BASE_HEIGHT,
                        "navigate_cmd": DEFAULT_NAV_CMD,
                    }
                    last_teleop_cmd = upper_body_cmd.copy()

                    time.sleep(0.5)

                msg = deepcopy(obs)
                for key in obs.keys():
                    if key.endswith("_image"):
                        del msg[key]

                # exporting data
                if last_teleop_cmd:
                    msg.update(
                        {
                            "action": wbc_action["q"],
                            "action.eef": last_teleop_cmd.get("wrist_pose", DEFAULT_WRIST_POSE),
                            "base_height_command": last_teleop_cmd.get(
                                "base_height_command", DEFAULT_BASE_HEIGHT
                            ),
                            "navigate_command": last_teleop_cmd.get(
                                "navigate_cmd", DEFAULT_NAV_CMD
                            ),
                            "torso_orientation_rpy": last_teleop_cmd.get(
                                "torso_orientation_rpy", [0.0, 0.0, 0.0]
                            ),
                            "timestamps": {
                                "main_loop": time.time(),
                                "proprio": time.time(),
                            },
                        }
                    )
                data_exp_pub.publish(msg)
                end_time = time.monotonic()

            if env.sim and (not env.sim.sim_thread or not env.sim.sim_thread.is_alive()):
                raise RuntimeError("Simulator thread is not alive")

            rate.sleep()

            # Log timing information - rate limited to avoid spam
            current_time = time.time()
            if config.verbose_timing:
                # When verbose timing is enabled, always show timing
                telemetry.log_timing_info(context="G1 Control Loop", threshold=0.0)
            elif (end_time - t_start) > 0.100 and not config.sim_sync_mode:
                # Only show timing when loop is significantly slow (>100ms)
                # and rate limit to once every 5 seconds
                if (
                    not hasattr(main, "_last_timing_log")
                    or (current_time - main._last_timing_log) > 5.0
                ):
                    main._last_timing_log = current_time
                    telemetry.log_timing_info(context="G1 Control Loop Missed", threshold=0.010)

    except ros_manager.exceptions() as e:
        print(f"\n[G1ControlLoop] ROSManager interrupted by user: {e}")
    except KeyboardInterrupt:
        print("\n[G1ControlLoop] Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"[G1ControlLoop] Error in control loop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("[G1ControlLoop] Cleaning up...")
        # Clean up in safe order - check if each resource exists before cleanup
        try:
            if dispatcher is not None:
                print("[G1ControlLoop] Stopping keyboard dispatcher...")
                dispatcher.stop()
        except Exception as e:
            print(f"[G1ControlLoop] Error stopping dispatcher: {e}")

        try:
            if ros_manager is not None:
                print("[G1ControlLoop] Shutting down ROS...")
                ros_manager.shutdown()
        except Exception as e:
            print(f"[G1ControlLoop] Error shutting down ROS: {e}")

        try:
            if env is not None:
                print("[G1ControlLoop] Closing environment...")
                env.close()
        except Exception as e:
            print(f"[G1ControlLoop] Error closing environment: {e}")

        # Kill orphan G1 processes only
        # NOTE: We do NOT clean DDS files - they are shared with robot and other participants
        print("[G1ControlLoop] Killing orphan processes...")
        kill_g1_processes(exclude_current=True, verbose=True)

        # Clean up signal handler
        signal_handler.cleanup()
        print("[G1ControlLoop] Cleanup complete")

        # Give DDS threads time to finish before exit
        time.sleep(0.5)

        # Force exit to ensure all threads terminate
        # Use os._exit to avoid C++ destructor issues with DDS
        os._exit(0)


if __name__ == "__main__":
    # # Setup logging to save all output to a file
    # log_file = setup_logging()
    # logging.info(f"[G1ControlLoop] All output being logged to: {log_file}")

    config = tyro.cli(ControlLoopConfig)
    main(config)
