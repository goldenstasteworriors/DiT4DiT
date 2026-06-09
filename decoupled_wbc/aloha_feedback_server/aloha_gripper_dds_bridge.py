import argparse
import logging
import signal
import sys
import time
from multiprocessing import Value, Array, Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelPublisher,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Vector3_

from aloha_feedback_client import Aloha_Gripper_Controller


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DDS bridge that forwards Aloha hand commands to the serial controller"
    )
    parser.add_argument("--dds-domain", type=int, default=0, help="DDS domain id")
    parser.add_argument(
        "--dds-hand-topic",
        type=str,
        default="rt/aloha_hand/cmd",
        help="DDS topic name that carries hand commands",
    )
    parser.add_argument(
        "--dds-state-topic",
        type=str,
        default="rt/aloha_hand/state",
        help="DDS topic name to publish gripper state feedback",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run controller in simulation mode (no serial access)",
    )
    parser.add_argument("--queue-length", type=int, default=1, help="DDS subscriber queue length")
    parser.add_argument(
        "--feedback-rate", type=float, default=30.0, help="Feedback publishing rate in Hz"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        ChannelFactoryInitialize(args.dds_domain)
    except Exception as exc:
        logger.error(f"Failed to initialize DDS factory: {exc}")
        sys.exit(1)

    left_cmd_value = Value("d", 0.0, lock=True)
    right_cmd_value = Value("d", 0.0, lock=True)
    dual_gripper_data_lock = Lock()
    dual_gripper_state_array = Array("d", 2, lock=False)
    dual_gripper_action_array = Array("d", 2, lock=False)

    controller = Aloha_Gripper_Controller(
        left_cmd_value,
        right_cmd_value,
        dual_gripper_data_lock,
        dual_gripper_state_array,
        dual_gripper_action_array,
        simulation_mode=args.simulation,
    )

    def command_handler(msg: Vector3_):
        with left_cmd_value.get_lock():
            left_cmd_value.value = float(msg.x)
        with right_cmd_value.get_lock():
            right_cmd_value.value = float(msg.y)
        # .3f
        logger.info(f"Received command - Left: {msg.x:.3f}, Right: {msg.y:.3f}")

    subscriber = ChannelSubscriber(args.dds_hand_topic, Vector3_)
    try:
        subscriber.Init(command_handler, args.queue_length)
        logger.info("Aloha hand DDS bridge initialized. Waiting for commands...")
    except Exception as exc:
        logger.error(f"Failed to initialize DDS subscriber: {exc}")
        controller.running = False
        sys.exit(1)

    # Create publisher for gripper state feedback
    state_publisher = ChannelPublisher(args.dds_state_topic, Vector3_)
    try:
        state_publisher.Init()
        logger.info(f"Gripper state publisher initialized on topic: {args.dds_state_topic}")
    except Exception as exc:
        logger.error(f"Failed to initialize DDS publisher: {exc}")
        subscriber.Close()
        controller.running = False
        sys.exit(1)

    terminate = False

    def handle_signal(signum, frame):
        nonlocal terminate
        terminate = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Feedback publishing loop
    feedback_interval = 1.0 / args.feedback_rate
    last_feedback_time = time.time()

    try:
        while not terminate:
            current_time = time.time()

            # Publish gripper state at specified rate
            if current_time - last_feedback_time >= feedback_interval:
                with dual_gripper_data_lock:
                    left_state = dual_gripper_state_array[0]
                    right_state = dual_gripper_state_array[1]

                # Create feedback message with actual gripper positions (0.0 to 0.065 range)
                state_msg = Vector3_(
                    x=float(left_state), y=float(right_state), z=0.0  # Reserved for future use
                )

                state_publisher.Write(state_msg)

                last_feedback_time = current_time

            time.sleep(0.001)  # Short sleep to prevent busy waiting
    finally:
        state_publisher.Close()
        subscriber.Close()
        controller.running = False
        logger.info("Aloha hand DDS bridge shutting down.")


if __name__ == "__main__":
    main()
