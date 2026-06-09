import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
import fcntl
import subprocess
import serial
import serial.tools.list_ports
from multiprocessing import Process, shared_memory, Array, Value, Lock
import logging
import struct

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger_mp = logging.getLogger(__name__)

ARDUINO_BAUD = 2000000  # Increased to 2M for lower latency
ARDUINO_TIMEOUT = 0.01  # Reduced timeout for faster response

# Binary protocol constants
PACKET_HEADER = 0xAA
PACKET_SIZE = 10  # Command packet: 1B header + 4B timestamp + 2B left + 2B right + 1B checksum
FEEDBACK_PACKET_SIZE = (
    14  # Compact feedback: [header(1)][timestamp(4)][left_load(2)][right_load(2)]
)
#                   [left_present(2)][right_present(2)][checksum(1)]


class Aloha_Gripper_Controller:
    def __init__(
        self,
        left_gripper_value_in,
        right_gripper_value_in,
        dual_gripper_data_lock=None,
        dual_gripper_state_out=None,
        dual_gripper_action_out=None,
        filter=True,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
    ):
        """
        Replaces DDS with Serial communication to Arduino/OpenRB + REAL FEEDBACK.
        All interface remains identical to original.
        """

        logger_mp.info("Initialize Aloha_Gripper_Controller (Serial + Feedback Version)...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if filter and not self.simulation_mode:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), 2)
        else:
            self.smooth_filter = None

        # Initialize serial connection
        self.ser = None
        self.serial_lock = threading.Lock()  # Prevent read/write collision
        self._connect_serial()

        # Shared state values
        self.left_gripper_state_value = Value("d", 0.0, lock=True)
        self.right_gripper_state_value = Value("d", 0.0, lock=True)

        # Load feedback values (0.1% units, signed)
        self.left_load_value = Value("i", 0, lock=True)
        self.right_load_value = Value("i", 0, lock=True)
        self.load_values = []
        self.load_lock = threading.Lock()

        # RTT measurement
        self.last_send_timestamp = 0
        self.rtt_values = []
        self.rtt_lock = threading.Lock()

        # Diagnostic counters
        self.feedback_packets_received = 0
        self.feedback_checksum_errors = 0

        # Start feedback reader thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # Give Arduino time to reset and initialize
        time.sleep(2)

        # Start control thread
        self.gripper_control_thread = threading.Thread(
            target=self.control_thread,
            args=(
                left_gripper_value_in,
                right_gripper_value_in,
                self.left_gripper_state_value,
                self.right_gripper_state_value,
                dual_gripper_data_lock,
                dual_gripper_state_out,
                dual_gripper_action_out,
            ),
        )
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        logger_mp.info("Initialize Aloha_Gripper_Controller (Serial + Feedback) OK!\n")

    def _usb_reset(self, port):
        """USB soft reset to recover USB-CDC TX after previous unclean close.

        When a serial port to OpenRB-150 is closed (even via os.close(fd)),
        the kernel cdc_acm driver can leave the USB-CDC TX channel disabled.
        Subsequent opens will be able to send commands but never receive data.
        A USB reset via ioctl restores normal bidirectional communication.
        """
        USBDEVFS_RESET = 21780  # 0x5514
        try:
            result = subprocess.run(
                ["udevadm", "info", "-q", "path", "-n", port],
                capture_output=True, text=True, timeout=5
            )
            sys_path = result.stdout.strip()
            usb_device_path = f"/sys{sys_path}"

            # Walk up to find the USB device with busnum/devnum
            p = usb_device_path
            for _ in range(10):
                p = os.path.dirname(p)
                busnum_path = os.path.join(p, "busnum")
                devnum_path = os.path.join(p, "devnum")
                if os.path.exists(busnum_path) and os.path.exists(devnum_path):
                    with open(busnum_path) as f:
                        busnum = int(f.read().strip())
                    with open(devnum_path) as f:
                        devnum = int(f.read().strip())
                    usb_dev = f"/dev/bus/usb/{busnum:03d}/{devnum:03d}"
                    fd = os.open(usb_dev, os.O_WRONLY)
                    fcntl.ioctl(fd, USBDEVFS_RESET, 0)
                    os.close(fd)
                    logger_mp.info(f"USB reset successful on {usb_dev}")
                    return True

            logger_mp.warning("Could not find USB device busnum/devnum for reset")
            return False
        except PermissionError:
            logger_mp.warning("USB reset requires root permission, skipping")
            return False
        except Exception as e:
            logger_mp.warning(f"USB reset failed: {e}")
            return False

    def _connect_serial(self):
        """Auto-detect and connect to Arduino/OpenRB"""
        ports = serial.tools.list_ports.comports()
        candidates = []

        KNOWN_VID_PID = {
            # OpenRB-150 (STM32F7)
            (0x0483, 0x5740),  # STMicroelectronics
            # Arduino Uno (ATmega16U2)
            (0x2341, 0x0043),
            (0x2341, 0x0001),
            # Arduino Mega2560
            (0x2341, 0x0010),
            (0x2341, 0x0057),
            # Arduino Leonardo / Micro
            (0x2341, 0x8036),
            # CH340 common USB-Serial (used in many clones)
            (0x1A86, 0x7523),
        }

        for port in ports:
            # Check by VID/PID if availables
            if port.vid and port.pid:
                if (port.vid, port.pid) in KNOWN_VID_PID:
                    candidates.append(port.device)
                    continue

            # Fallback: Check description for Arduino/OpenRB keywords
            desc = port.description.lower()
            if any(kw in desc for kw in ["arduino", "openrb", "ch340", "usb serial", "stm32"]):
                candidates.append(port.device)

        if not candidates:
            logger_mp.error("No Arduino/OpenRB found!")
            raise RuntimeError("No compatible serial device found.")

        for port in candidates:
            try:
                # USB reset to recover USB-CDC TX after previous unclean close.
                # Without this, OpenRB-150 may accept commands but never send feedback.
                if self._usb_reset(port):
                    # Wait for device to re-enumerate and Arduino to recalibrate
                    logger_mp.info("Waiting for Arduino to recalibrate after USB reset...")
                    time.sleep(2)
                    # Device name may have changed, re-scan
                    for _ in range(15):
                        if os.path.exists(port):
                            break
                        time.sleep(1)
                    else:
                        logger_mp.warning(f"{port} did not reappear after USB reset")
                    # Wait for calibration (LED blinks 3 times when done)
                    time.sleep(8)
                    logger_mp.info("USB reset recovery complete")

                # Open serial with specific settings to ensure clean USB-CDC state
                self.ser = serial.Serial()
                self.ser.port = port
                self.ser.baudrate = ARDUINO_BAUD
                self.ser.timeout = ARDUINO_TIMEOUT
                # Don't set DTR/RTS before open - let the system handle it
                self.ser.open()

                # Now explicitly set DTR high to signal we're connected
                # This is required for USB-CDC to enable TX from Arduino
                self.ser.dtr = True
                self.ser.rts = True
                time.sleep(0.1)

                # Clear any stale data in buffer
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()

                logger_mp.info(f"Connected to Arduino on {port}")
                print(f"Connected to Arduino on {port}")
                return
            except Exception as e:
                logger_mp.warning(f"Failed to open {port}: {e}")
                continue

        raise RuntimeError("Could not connect to any candidate port.")

    def _subscribe_gripper_state(self):
        """READ BINARY FEEDBACK FROM ARDUINO with PID debug data"""
        buffer = bytearray()
        last_log_time = time.time()
        no_data_logged = False
        read_attempts = 0

        while True:
            try:
                # Log if no data for 2 seconds
                current_time = time.time()
                read_attempts += 1

                # Debug: periodically log serial status
                if read_attempts % 1000 == 0:
                    logger_mp.info(
                        f"Serial status check: port={self.ser.port}, is_open={self.ser.is_open}, in_waiting={self.ser.in_waiting}"
                    )

                if self.ser.in_waiting < FEEDBACK_PACKET_SIZE:
                    if not no_data_logged and (current_time - last_log_time) > 2.0:
                        logger_mp.warning(
                            f"No feedback data from Arduino (in_waiting={self.ser.in_waiting} bytes, need {FEEDBACK_PACKET_SIZE})"
                        )
                        no_data_logged = True
                        last_log_time = current_time

                # Read all available data to prevent buffer overflow
                # This is more robust than waiting for exactly FEEDBACK_PACKET_SIZE bytes
                bytes_available = self.ser.in_waiting
                if bytes_available > 0:
                    no_data_logged = False
                    with self.serial_lock:
                        data = self.ser.read(bytes_available)
                    buffer.extend(data)

                    # Search for packet header
                    while len(buffer) >= FEEDBACK_PACKET_SIZE:
                        header_idx = buffer.find(bytes([PACKET_HEADER]))
                        if header_idx == -1:
                            buffer.clear()
                            break

                        if header_idx > 0:
                            buffer = buffer[header_idx:]

                        if len(buffer) < FEEDBACK_PACKET_SIZE:
                            break

                        packet = buffer[:FEEDBACK_PACKET_SIZE]

                        # Verify checksum
                        checksum = sum(packet[:-1]) & 0xFF
                        if checksum != packet[-1]:
                            self.feedback_checksum_errors += 1
                            if self.feedback_checksum_errors % 100 == 1:
                                logger_mp.warning(
                                    f"Checksum error #{self.feedback_checksum_errors} (calculated={checksum}, received={packet[-1]})"
                                )
                            # On checksum error, skip only 1 byte and search for next header
                            # This handles cases where the real header is within the "packet"
                            buffer = buffer[1:]
                            continue

                        # Checksum passed - consume this packet from buffer
                        buffer = buffer[FEEDBACK_PACKET_SIZE:]

                        # Parse compact feedback packet: [header][timestamp][left_load][right_load][left_present][right_present][checksum]
                        try:
                            # Packet layout: B (header), I (timestamp), 2x h (load signed), 2x H (positions unsigned), B (checksum) = 13 bytes
                            data = struct.unpack("<BIhhHHB", bytes(packet))
                            _, timestamp, left_load, right_load, left_present, right_present, _ = (
                                data
                            )

                            # Calculate RTT
                            current_time_ms = int(time.time() * 1000) & 0xFFFFFFFF
                            rtt_ms = (current_time_ms - timestamp) & 0xFFFFFFFF
                            if rtt_ms < 1000:  # Filter out invalid values
                                with self.rtt_lock:
                                    self.rtt_values.append(rtt_ms)
                                    if len(self.rtt_values) > 100:
                                        self.rtt_values.pop(0)

                            # Map positions to internal range (use present position)
                            # Arduino now sends standardized positions: 0 (open) to 1000 (closed)
                            mapped_left = np.interp(left_present, [0, 1000], [0.0, 0.065])
                            mapped_right = np.interp(right_present, [0, 1000], [0.0, 0.065])

                            with self.left_gripper_state_value.get_lock():
                                self.left_gripper_state_value.value = mapped_left
                            with self.right_gripper_state_value.get_lock():
                                self.right_gripper_state_value.value = mapped_right

                            # Log first successful packet
                            self.feedback_packets_received += 1
                            if self.feedback_packets_received == 1:
                                logger_mp.info(
                                    f"First feedback packet received! Left={mapped_left:.4f}, Right={mapped_right:.4f}"
                                )
                            elif self.feedback_packets_received % 100 == 0:
                                logger_mp.info(
                                    f"Feedback packets: {self.feedback_packets_received}, Checksum errors: {self.feedback_checksum_errors}"
                                )

                            # Store load values
                            with self.left_load_value.get_lock():
                                self.left_load_value.value = left_load
                            with self.right_load_value.get_lock():
                                self.right_load_value.value = right_load

                            with self.load_lock:
                                self.load_values.append((left_load, right_load))
                                if len(self.load_values) > 100:
                                    self.load_values.pop(0)

                        except struct.error:
                            continue
                else:
                    time.sleep(0.0002)  # Minimal sleep to reduce CPU load
            except Exception as e:
                logger_mp.error(f"Error in feedback thread: {e}")
                time.sleep(0.01)

    def ctrl_dual_gripper(self, dual_gripper_action):
        """Send gripper commands via binary packet to Arduino"""
        left_action, right_action = dual_gripper_action

        # Map internal action (0.0-0.065) to standardized position range (0-1000)
        # Arduino expects: 0 (open) to 1000 (closed)
        # Arduino will internally map to calibrated MIN_POS/MAX_POS for each gripper
        left_pos = int(np.clip(np.interp(left_action, [0.0, 0.065], [0, 1000]), 0, 1000))
        right_pos = int(np.clip(np.interp(right_action, [0.0, 0.065], [0, 1000]), 0, 1000))

        # Create timestamp (milliseconds, 32-bit)
        timestamp_ms = int(time.time() * 1000) & 0xFFFFFFFF

        try:
            # Build binary packet: [0xAA][timestamp(4)][left(2)][right(2)][checksum(1)]
            packet = struct.pack("<BIHH", PACKET_HEADER, timestamp_ms, left_pos, right_pos)
            checksum = sum(packet) & 0xFF
            packet += struct.pack("B", checksum)

            with self.serial_lock:
                self.ser.write(packet)

            self.last_send_timestamp = timestamp_ms
        except Exception as e:
            logger_mp.error(f"Serial write error: {e}")

    def get_rtt_stats(self):
        """Get RTT statistics in milliseconds"""
        with self.rtt_lock:
            if not self.rtt_values:
                return None
            return {
                "min": min(self.rtt_values),
                "max": max(self.rtt_values),
                "avg": sum(self.rtt_values) / len(self.rtt_values),
                "count": len(self.rtt_values),
            }

    def get_load_stats(self):
        """Get load statistics (0.1% units)"""
        with self.load_lock:
            if not self.load_values:
                return None
            left_loads = [c[0] for c in self.load_values]
            right_loads = [c[1] for c in self.load_values]
            return {
                "left_min": min(left_loads),
                "left_max": max(left_loads),
                "left_avg": sum(left_loads) / len(left_loads),
                "right_min": min(right_loads),
                "right_max": max(right_loads),
                "right_avg": sum(right_loads) / len(right_loads),
                "count": len(self.load_values),
            }

    def control_thread(
        self,
        left_gripper_value_in,
        right_gripper_value_in,
        left_gripper_state_value,
        right_gripper_state_value,
        dual_hand_data_lock=None,
        dual_gripper_state_out=None,
        dual_gripper_action_out=None,
    ):
        self.running = True

        DELTA_GRIPPER_CMD = 0.065
        THUMB_INDEX_DISTANCE_MIN = 0
        THUMB_INDEX_DISTANCE_MAX = 0.065
        LEFT_MAPPED_MIN = 0.0
        RIGHT_MAPPED_MIN = 0.0
        LEFT_MAPPED_MAX = LEFT_MAPPED_MIN + 0.0650
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 0.0650

        # Initial target actions, set the gripper to the maximum open position
        left_target_action = LEFT_MAPPED_MAX - LEFT_MAPPED_MIN
        right_target_action = RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN

        last_debug_print = time.time()

        try:
            while self.running:
                start_time = time.time()

                with left_gripper_value_in.get_lock():
                    left_gripper_value = left_gripper_value_in.value
                with right_gripper_value_in.get_lock():
                    right_gripper_value = right_gripper_value_in.value

                if left_gripper_value != 0.0 or right_gripper_value != 0.0:
                    left_target_action = np.interp(
                        left_gripper_value,
                        [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX],
                        [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX],
                    )
                    right_target_action = np.interp(
                        right_gripper_value,
                        [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX],
                        [RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX],
                    )

                # Get REAL state from Arduino feedback (updated by _subscribe_gripper_state)
                # The values in left_gripper_state_value and right_gripper_state_value are already mapped to 0.0-0.065
                with left_gripper_state_value.get_lock():
                    current_left_state = left_gripper_state_value.value
                with right_gripper_state_value.get_lock():
                    current_right_state = right_gripper_state_value.value

                dual_gripper_state = np.array([current_left_state, current_right_state])
                # print("action and state1: \n")
                # print(left_target_action, "\n")
                # print(right_target_action, "\n")

                # Smooth and clip based on REAL feedback
                left_actual_action = np.clip(
                    left_target_action,
                    dual_gripper_state[0] - DELTA_GRIPPER_CMD,
                    dual_gripper_state[0] + DELTA_GRIPPER_CMD,
                )
                right_actual_action = np.clip(
                    right_target_action,
                    dual_gripper_state[1] - DELTA_GRIPPER_CMD,
                    dual_gripper_state[1] + DELTA_GRIPPER_CMD,
                )

                dual_gripper_action = np.array([left_actual_action, right_actual_action])

                if self.smooth_filter:
                    self.smooth_filter.add_data(dual_gripper_action)
                    dual_gripper_action = self.smooth_filter.filtered_data

                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array(
                            [LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN]
                        )
                        dual_gripper_action_out[:] = dual_gripper_action - np.array(
                            [LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN]
                        )

                self.ctrl_dual_gripper(dual_gripper_action)

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger_mp.error(f"Control thread error: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()
            logger_mp.info("Aloha_Gripper_Controller (Serial + Feedback) has been closed.")

    def __del__(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


class Gripper_JointIndex(IntEnum):
    kGripper = 0


# ========== MOCK DEPENDENCIES FOR STANDALONE TESTING ========== #
class WeightedMovingFilter:
    def __init__(self, weights, size):
        self.weights = np.array(weights)
        self.size = size
        self.buffer = np.zeros((len(weights), size))

    def add_data(self, data):
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = data

    @property
    def filtered_data(self):
        return np.average(self.buffer, weights=self.weights, axis=0)


# ========== STANDALONE TEST ========== #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aloha Gripper Controller")
    parser.add_argument("--xr-mode", type=str, choices=["hand", "controller"], default="controller")
    parser.add_argument("--ee", type=str, choices=["aloha"], default="aloha")
    args = parser.parse_args()

    # Mock shared memory
    class MockArray:
        def __init__(self, size):
            self.arr = [0.0] * size

        def __setitem__(self, key, value):
            self.arr[key] = value

        def __getitem__(self, key):
            return self.arr[key]

        def __len__(self):
            return len(self.arr)

    class MockValue:
        def __init__(self, val):
            self.value = val

        def get_lock(self):
            return threading.Lock()

    # Use real Value/Array if not mocking
    try:
        left_gripper_value = Value("d", 0.0, lock=True)
        right_gripper_value = Value("d", 0.0, lock=True)
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array("d", 2, lock=False)
        dual_gripper_action_array = Array("d", 2, lock=False)
    except Exception:
        # Fallback for environments without multiprocessing (e.g., some IDEs)
        left_gripper_value = MockValue(0.06)
        right_gripper_value = MockValue(0.06)
        dual_gripper_data_lock = threading.Lock()
        dual_gripper_state_array = MockArray(2)
        dual_gripper_action_array = MockArray(2)

    # Initialize controller
    gripper_ctrl = Aloha_Gripper_Controller(
        left_gripper_value,
        right_gripper_value,
        dual_gripper_data_lock,
        dual_gripper_state_array,
        dual_gripper_action_array,
        simulation_mode=False,
    )

    print("Controller initialized. Enter 's' to start sending commands...")
    user_input = input()
    if user_input.lower() == "s":
        try:
            i = 0
            last_stats_print = time.time()
            while True:
                # Simulate XR input: cycling between open/close
                # Slower cycle: period = 2*pi / 0.01 * 0.01s = ~6.28 seconds per full cycle
                val = 0 + 0.065 * (0.5 + 0.5 * np.sin(i * 0.01))  # 0 to 0.065, ~6 seconds per cycle
                left_gripper_value.value = val
                right_gripper_value.value = val

                # Print RTT and load statistics every 0.2 seconds
                current_time = time.time()
                if current_time - last_stats_print >= 0.2:
                    stats = gripper_ctrl.get_rtt_stats()
                    if stats:
                        print(
                            f"RTT Stats: Min={stats['min']:.1f}ms, Max={stats['max']:.1f}ms, "
                            f"Avg={stats['avg']:.1f}ms, Count={stats['count']}"
                        )

                    # Load statistics (0.1% units, signed)
                    load_stats = gripper_ctrl.get_load_stats()
                    if load_stats:
                        print(
                            f"Load: Left={load_stats['left_avg']:.1f} (Max={load_stats['left_max']:.1f}), "
                            f"Right={load_stats['right_avg']:.1f} (Max={load_stats['right_max']:.1f}) [0.1% units]"
                        )

                    last_stats_print = current_time

                time.sleep(0.01)
                i += 1
        except KeyboardInterrupt:
            print("\nStopping...")
