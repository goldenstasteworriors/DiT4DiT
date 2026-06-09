"""ALOHA gripper IK solver for G1 robot.

This solver maps hand gestures to ALOHA gripper positions (1 DOF per gripper).
The ALOHA gripper is a simple parallel jaw gripper with 1 actuated DOF.

NOTE: This IK solver is currently NOT USED for Pico VR controller teleoperation.
When using Pico controllers, g1_env.py has a shortcut path that directly maps
trigger values to hardware commands, bypassing this IK solver entirely.

This IK solver is only used for:
- LeapMotion hand tracking (pinch gesture detection)
- Policy replay where gripper commands come from recorded trajectories

Output convention for Arduino hardware:
- Range: 0.0 (closed) to 0.065 (open)  [Arduino interprets inversely!]
- Trigger released (0.0) → outputs 0.065 → gripper opens
- Trigger pressed (1.0) → outputs 0.0 → gripper closes
"""

from typing import Any, Dict, Optional

import numpy as np

from gr00t_wbc.control.teleop.solver.solver import Solver


class G1AlohaGripperInverseKinematicsSolver(Solver):
    """IK solver for G1 ALOHA gripper (1 DOF prismatic gripper).

    Maps hand tracking data to gripper position commands in hardware convention.
    Output range: 0.0 (open) to 0.065 (closed).
    """

    # Hardware convention range
    HW_MIN = 0.0    # Fully open
    HW_MAX = 0.065  # Fully closed

    def __init__(self, side: str) -> None:
        self.side: str = "L" if side.lower() == "left" else "R"
        # Start at fully open position (hardware convention)
        self.current_position: float = self.HW_MIN

    def register_robot(self, robot: Any) -> None:
        pass

    def __call__(self, finger_data: Optional[Dict[str, Any]]) -> np.ndarray:
        """Compute gripper position from finger data.

        Args:
            finger_data: Dictionary with finger tracking data from hand controller
                        Can contain:
                        - "trigger": continuous value 0.0-1.0 (Pico/VR controllers)
                        - "position": 25x4x4 array for hand tracking (LeapMotion)

        Returns:
            np.ndarray: Single element array with gripper position (0.0 to 0.065, hardware convention)
        """
        if finger_data is None:
            # No hand data, return current position
            return np.array([self.current_position])

        # Priority 1: Use trigger value if available (Pico/VR controllers)
        if "trigger" in finger_data:
            trigger_value = finger_data["trigger"]
            # Inverted mapping for Arduino hardware (0.065=open, 0.0=closed):
            # - Trigger released (0.0) → send 0.065 → gripper opens
            # - Trigger pressed (1.0) → send 0.0 → gripper closes
            self.current_position = self.HW_MAX * (1.0 - trigger_value)
            return np.array([self.current_position])

        # Priority 2: Use pinch detection from hand tracking data (LeapMotion)
        if "position" not in finger_data:
            # No valid data, return current position
            return np.array([self.current_position])

        fingertips = finger_data["position"]

        # Extract X, Y, Z positions of fingertips from the transformation matrices
        positions = np.array([finger[:3, 3] for finger in fingertips])
        positions = np.reshape(positions, (-1, 3))  # Ensure 2D array with shape (N, 3)

        # Get thumb and index finger positions for pinch detection
        thumb_pos = positions[4, :]
        index_pos = positions[4 + 5, :]

        # Calculate pinch distance
        pinch_dist = np.linalg.norm(thumb_pos - index_pos)

        # Map pinch distance to gripper position (hardware convention)
        # pinch_dist < 0.03: fully closed (0.065)
        # pinch_dist > 0.10: fully open (0.0)
        min_dist = 0.03  # Closed threshold
        max_dist = 0.10  # Open threshold

        # Clamp and normalize
        normalized_dist = np.clip((pinch_dist - min_dist) / (max_dist - min_dist), 0.0, 1.0)

        # Map to hardware range (inverted: small pinch distance = closed gripper)
        # Hardware: 0.0 (open) to 0.065 (closed)
        self.current_position = self.HW_MAX * (1.0 - normalized_dist)

        return np.array([self.current_position])
