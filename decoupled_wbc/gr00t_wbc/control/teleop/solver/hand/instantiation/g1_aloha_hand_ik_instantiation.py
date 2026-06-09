"""Instantiation function for G1 ALOHA gripper IK solvers."""

from typing import Tuple

from gr00t_wbc.control.teleop.solver.hand.g1_aloha_gripper_ik_solver import (
    G1AlohaGripperInverseKinematicsSolver,
)


def instantiate_g1_aloha_hand_ik_solver() -> Tuple[
    G1AlohaGripperInverseKinematicsSolver, G1AlohaGripperInverseKinematicsSolver
]:
    """Instantiate ALOHA gripper IK solvers for left and right hands.

    Returns:
        Tuple containing left and right ALOHA gripper IK solvers.
    """
    left_hand_ik_solver = G1AlohaGripperInverseKinematicsSolver(side="left")
    right_hand_ik_solver = G1AlohaGripperInverseKinematicsSolver(side="right")
    return left_hand_ik_solver, right_hand_ik_solver
