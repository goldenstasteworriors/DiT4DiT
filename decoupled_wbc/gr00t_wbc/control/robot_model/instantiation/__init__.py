from .g1 import instantiate_g1_robot_model, instantiate_g1_aloha_robot_model


def get_robot_type_and_model(robot: str, high_elbow_pose: bool = False):
    """Get the robot type and instantiate the corresponding robot model.

    Args:
        robot: Robot name string. Supported patterns:
            - "g1*" or "G1*": G1 robot with three-finger hands (43 DOF)
            - "g1_aloha*" or "G1_aloha*": G1 robot with ALOHA grippers (31 DOF)
        high_elbow_pose: Whether to use high elbow pose configuration

    Returns:
        Tuple of (robot_type, robot_model)
    """
    robot_lower = robot.lower()

    # Check for ALOHA gripper variant first (more specific)
    if "aloha" in robot_lower and robot_lower.startswith("g1"):
        return "g1_aloha", instantiate_g1_aloha_robot_model(high_elbow_pose=high_elbow_pose)

    # Standard G1 with three-finger hands
    elif robot_lower.startswith("g1"):
        return "g1", instantiate_g1_robot_model(high_elbow_pose=high_elbow_pose)

    else:
        raise ValueError(f"Invalid robot name: {robot}")
