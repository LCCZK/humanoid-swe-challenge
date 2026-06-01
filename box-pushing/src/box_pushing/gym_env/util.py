import numpy as np

def quat_to_yaw(quat:np.ndarray):
    t3 = +2.0 * (quat[0] * quat[3] + quat[1] * quat[2])
    t4 = 1.0 - 2.0 * (quat[2]*quat[2] + quat[3] * quat[3])  
    return np.arctan2(t3, t4)

def yaw_to_quaternion(yaw):
    """Convert a batch of 2D yaw angles (in radians) to quaternions (w, x, y, z)."""
    w = np.cos(yaw / 2)
    z = np.sin(yaw / 2)
    x = 0  # No roll or pitch in 2D
    y = 0
    return np.array([w, x, y, z])  # Shape (num_envs, 4)