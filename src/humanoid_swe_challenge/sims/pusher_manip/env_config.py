import numpy as np

class PusherManipEnvCfg():
    #Sim
    dt = 1/120
    decimation = 5
    max_step_count = 1000

    #pusher
    pusher_body_name = "pusher"
    pusher_joint_x_name = "pusher_x"
    pusher_joint_y_name = "pusher_y"
    pusher_joint_z_name = "pusher_z"
    pusher_radius = 0.0125

    #goal
    goal_r_body_name = "goal_r"
    goal_g_body_name = "goal_g"
    goal_b_body_name = "goal_b"
    random_goal_pose = True
    random_goal_pose_low = np.array([-0.3, -0.3, 0.05])
    random_goal_pose_high = np.array([0.3, 0.3, 0.65])
