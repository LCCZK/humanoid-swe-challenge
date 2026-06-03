import numpy as np

class BoxPushingEnvCfg():
    #Sim
    dt = 1/120
    decimation = 5
    max_step_count = 1000

    #pusher
    pusher_body_name = "pusher"
    pusher_joint_x_name = "pusher_x"
    pusher_joint_y_name = "pusher_y"
    pusher_joint_z_name = "pusher_z"

    #box
    box_body_name = "box"
    box_random_init_pos = False

    #goal
    goal_r_body_name = "goal_r"
    goal_g_body_name = "goal_g"
    goal_b_body_name = "goal_b"
    goal_pos_only = False
    random_goal_pose = False
    random_goal_pose_range = []

    #Env configs
    # history_lenth = 1
    
    #Noise config
    # pusher_xy_std = [0, 0]
    # box_xy_std = [0, 0]
    # goal_xy_std = [0, 0]
    # box_yaw_std = 0
    # goal_yaw_std = 0

    # Goal random range
    # workspace_x = [-1.0 * (table_size[0] / 2.0), table_size[0] / 2.0]
    # workspace_y = [-1.0 * (table_size[1] / 2.0), table_size[1] / 2.0]

    # goal_max_x = workspace_x[1] - (box_size[0] / 2.0) - 0.05
    # goal_min_x = workspace_x[0] + (box_size[0] / 2.0) + 0.05
    # goal_max_y = workspace_y[1] - (box_size[1] / 2.0) - 0.05
    # goal_min_y = workspace_y[0] + (box_size[1] / 2.0) + 0.05
    # goal_max_yaw = np.pi/2
    # goal_min_yaw = -np.pi/2