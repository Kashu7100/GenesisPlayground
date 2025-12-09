from gs_env.sim.envs.config.schema import (
    EnvArgs,
    GenesisInitArgs,
    LeggedRobotEnvArgs,
    MotionEnvArgs,
    WalkingEnvArgs,
)
from gs_env.sim.objects.config.registry import ObjectArgsRegistry
from gs_env.sim.robots.config.registry import RobotArgsRegistry
from gs_env.sim.scenes.config.registry import SceneArgsRegistry
from gs_env.sim.sensors.config.registry import SensorArgsRegistry

# ------------------------------------------------------------
# Genesis init
# ------------------------------------------------------------


GenesisInitArgsRegistry: dict[str, GenesisInitArgs] = {}


GenesisInitArgsRegistry["default"] = GenesisInitArgs(
    seed=0,
    precision="32",
    logging_level="info",
    backend=None,
)


# ------------------------------------------------------------
# Manipulation
# ------------------------------------------------------------


EnvArgsRegistry: dict[str, EnvArgs] = {}

EnvArgsRegistry["goal_reach_default"] = EnvArgs(
    env_name="GoalReachingEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["franka_default"],
    objects_args=[ObjectArgsRegistry["box_default"]],
    sensors_args=[
        SensorArgsRegistry["oak_camera_default"],
        SensorArgsRegistry["ee_link_pos"],
        SensorArgsRegistry["ee_link_quat"],
        SensorArgsRegistry["joint_angles"],
        SensorArgsRegistry["gripper_width"],
    ],
    reward_term="reward",
    reward_args={
        "ActionL2Penalty": 0.0,
        "KeypointsAlign": 1.0,
    },
    img_resolution=(480, 270),
)


EnvArgsRegistry["pick_cube_default"] = EnvArgs(
    env_name="GoalReachingEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["franka_teleop"],
    objects_args=[ObjectArgsRegistry["box_default"]],
    sensors_args=[],
    reward_args={},
    img_resolution=(480, 270),
)

# ------------------------------------------------------------
# G1 Configuration
# ------------------------------------------------------------

EnvArgsRegistry["g1_walk"] = WalkingEnvArgs(
    env_name="WalkingEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_legged"],
    robot_args=RobotArgsRegistry["g1_default"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1",
    reward_args={
        ### Velocity Tracking ###
        "LinVelXYReward": 10.0,
        "AngVelZReward": 10.0,
        "LinVelZPenalty": 20.0,
        "AngVelXYPenalty": 1.0,
        ### Pose Tracking ###
        "OrientationPenalty": 100.0,
        ### Regularization ###
        "TorquePenalty": 0.0005,
        "ActionRatePenalty": 0.3,
        "DofPosLimitPenalty": 100.0,
        "DofVelPenalty": 0.05,
        "G1BaseHeightPenalty": 100.0,
        "ActionLimitPenalty": 0.1,
        ### Motion Constraints ###
        "AnkleTorquePenalty": 0.001,
        "HipYawPenalty": 100.0,
        "HipRollPenalty": 100.0,
        # "HipYawVelPenalty": 0.05,
        # "HipRollVelPenalty": 0.05,
        # "HipTorquePenalty": 0.001,
        # "HipPositionPenalty": 10.0,
        "UpperBodyDofPenalty": 1,
        "UpperBodyActionPenalty": 0.5,
        "WaistDofPenalty": 300.0,
        # "FeetAirTimeReward": 200.0,
        "FeetAirTimePenalty": 500.0,
        "G1FeetHeightPenalty": 100.0,
        "G1FeetContactForcePenalty": 30.0,
        "FeetZVelocityPenalty": 30.0,
        "FeetOrientationPenalty": 30.0,
        "StandStillFeetContactPenalty": 1.0e-05,
        "StandStillActionRatePenalty": 0.1,
        "StandStillReward": 20.0,
        # "StandStillAnkleTorquePenalty": 0.01,
        "G1FeetContactForceLimitPenalty": 1e-4,
    },
    img_resolution=(480, 270),
    action_latency=0,
    obs_history_len=1,
    obs_scales={
        "dof_vel": 0.1,
        "base_ang_vel": 0.5,
        "foot_contact_force": 0.001,
    },
    obs_noises={
        "dof_pos": 0.01,
        "dof_vel": 0.02,
        "projected_gravity": 0.05,
        "base_ang_vel": 0.2,
    },
    actor_obs_terms=[
        "last_action",
        "dof_pos",
        "dof_vel",
        "projected_gravity",
        "base_ang_vel",
        "commands",
    ],
    critic_obs_terms=[
        "last_action",
        "dof_pos",
        "dof_vel",
        "projected_gravity",
        "base_lin_vel",
        "base_ang_vel",
        "commands",
        "feet_height",
        "foot_contact_force",
    ],
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
    command_resample_time=10.0,
    commands_range=(
        (-1.0, 1.0),  # Forward/Backward
        (0.0, 0.0),  # Left/Right
        (-1.0, 1.0),  # Turn
    ),
)


EnvArgsRegistry["g1_motion_teacher"] = MotionEnvArgs(
    env_name="MotionEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["custom_scene_g1_mocap"],
    robot_args=RobotArgsRegistry["g1_default"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1",
    reward_args={
        ### Motion Tracking ###
        "DofPosReward": 3.0,
        "DofVelReward": 0.02,
        # "BaseHeightReward": 1.0,
        "BasePosReward": 200.0,
        "BaseHeightReward": 100.0,
        "BaseQuatReward": 20.0,
        "BaseLinVelReward": 10.0,
        "BaseAngVelReward": 1.0,
        "TrackingLinkPosReward": 30.0,
        "TrackingLinkQuatReward": 1.0,
        "TrackingLinkLinVelReward": 0.5,
        "TrackingLinkAngVelReward": 0.02,
        "TrackingFootContactReward": 1.0,
        ### Regularization ###
        "TorquePenalty": 0.0001,
        "ActionRatePenalty": 0.3,
        "DofPosLimitPenalty": 10.0,
        "ActionLimitPenalty": 0.1,
        "AnkleTorquePenalty": 0.003,
        "BodyAngVelXYPenalty": 1.0,
        "WaistVelPenalty": 0.5,
        # "G1FeetContactForceLimitPenalty": 1e-4,
        "MotionFeetAirTimePenalty": 200.0,
        "G1FeetSlidePenalty": 5.0,
    },
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
    obs_scales={
        "dof_vel": 0.1,
        "diff_dof_vel": 0.1,
    },
    obs_noises={
        # "dof_pos": 0.01,
        # "dof_vel": 0.2,
        # "diff_dof_pos": 0.01,
        # "diff_dof_vel": 0.2,
        # "projected_gravity": 0.05,
        # "base_ang_vel_local": 0.2,
        # "diff_base_ang_vel_local": 0.2,
        # "base_lin_vel_local": 0.3,
    },
    actor_obs_terms=[
        "last_action",
        # Proprioception
        "dof_pos",
        "dof_vel",
        "base_euler",
        "base_ang_vel_local",
        "base_rotation_6D",
        "projected_gravity",
        # Motion Difference
        "diff_dof_pos",
        "diff_dof_vel",
        "diff_base_rotation_6D",
        "diff_base_euler",
        "diff_base_ang_vel_local",
        "diff_base_pos_local_yaw",
        "diff_base_lin_vel_local",
        "diff_tracking_link_pos_local_yaw",
        "diff_tracking_link_rotation_6D",
        "diff_tracking_link_lin_vel_local_yaw",
        "diff_tracking_link_ang_vel_local_yaw",
        # Reference
        "motion_obs",
        # Privileged
        "dr_obs",
        "base_lin_vel_local",
        "tracking_link_pos_local_yaw",
        "foot_contact_weighted",
    ],
    critic_obs_terms=[
        "last_action",
        # Proprioception
        "dof_pos",
        "dof_vel",
        "base_euler",
        "base_ang_vel_local",
        "base_rotation_6D",
        "projected_gravity",
        # Motion Difference
        "diff_dof_pos",
        "diff_dof_vel",
        "diff_base_rotation_6D",
        "diff_base_euler",
        "diff_base_ang_vel_local",
        "diff_base_pos_local_yaw",
        "diff_base_lin_vel_local",
        "diff_tracking_link_pos_local_yaw",
        "diff_tracking_link_rotation_6D",
        "diff_tracking_link_lin_vel_local_yaw",
        "diff_tracking_link_ang_vel_local_yaw",
        # Reference
        "motion_obs",
        # Privileged
        "dr_obs",
        "base_lin_vel_local",
        "tracking_link_pos_local_yaw",
        "foot_contact_weighted",
    ],
    reset_yaw_range=(-0.15, 0.15),
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
    tracking_link_names=[
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "torso_link",
        "pelvis",
    ],
    dof_weights={
        "hip": 1.0,
        "knee": 0.6,
        "ankle": 0.3,
        "waist": 2.0,
        "shoulder": 1.0,
        "elbow": 0.6,
        "wrist": 0.3,
    },
    link_weights={
        "ankle": 1.0,
        "wrist": 2.0,
        "torso": 2.0,
    },
    no_terminate_before_motion_time=1.0,
    no_terminate_after_reset_time=2.0,
    no_terminate_after_random_push_time=2.0,
    # [initial_threshold, [min_threshold, max_threshold]]
    terminate_after_error={
        "base_pos_error": [1.0, [0.1, 1.0]],
        # "base_height_error": [0.2, [0.05, 0.2]],
        "base_quat_error": [2.0, [0.1, 2.0]],
        # "dof_pos_error": [6.0, [1.0, 6.0]],
        # "tracking_link_pos_error": [0.7, [0.03, 1.0]],
        # "foot_contact_force_error": [500.0, [50.0, 500.0]],
    },
    adaptive_termination_ratio=None,
    motion_file=None,
    observed_steps={
        "base_pos": [1, 2, 3, 4, 5, 6, 7, 8],
        "base_quat": [1, 2, 3, 4, 5, 6, 7, 8],
        "base_lin_vel": [1, 2, 3, 4, 5, 6, 7, 8],
        "base_ang_vel": [1, 2, 3, 4, 5, 6, 7, 8],
        "base_ang_vel_local": [1, 2, 3, 4, 5, 6, 7, 8],
        "dof_pos": [1, 2, 3],
        "dof_vel": [1, 2, 3],
        "link_pos_local": [1, 2, 3],
        "link_quat_local": [1, 2, 3],
        "link_lin_vel": [1, 2, 3],
        "link_ang_vel": [1, 2, 3],
        "foot_contact": [1, 2, 3, 4, 5, 6, 7, 8],
    },
)


EnvArgsRegistry["g1_motion"] = MotionEnvArgs(
    env_name="MotionEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["custom_scene_g1_mocap"],
    robot_args=RobotArgsRegistry["g1_default"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1",
    reward_args={
        ### Motion Tracking ###
        "DofPosReward": 3.0,
        "DofVelReward": 0.05,
        # "BaseHeightReward": 1.0,
        "BaseQuatReward": 30.0,
        "BaseLinVelReward": 30.0,
        "BaseAngVelReward": 1.0,
        "TrackingLinkPosReward": 60.0,
        "TrackingLinkQuatReward": 3.0,
        "FootContactForceReward": 4e-4,
        ### Regularization ###
        "TorquePenalty": 0.0001,
        "ActionRatePenalty": 0.2,
        "DofPosLimitPenalty": 10.0,
        "ActionLimitPenalty": 0.1,
        "AnkleTorquePenalty": 0.003,
        "BodyAngVelXYPenalty": 1.0,
        "WaistVelPenalty": 0.5,
        "G1FeetContactForceLimitPenalty": 1e-4,
        "G1FeetSlidePenalty": 5.0,
    },
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
    obs_scales={
        "dof_vel": 0.1,
        "diff_dof_vel": 0.1,
    },
    obs_noises={
        # "dof_pos": 0.01,
        # "dof_vel": 0.2,
        # "diff_dof_pos": 0.01,
        # "diff_dof_vel": 0.2,
        # "projected_gravity": 0.05,
        # "base_ang_vel_local": 0.2,
        # "diff_base_ang_vel_local": 0.2,
        # "base_lin_vel_local": 0.3,
    },
    actor_obs_terms=[
        "last_action",
        # Proprioception
        "dof_pos",
        "dof_vel",
        "base_ang_vel_local",
        "base_rotation_6D",
        "projected_gravity",
        # Reference
        "motion_obs",
    ],
    critic_obs_terms=[
        "last_action",
        # Proprioception
        "dof_pos",
        "dof_vel",
        "base_euler",
        "base_ang_vel_local",
        "base_rotation_6D",
        "projected_gravity",
        # Motion Difference
        "diff_dof_pos",
        "diff_dof_vel",
        "diff_base_rotation_6D",
        "diff_base_euler",
        "diff_base_ang_vel_local",
        "diff_base_pos_local_yaw",
        "diff_base_lin_vel_local",
        "diff_tracking_link_pos_local_yaw",
        "diff_tracking_link_rotation_6D",
        "diff_tracking_link_lin_vel_local_yaw",
        "diff_tracking_link_ang_vel_local_yaw",
        # Reference
        "motion_obs",
        # Privileged
        "dr_obs",
        "base_lin_vel_local",
        "tracking_link_pos_local_yaw",
        "foot_contact_weighted",
    ],
    reset_yaw_range=(-0.15, 0.15),
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
    tracking_link_names=[
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "torso_link",
        "pelvis",
    ],
    dof_weights={
        "hip": 1.0,
        "knee": 0.6,
        "ankle": 0.3,
        "waist": 2.0,
        "shoulder": 1.0,
        "elbow": 0.6,
        "wrist": 0.3,
    },
    link_weights={
        "ankle": 1.0,
        "wrist": 1.0,
        "torso": 1.0,
    },
    no_terminate_before_motion_time=1.0,
    no_terminate_after_reset_time=2.0,
    no_terminate_after_random_push_time=2.0,
    # [initial_threshold, [min_threshold, max_threshold]]
    terminate_after_error={
        # "base_pos_error": [1.0, [0.1, 1.0]],
        # "base_height_error": [0.2, [0.05, 0.2]],
        "base_quat_error": [1.0, [0.1, 1.0]],
        # "dof_pos_error": [6.0, [1.0, 6.0]],
        # "tracking_link_pos_error": [0.5, [0.03, 1.0]],
        # "foot_contact_force_error": [500.0, [50.0, 500.0]],
    },
    adaptive_termination_ratio=None,
    motion_file=None,
    observed_steps={
        "base_pos": [
            1,
        ],
        "base_quat": [
            1,
        ],
        "base_lin_vel": [
            1,
        ],
        "base_ang_vel": [
            1,
        ],
        # "dof_pos": [1,],
        # "dof_vel": [1,],
        "link_pos_local": [
            1,
        ],
        "link_quat_local": [
            1,
        ],
        "foot_contact": [
            1,
        ],
    },
)


EnvArgsRegistry["g1_fixed"] = LeggedRobotEnvArgs(
    env_name="WalkingEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_legged"],
    robot_args=RobotArgsRegistry["g1_fixed"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1_no_waist",
    reward_args={},
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
    obs_scales={},
    obs_noises={},
    actor_obs_terms=[],
    critic_obs_terms=[],
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
)


EnvArgsRegistry["custom_g1_mocap"] = LeggedRobotEnvArgs(
    env_name="CustomEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["custom_scene_g1_mocap"],
    robot_args=RobotArgsRegistry["g1_fixed"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1_no_waist",
    reward_args={},
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
    obs_scales={},
    obs_noises={},
    actor_obs_terms=[],
    critic_obs_terms=[],
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
)
