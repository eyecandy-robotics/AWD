import time
import warnings
import json
import xml.etree.ElementTree as ET
import math

import numpy as np
import placo
import os

warnings.filterwarnings("ignore")

DT = 0.01
REFINE = 10


def extract_joint_limits_from_urdf(urdf_path):
    """
    Extract joint limits from a URDF file.
    Returns a dictionary mapping joint names to [lower_limit, upper_limit] in radians.
    """
    if not os.path.isfile(urdf_path):
        print(f"Warning: URDF file not found - {urdf_path}")
        return {}

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        joint_limits = {}
        
        for joint in root.findall('joint'):
            name = joint.attrib.get('name', 'unknown')
            joint_type = joint.attrib.get('type', 'unknown')
            
            if joint_type not in ['revolute', 'prismatic']:
                continue
                
            limit_elem = joint.find('limit')
            if limit_elem is not None:
                lower = limit_elem.attrib.get('lower')
                upper = limit_elem.attrib.get('upper')
                
                if lower is not None and upper is not None:
                    # Store limits in radians (already in radians in the URDF)
                    joint_limits[name] = [float(lower), float(upper)]
        
        return joint_limits
    except Exception as e:
        print(f"Error parsing URDF file: {e}")
        return {}


class PlacoWalkEngine:
    def __init__(
        self,
        asset_path: str = "",
        model_filename: str = "go_bdx.urdf",
        init_params: dict = {},
        ignore_feet_contact: bool = False,
        joint_limits: dict = {},
    ) -> None:
        model_filename = os.path.join(asset_path, model_filename)
        self.asset_path = asset_path
        self.model_filename = model_filename
        self.ignore_feet_contact = ignore_feet_contact
    
        robot_type = asset_path.split("/")[-1]

        # Loading the robot
        self.robot = placo.HumanoidRobot(model_filename)

        # Initialize custom attributes before loading parameters
        self.joints = []
        self.joint_angles = {}

        self.parameters = placo.HumanoidParameters()
        if init_params is not None:
            self.load_parameters(init_params)
        else:
            defaults_filename = os.path.join(asset_path, "placo_defaults.json")
            self.load_defaults(defaults_filename)

        self.head_bob = init_params.get('head_bob', False)
        self.head_bob_amplitude = init_params.get('head_bob_amplitude', 0.15)

        self.invert_neck_pitch = init_params.get('invert_neck_pitch', False)
        self.neck_pitch_sign = -1 if self.invert_neck_pitch else 1 
        
        self.tail_wiggle = init_params.get('tail_wiggle', False)
        self.tail_wiggle_amplitude = init_params.get('tail_wiggle_amplitude', 0.15)
        
        # Creating the kinematics solver
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.enable_velocity_limits(True)
        self.robot.set_velocity_limits(5.0)
        self.solver.enable_joint_limits(False)
        self.solver.dt = DT / REFINE

        # Extract joint limits from the URDF file
        urdf_joint_limits = extract_joint_limits_from_urdf(model_filename)
        
        # Merge provided joint_limits with those from URDF, with provided limits taking precedence
        if joint_limits:
            urdf_joint_limits.update(joint_limits)
            
        # Apply joint limits to all joints
        # for joint_name, limits in urdf_joint_limits.items():
        #     print(f"limits {joint_name}: {np.rad2deg(limits)}")
        #     self.robot.set_joint_limits(joint_name, limits[0], limits[1])
                        
        # Apply specific knee limits if not already set from URDF
        knee_limits = np.deg2rad([-60, 30])
        self.robot.set_joint_limits("left_knee", *knee_limits)
        self.robot.set_joint_limits("right_knee", *knee_limits)
    
        # Set default joint angles based on robot type if not provided in init_params
        # Initialize default joint positions based on robot type
        if "mini2" in robot_type:
            self.default_angles = {
                "left_hip_yaw": 0.002,
                "left_hip_roll": 0.053,
                "left_hip_pitch": -0.63,
                "left_knee": 1.368,
                "left_ankle": -0.784,
                "neck_pitch": 0.5,
                "head_pitch": -0.5,
                "head_yaw": 0,
                "head_roll": 0,
                "right_hip_yaw": -0.003,
                "right_hip_roll": -0.065,
                "right_hip_pitch": 0.635,
                "right_knee": 1.379,
                "right_ankle": -0.796,
            }
        elif "dino" in robot_type:
            self.default_angles = {
                "neck_pitch": 0,
                "head_pitch": 0,
                "head_yaw": 0,
                "tail": 0.0,
                "right_hip_yaw": 0,
                "right_hip_roll": 0,
                "right_hip_pitch": 0.19610,
                "right_knee": 0.4056,
                "right_ankle": 0.2093,
                "left_hip_yaw": 0,
                "left_hip_roll": 0,
                "left_hip_pitch": -0.1961,
                "left_knee": -0.4055,
                "left_ankle": -0.2093,
            }
        else:
            # Default for mini_bdx and go_bdx
            self.default_angles = {
                "left_hip_yaw": 0.0,
                "left_hip_roll": 0.0,
                "left_hip_pitch": 0.0,
                "left_knee": 0.0,
                "left_ankle": 0.0,
                "neck_pitch": 0.0,
                "head_pitch": 0.0,
                "head_yaw": 0.0,
                "right_hip_yaw": 0.0,
                "right_hip_roll": 0.0,
                "right_hip_pitch": 0.0,
                "right_knee": 0.0,
                "right_ankle": 0.0,
            }

        # Apply joint angles
        for joint_name in self.default_angles:
            self.robot.set_joint(joint_name, self.default_angles[joint_name])

        # self.robot.update_kinematics()
        # self.solver.solve(True)
        
        # Creating the walk QP tasks
        self.tasks = placo.WalkTasks()
        if hasattr(self.parameters, 'trunk_mode'):
            self.tasks.trunk_mode = self.parameters.trunk_mode
        self.tasks.com_x = init_params.get('com_x', 0)
        self.tasks.initialize_tasks(self.solver, self.robot)
        self.tasks.left_foot_task.orientation().mask.set_axises("yz", "local")
        self.tasks.right_foot_task.orientation().mask.set_axises("yz", "local")
        # self.tasks.trunk_orientation_task.configure("trunk_orientation", "soft", 1e-4)
        self.tasks.left_foot_task.orientation().configure("left_foot_orientation", "soft", 1e-5)
        self.tasks.right_foot_task.orientation().configure("right_foot_orientation", "soft", 1e-5)

        # # Creating a joint task to assign DoF values for upper body
        # self.joints and self.joint_angles are set in load_parameters
        joint_radians = {joint: np.deg2rad(degrees) for joint, degrees in self.joint_angles.items()}
        self.joints_task = self.solver.add_joints_task()
        self.joints_task.set_joints(joint_radians)
        self.joints_task.configure("joints", "soft", 0.1)

        # Placing the robot in the initial position
        print("Placing the robot in the initial position...")
        self.tasks.reach_initial_pose(
            np.eye(4),
            self.parameters.feet_spacing,
            self.parameters.walk_com_height,
            self.parameters.walk_trunk_pitch,
        )
        
        self.robot.update_kinematics()
        print("Initial position reached")
        # exit()

        # Creating the FootstepsPlanner
        self.repetitive_footsteps_planner = placo.FootstepsPlannerRepetitive(
            self.parameters
        )
        # Initialize trajectory from init_params if provided, otherwise default to 0
        self.d_x = init_params.get('dx', 0.0) if init_params else 0.0
        self.d_y = init_params.get('dy', 0.0) if init_params else 0.0
        self.d_theta = init_params.get('dtheta', 0.0) if init_params else 0.0
        self.nb_steps = 10
        self.repetitive_footsteps_planner.configure(
            self.d_x, self.d_y, self.d_theta, self.nb_steps
        )

        # Planning footsteps
        self.T_world_left = placo.flatten_on_floor(self.robot.get_T_world_left())
        self.T_world_right = placo.flatten_on_floor(self.robot.get_T_world_right())
        self.footsteps = self.repetitive_footsteps_planner.plan(
            placo.HumanoidRobot_Side.left, self.T_world_left, self.T_world_right
        )

        self.supports = placo.FootstepsPlanner.make_supports(
            self.footsteps, 0.0, True, self.parameters.has_double_support(), True
        )

        # Creating the pattern generator and making an initial plan
        self.walk = placo.WalkPatternGenerator(self.robot, self.parameters)
        self.trajectory = self.walk.plan(self.supports, self.robot.com_world(), 0.0)

        self.time_since_last_right_contact = 0.0
        self.time_since_last_left_contact = 0.0
        self.start = None
        self.initial_delay = 0.0
        # self.initial_delay = 0
        self.t = self.initial_delay
        self.last_replan = 0

        # TODO remove startend_double_support_duration() when starting and ending ?
        self.period = (
            2 * self.parameters.single_support_duration
            + 2 * self.parameters.double_support_duration()
        )
        
        # Calculate warmup time (time before stable walk cycle begins)
        # This includes the startend double support phase + 1 full period for the robot to stabilize
        self.warmup_time = (
            self.parameters.startend_double_support_duration() 
            + self.period
        )
        
        self.robot.update_kinematics()
        self.solver.solve(True)
        
        print("## period:", self.period)
        
    def load_defaults(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        params = self.parameters
        self.load_parameters(data)

    def load_parameters(self, data):
        params = self.parameters
        params.double_support_ratio = data.get('double_support_ratio', params.double_support_ratio)
        params.startend_double_support_ratio = data.get('startend_double_support_ratio', params.startend_double_support_ratio)
        params.planned_timesteps = data.get('planned_timesteps', params.planned_timesteps)
        # params.replan_timesteps = data.get('replan_timesteps', params.replan_timesteps)  # Removed in Placo 0.9+
        params.walk_com_height = data.get('walk_com_height', params.walk_com_height)
        params.walk_foot_height = data.get('walk_foot_height', params.walk_foot_height)
        params.walk_trunk_pitch = np.deg2rad(data.get('walk_trunk_pitch', np.rad2deg(params.walk_trunk_pitch)))
        params.walk_foot_rise_ratio = data.get('walk_foot_rise_ratio', params.walk_foot_rise_ratio)
        params.single_support_duration = data.get('single_support_duration', params.single_support_duration)
        params.single_support_timesteps = data.get('single_support_timesteps', params.single_support_timesteps)
        params.foot_length = data.get('foot_length', params.foot_length)
        params.feet_spacing = data.get('feet_spacing', params.feet_spacing)
        params.zmp_margin = data.get('zmp_margin', params.zmp_margin)
        params.foot_zmp_target_x = data.get('foot_zmp_target_x', params.foot_zmp_target_x)
        params.foot_zmp_target_y = data.get('foot_zmp_target_y', params.foot_zmp_target_y)
        params.walk_max_dtheta = data.get('walk_max_dtheta', params.walk_max_dtheta)
        params.walk_max_dy = data.get('walk_max_dy', params.walk_max_dy)
        params.walk_max_dx_forward = data.get('walk_max_dx_forward', params.walk_max_dx_forward)
        params.walk_max_dx_backward = data.get('walk_max_dx_backward', params.walk_max_dx_backward)
        # Store joints info on the walk engine instance, not on HumanoidParameters
        self.joints = data.get('joints', [])
        self.joint_angles = data.get('joint_angles', {})
        if 'trunk_mode' in data:
            params.trunk_mode = data.get('trunk_mode')

    def get_angles(self):
        angles = {joint: self.robot.get_joint(joint) for joint in self.joints}
        return angles

    def reset(self):
        self.t = 0
        self.start = None
        self.last_replan = 0
        self.time_since_last_right_contact = 0.0
        self.time_since_last_left_contact = 0.0

        self.tasks.reach_initial_pose(
            np.eye(4),
            self.parameters.feet_spacing,
            self.parameters.walk_com_height,
            self.parameters.walk_trunk_pitch,
        )

        # Planning footsteps
        self.T_world_left = placo.flatten_on_floor(self.robot.get_T_world_left())
        self.T_world_right = placo.flatten_on_floor(self.robot.get_T_world_right())
        self.footsteps = self.repetitive_footsteps_planner.plan(
            placo.HumanoidRobot_Side.left, self.T_world_left, self.T_world_right
        )

        self.supports = placo.FootstepsPlanner.make_supports(
            self.footsteps, 0.0, True, self.parameters.has_double_support(), True
        )
        self.trajectory = self.walk.plan(self.supports, self.robot.com_world(), 0.0)

    def set_traj(self, d_x, d_y, d_theta):
        self.d_x = d_x
        self.d_y = d_y
        self.d_theta = d_theta
        self.repetitive_footsteps_planner.configure(
            self.d_x, self.d_y, self.d_theta, self.nb_steps
        )

    def get_footsteps_in_world(self):
        footsteps = self.trajectory.get_supports()
        footsteps_in_world = []
        for footstep in footsteps:
            if not footstep.is_both():
                footsteps_in_world.append(footstep.frame())

        for i in range(len(footsteps_in_world)):
            footsteps_in_world[i][:3, 3][1] += self.parameters.feet_spacing / 2

        return footsteps_in_world

    def get_footsteps_in_robot_frame(self):
        T_world_fbase = self.robot.get_T_world_fbase()

        footsteps = self.trajectory.get_supports()
        footsteps_in_robot_frame = []
        for footstep in footsteps:
            if not footstep.is_both():
                T_world_footstepFrame = footstep.frame().copy()
                T_fbase_footstepFrame = (
                    np.linalg.inv(T_world_fbase) @ T_world_footstepFrame
                )
                T_fbase_footstepFrame = placo.flatten_on_floor(T_fbase_footstepFrame)
                T_fbase_footstepFrame[:3, 3][2] = -T_world_fbase[:3, 3][2]

                footsteps_in_robot_frame.append(T_fbase_footstepFrame)

        return footsteps_in_robot_frame

    def get_current_support_phase(self):
        if self.trajectory.support_is_both(self.t):
            return [1, 1]
        elif str(self.trajectory.support_side(self.t)) == "left":
            return [1, 0]
        elif str(self.trajectory.support_side(self.t)) == "right":
            return [0, 1]
        else:
            raise AssertionError(f"Invalid phase: {self.trajectory.support_side(self.t)}")

    def tick(self, dt, left_contact=True, right_contact=True):
        if self.start is None:
            self.start = time.time()
        
        if not self.ignore_feet_contact:
            if left_contact:
                self.time_since_last_left_contact = 0.0
            if right_contact:
                self.time_since_last_right_contact = 0.0

        falling = not self.ignore_feet_contact and (
            self.time_since_last_left_contact > self.parameters.single_support_duration
            or self.time_since_last_right_contact
            > self.parameters.single_support_duration
        )

        for k in range(REFINE):
            # Updating the QP tasks from planned trajectory
            if not falling:
                self.tasks.update_tasks_from_trajectory(
                    self.trajectory, self.t - dt + k * dt / REFINE
                )

            self.robot.update_kinematics()
            _ = self.solver.solve(True)
        
        if self.head_bob:
            # Triangle wave with 2 cycles per period instead of 4
            t_normalized = 2 * self.t / self.period
            triangle_wave = 2 * np.abs(2 * (t_normalized % 1) - 1) - 1
            triangle_wave *= self.head_bob_amplitude
            
            self.robot.set_joint("head_pitch", self.default_angles["head_pitch"] + triangle_wave)
            self.robot.set_joint("neck_pitch", self.default_angles["neck_pitch"] + self.neck_pitch_sign * triangle_wave)
        
        if self.tail_wiggle:
            self.robot.set_joint("tail", self.default_angles["tail"] + self.tail_wiggle_amplitude*np.sin(2*np.pi*self.t / self.period))

        # If enough time elapsed and we can replan, do the replanning
        if (
            self.t - self.last_replan
            > self.parameters.planned_timesteps * self.parameters.dt()  # Using planned_timesteps (replan_timesteps removed in Placo 0.9+)
            and self.walk.can_replan_supports(self.trajectory, self.t)
        ):
            self.last_replan = self.t

            # Replanning footsteps from current trajectory
            # API changed in Placo 0.9+ to require additional time parameter
            horizon = self.parameters.planned_timesteps * self.parameters.dt()
            self.supports = self.walk.replan_supports(
                self.repetitive_footsteps_planner, self.trajectory, self.t, horizon
            )

            # Replanning CoM trajectory, yielding a new trajectory we can switch to
            self.trajectory = self.walk.replan(self.supports, self.trajectory, self.t)

        self.time_since_last_left_contact += dt
        self.time_since_last_right_contact += dt
        self.t += dt

        # while time.time() < self.start_t + self.t:
        #     time.sleep(1e-3)
