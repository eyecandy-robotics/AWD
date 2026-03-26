import sys

sys.path = [p for p in sys.path if "mjlab" not in p] + sys.path

import argparse
import json
import os
import pickle
import threading
import time
import webbrowser

import numpy as np
import placo

import torch
from placo_utils.visualization import *
from scipy.spatial.transform import Rotation as R


def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
  """Returns a 4x4 homogeneous rotation matrix from roll, pitch, yaw in degrees.
  Order: Rz(yaw) * Ry(pitch) * Rx(roll), applied in world/global frame (absolute axes).
  """
  roll = np.radians(roll_deg)
  pitch = np.radians(pitch_deg)
  yaw = np.radians(yaw_deg)

  Rx = np.array(
    [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
  )

  Ry = np.array(
    [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
  )

  Rz = np.array(
    [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
  )

  # Final rotation: R = Rz * Ry * Rx
  R = Rz @ Ry @ Rx

  # Make it a 4x4 homogeneous matrix
  R_homogeneous = np.eye(4)
  R_homogeneous[:3, :3] = R
  return R_homogeneous


def open_browser():
  try:
    webbrowser.open_new("http://127.0.0.1:7000/static/")
  except:
    print("Failed to open the default browser. Trying Google Chrome.")
    try:
      webbrowser.get("google-chrome").open_new("http://127.0.0.1:7000/static/")
    except:
      print("Failed to open Google Chrome. Make sure it's installed and accessible.")


class PerpetualController:
  def __init__(self, urdf_path=None, target_frame_name="trunk_assembly", robot="mini2"):
    if not urdf_path:
      return
    self.robot_name = robot
    # Initialize the robot
    self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)

    self.config = self._load_config(urdf_path, robot, target_frame_name)

    # Creating the solver
    self.solver = placo.KinematicsSolver(self.robot)
    self.target_frame_name = self.config.get("target_frame_name", target_frame_name)

    # Adding a custom regularization task
    # self.regularization_task = self.solver.add_regularization_task(1e-4)

    # Set up the initial trunk position
    default_height = self.config.get(
      "default_height",
      0.16 if robot == "mini2" else (0.145 if robot == "dinoJr" else 0.11),
    )

    self.T_world_trunk = tf.translation_matrix([0.0, 0.0, default_height])
    self.effector_task = self.solver.add_frame_task(
      self.target_frame_name, self.T_world_trunk
    )
    effector_cfg = self.config.get("effector_task", {})
    self.effector_task.configure(
      self.target_frame_name,
      "soft",
      effector_cfg.get("position_weight", 2.0),
      effector_cfg.get("orientation_weight", 1.0),
    )

    self.solver.solve(True)

    # Set up initial joint positions
    self.initial_joints = self.config.get("initial_joints", {})

    for joint in self.initial_joints:
      self.robot.set_joint(joint, self.initial_joints[joint])
    self.robot.update_kinematics()

    # Store default transform
    self.default_trunk_transform = self.robot.get_T_world_frame(self.target_frame_name)

    # Retrieving initial position of the feet
    self.T_world_left = self.robot.get_T_world_frame("left_foot")
    self.T_world_right = self.robot.get_T_world_frame("right_foot")

    # Keep left and right foot on the floor
    self.left_foot_task = self.solver.add_frame_task("left_foot", self.T_world_left)
    self.right_foot_task = self.solver.add_frame_task("right_foot", self.T_world_right)
    foot_task_cfg = self.config.get("foot_task", {})
    self.right_foot_task.configure("right_foot", "soft", foot_task_cfg.get("position_weight", 1.0), foot_task_cfg.get("orientation_weight", 1e-3))
    self.left_foot_task.configure("left_foot", "soft", foot_task_cfg.get("position_weight", 1.0), foot_task_cfg.get("orientation_weight", 1e-3))

    # Set solver timestep
    self.solver.dt = self.config.get("solver_dt", 0.01)

    self.lookup_ranges = self.config.get("lookup_ranges", {})
    self.runtime_random_ranges = self.config.get(
      "runtime_random_ranges",
      {
        "delta_roll": [-10.0, 10.0],
        "delta_pitch": [-10.0, 10.0],
        "delta_yaw": [-10.0, 10.0],
        "delta_height": [-0.015, 0.015],
      },
    )

  def _build_range(self, spec, default_min, default_max, default_step):
    if not isinstance(spec, dict):
      return np.arange(default_min, default_max + default_step / 2, default_step)
    min_value = spec.get("min", default_min)
    max_value = spec.get("max", default_max)
    step = spec.get("step", default_step)
    return np.arange(min_value, max_value + step / 2, step)

  def _load_config(self, urdf_path, robot, target_frame_name):
    defaults_path = os.path.join(os.path.dirname(os.path.abspath(urdf_path)), "placo_defaults.json")
    config = {"target_frame_name": target_frame_name}
    if not os.path.isfile(defaults_path):
      return config

    with open(defaults_path, "r") as f:
      defaults_data = json.load(f)

    config.update(defaults_data.get("perpetual_control", {}))

    config.setdefault(
      "lookup_ranges",
      {
        "delta_roll": {"min": -10.0, "max": 10.0, "step": 1.0},
        "delta_pitch": {"min": -10.0, "max": 10.0, "step": 0.5},
        "delta_yaw": {"min": -10.0, "max": 10.0, "step": 0.5},
        "delta_height": {"min": -0.01, "max": 0.015, "step": 0.002},
      },
    )

    if "initial_joints" not in config:
      config["initial_joints"] = defaults_data.get("default_angles", {})
    return config

  def get_frame(self, delta_roll, delta_pitch, delta_yaw, delta_height):
    """
    Returns the root position, orientation, and joint angles based on delta values.

    Args:
        delta_roll: Change in roll angle in degrees
        delta_pitch: Change in pitch angle in degrees
        delta_yaw: Change in yaw angle in degrees
        delta_height: Change in height

    Returns:
        array: Concatenated array of [root_pos, root_orient, joint_angles]
    """
    # Apply rotations
    target = euler_to_rotation_matrix(delta_roll, delta_pitch, delta_yaw)
    # Apply height change
    target[2, 3] = delta_height + self.default_trunk_transform[2, 3]

    # Update the target
    self.effector_task.T_world_frame = target

    # Solve the IK
    self.solver.solve(True)
    self.robot.update_kinematics()

    # Get root position and orientation
    current_T = self.robot.get_T_world_frame(self.target_frame_name)
    root_pos = current_T[:3, 3]
    root_orient = R.from_matrix(current_T[:3, :3]).as_quat()

    # Get joint angles
    joint_angles = []
    for joint_name in self.initial_joints:
      joint_angles.append(self.robot.get_joint(joint_name))

    # Concatenate all values into a single array and return
    return np.concatenate([root_pos, root_orient, np.array(joint_angles)])

  def build_lookup_table(self):
    """
    Build a lookup table for frame values across the specified ranges.

    Returns:
        dict: Lookup table with ranges and frame data
    """
    delta_roll_range = self._build_range(self.lookup_ranges.get("delta_roll"), -10.0, 10.0, 1.0)
    delta_pitch_range = self._build_range(self.lookup_ranges.get("delta_pitch"), -10.0, 10.0, 0.5)
    delta_yaw_range = self._build_range(self.lookup_ranges.get("delta_yaw"), -10.0, 10.0, 0.5)
    delta_height_range = self._build_range(self.lookup_ranges.get("delta_height"), -0.01, 0.015, 0.002)

    # Get dimensions
    n_roll = len(delta_roll_range)
    n_pitch = len(delta_pitch_range)
    n_yaw = len(delta_yaw_range)
    n_height = len(delta_height_range)

    # Sample one frame to get its dimension
    sample_frame = self.get_frame(0, 0, 0, 0)
    frame_dim = len(sample_frame)

    # Create the lookup table
    lookup_table = {
      "delta_roll_range": delta_roll_range,
      "delta_pitch_range": delta_pitch_range,
      "delta_yaw_range": delta_yaw_range,
      "delta_height_range": delta_height_range,
      "frames": np.zeros((n_roll, n_pitch, n_yaw, n_height, frame_dim)),
    }

    # Total number of combinations for progress tracking
    total_combinations = n_roll * n_pitch * n_yaw * n_height
    count = 0

    print(f"Building lookup table with {total_combinations} entries...")

    # Build the lookup table
    for i, roll in enumerate(delta_roll_range):
      for j, pitch in enumerate(delta_pitch_range):
        for k, yaw in enumerate(delta_yaw_range):
          for l, height in enumerate(delta_height_range):
            # Get the frame for this combination
            frame = self.get_frame(roll, pitch, yaw, height)
            lookup_table["frames"][i, j, k, l] = frame

            # Update progress
            count += 1
            if count % 1000 == 0:
              print(
                f"Progress: {count}/{total_combinations} ({count / total_combinations * 100:.2f}%)"
              )

    return lookup_table

  def save_lookup_table(self, lookup_table, filename="perpetual_lookup.pkl"):
    """
    Save the lookup table to a pickle file
    """
    with open(filename, "wb") as f:
      pickle.dump(lookup_table, f)

    print(f"Lookup table saved to {filename}")

  def load_lookup_table_torch(self, filename="perpetual_lookup.pkl", device="cuda"):
    """
    Load the lookup table from a pickle file and store as PyTorch tensors
    for faster interpolation
    """
    with open(filename, "rb") as f:
      self.lookup_table = pickle.load(f)

    self.device = device
    # Convert arrays to torch tensors
    self.rolls = torch.tensor(
      self.lookup_table["delta_roll_range"], dtype=torch.float32, device=device
    )
    self.pitches = torch.tensor(
      self.lookup_table["delta_pitch_range"], dtype=torch.float32, device=device
    )
    self.yaws = torch.tensor(
      self.lookup_table["delta_yaw_range"], dtype=torch.float32, device=device
    )
    self.heights = torch.tensor(
      self.lookup_table["delta_height_range"], dtype=torch.float32, device=device
    )
    self.frames = torch.tensor(
      self.lookup_table["frames"], dtype=torch.float32, device=device
    )

    print(f"Torch lookup table loaded on {device}.")
    return self.lookup_table

  def query_lookup_table_torch(
    self, delta_roll, delta_pitch, delta_yaw, delta_height, in_degrees=True
  ):
    """
    Query the lookup table using multilinear interpolation for continuous inputs.
    Handles both scalar and batched inputs.

    Args:
        delta_roll: Change in roll angle - scalar or shape (num_envs,)
        delta_pitch: Change in pitch angle - scalar or shape (num_envs,)
        delta_yaw: Change in yaw angle - scalar or shape (num_envs,)
        delta_height: Change in height - scalar or shape (num_envs,)
        in_degrees: Whether input angles are in degrees (default True)

    Returns:
        tensor: Frame values - shape (frame_dim,) for scalar input or (num_envs, frame_dim) for batched
    """
    # Convert to tensor if not already
    if not isinstance(delta_roll, torch.Tensor):
      delta_roll = torch.tensor(delta_roll, dtype=torch.float32, device=self.device)
    if not isinstance(delta_pitch, torch.Tensor):
      delta_pitch = torch.tensor(delta_pitch, dtype=torch.float32, device=self.device)
    if not isinstance(delta_yaw, torch.Tensor):
      delta_yaw = torch.tensor(delta_yaw, dtype=torch.float32, device=self.device)
    if not isinstance(delta_height, torch.Tensor):
      delta_height = torch.tensor(delta_height, dtype=torch.float32, device=self.device)

    # Ensure inputs are on the correct device
    delta_roll = delta_roll.to(self.device)
    delta_pitch = delta_pitch.to(self.device)
    delta_yaw = delta_yaw.to(self.device)
    delta_height = delta_height.to(self.device)

    # Check if input is scalar (0-dim tensor)
    is_scalar = delta_roll.dim() == 0

    # Ensure at least 1D for consistent processing
    if is_scalar:
      delta_roll = delta_roll.unsqueeze(0)
      delta_pitch = delta_pitch.unsqueeze(0)
      delta_yaw = delta_yaw.unsqueeze(0)
      delta_height = delta_height.unsqueeze(0)

    if not in_degrees:
      delta_roll = torch.rad2deg(delta_roll)
      delta_pitch = torch.rad2deg(delta_pitch)
      delta_yaw = torch.rad2deg(delta_yaw)

    # Clamp values to valid range
    delta_roll = torch.clamp(delta_roll, self.rolls.min(), self.rolls.max())
    delta_pitch = torch.clamp(delta_pitch, self.pitches.min(), self.pitches.max())
    delta_yaw = torch.clamp(delta_yaw, self.yaws.min(), self.yaws.max())
    delta_height = torch.clamp(delta_height, self.heights.min(), self.heights.max())

    # Find indices for each environment (shape: num_envs or 1)
    i0 = torch.searchsorted(self.rolls, delta_roll, side="right") - 1
    j0 = torch.searchsorted(self.pitches, delta_pitch, side="right") - 1
    k0 = torch.searchsorted(self.yaws, delta_yaw, side="right") - 1
    l0 = torch.searchsorted(self.heights, delta_height, side="right") - 1

    # Handle edge cases
    i0 = torch.clamp(i0, 0, len(self.rolls) - 2)
    j0 = torch.clamp(j0, 0, len(self.pitches) - 2)
    k0 = torch.clamp(k0, 0, len(self.yaws) - 2)
    l0 = torch.clamp(l0, 0, len(self.heights) - 2)

    i1, j1, k1, l1 = i0 + 1, j0 + 1, k0 + 1, l0 + 1

    # Calculate interpolation weights
    wd = (delta_roll - self.rolls[i0]) / (self.rolls[i1] - self.rolls[i0])
    wp = (delta_pitch - self.pitches[j0]) / (self.pitches[j1] - self.pitches[j0])
    wy = (delta_yaw - self.yaws[k0]) / (self.yaws[k1] - self.yaws[k0])
    wh = (delta_height - self.heights[l0]) / (self.heights[l1] - self.heights[l0])

    # Expand weights for broadcasting (shape: num_envs, 1)
    wd = wd.unsqueeze(-1)
    wp = wp.unsqueeze(-1)
    wy = wy.unsqueeze(-1)
    wh = wh.unsqueeze(-1)

    # Get corner values for all environments (shape: num_envs, frame_dim)
    c0000 = self.frames[i0, j0, k0, l0]
    c0001 = self.frames[i0, j0, k0, l1]
    c0010 = self.frames[i0, j0, k1, l0]
    c0011 = self.frames[i0, j0, k1, l1]
    c0100 = self.frames[i0, j1, k0, l0]
    c0101 = self.frames[i0, j1, k0, l1]
    c0110 = self.frames[i0, j1, k1, l0]
    c0111 = self.frames[i0, j1, k1, l1]
    c1000 = self.frames[i1, j0, k0, l0]
    c1001 = self.frames[i1, j0, k0, l1]
    c1010 = self.frames[i1, j0, k1, l0]
    c1011 = self.frames[i1, j0, k1, l1]
    c1100 = self.frames[i1, j1, k0, l0]
    c1101 = self.frames[i1, j1, k0, l1]
    c1110 = self.frames[i1, j1, k1, l0]
    c1111 = self.frames[i1, j1, k1, l1]

    # 4D linear interpolation using weights (vectorized)
    c000 = c0000 * (1 - wd) + c1000 * wd
    c001 = c0001 * (1 - wd) + c1001 * wd
    c010 = c0010 * (1 - wd) + c1010 * wd
    c011 = c0011 * (1 - wd) + c1011 * wd
    c100 = c0100 * (1 - wd) + c1100 * wd
    c101 = c0101 * (1 - wd) + c1101 * wd
    c110 = c0110 * (1 - wd) + c1110 * wd
    c111 = c0111 * (1 - wd) + c1111 * wd

    c00 = c000 * (1 - wp) + c100 * wp
    c01 = c001 * (1 - wp) + c101 * wp
    c10 = c010 * (1 - wp) + c110 * wp
    c11 = c011 * (1 - wp) + c111 * wp

    c0 = c00 * (1 - wy) + c10 * wy
    c1 = c01 * (1 - wy) + c11 * wy

    result = c0 * (1 - wh) + c1 * wh

    # Squeeze back to scalar output if input was scalar
    if is_scalar:
      result = result.squeeze(0)

    return result


# Example usage in main program:
if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("path", help="Path to the URDF")
  arg_parser.add_argument(
    "--build-lookup", action="store_true", help="Build and save the lookup table"
  )
  arg_parser.add_argument(
    "--use-lookup",
    action="store_true",
    help="Use the lookup table instead of direct computation",
  )
  arg_parser.add_argument(
    "--use-torch",
    action="store_true",
    help="Use PyTorch for faster lookup table queries",
  )
  arg_parser.add_argument(
    "--lookup-file",
    default="perpetual_lookup.pkl",
    help="Path to the lookup table file",
  )
  arg_parser.add_argument("--robot", choices=["mini2", "dino", "dinoJr"], required=True)
  arg_parser.add_argument("--track_target", default="trunk_assembly", required=True)
  args = arg_parser.parse_args()

  controller = PerpetualController(args.path, args.track_target, args.robot)

  # Build and save the lookup table if requested
  if args.build_lookup:
    print("Building lookup table...")
    lookup_table = controller.build_lookup_table()
    controller.save_lookup_table(lookup_table, args.lookup_file)
    print("Lookup table built and saved.")
    exit(0)

  # Load the lookup table if using it
  if args.use_lookup:
    print("Loading lookup table...")
    if args.use_torch:
      controller.load_lookup_table_torch(args.lookup_file)

  # Set up visualization
  viz = robot_viz(controller.robot)
  threading.Timer(1, open_browser).start()

  t = 0
  dt = controller.config.get("runtime_dt", 0.1)
  runtime_ranges = controller.runtime_random_ranges
  sweep_periods = controller.config.get(
    "runtime_sweep_periods",
    {
      "delta_roll": 8.0,
      "delta_pitch": 10.0,
      "delta_yaw": 12.0,
      "delta_height": 14.0,
    },
  )

  def sin_sweep_value(ranges, key, default_range, phase):
    min_val, max_val = ranges.get(key, default_range)
    center = 0.5 * (min_val + max_val)
    amplitude = 0.5 * (max_val - min_val)
    period = max(float(sweep_periods.get(key, 8.0)), 1e-6)
    omega = 2.0 * np.pi / period
    return center + amplitude * np.sin(omega * t + phase)

  while True:
    t += dt

    delta_roll = sin_sweep_value(runtime_ranges, "delta_roll", [-10.0, 10.0], 0.0)
    delta_pitch = sin_sweep_value(runtime_ranges, "delta_pitch", [-10.0, 10.0], np.pi / 2)
    delta_yaw = sin_sweep_value(runtime_ranges, "delta_yaw", [-10.0, 10.0], np.pi)
    delta_height = sin_sweep_value(runtime_ranges, "delta_height", [-0.015, 0.015], 3 * np.pi / 2)

    start_time = time.time()

    if args.use_lookup and args.use_torch:
      frame_data = (
        controller.query_lookup_table_torch(
          delta_roll, delta_pitch, delta_yaw, delta_height
        )
        .cpu()
        .numpy()
      )

      # Update the robot state with the frame data
      root_pos = frame_data[:3]
      root_orient = frame_data[3:7]
      joint_angles = frame_data[7:]

      # Set the joint positions from the lookup
      for i, joint_name in enumerate(controller.robot.joint_names()):
        print(joint_name, joint_angles[i])
        controller.robot.set_joint(joint_name, float(joint_angles[i]))
      controller.robot.update_kinematics()
    else:
      # Directly compute the frame
      frame_data = controller.get_frame(
        delta_roll, delta_pitch, delta_yaw, delta_height
      )

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    method = "Direct Computation"
    if args.use_lookup:
      method = "Torch Lookup Table" if args.use_torch else "Lookup Table"
    print(f"Method: {method}, Elapsed time: {elapsed_time_ms:.2f} ms")

    # Display the robot
    viz.display(controller.robot.state.q)
    robot_frame_viz(controller.robot, controller.target_frame_name)
    frame_viz("target", controller.effector_task.T_world_frame)

    time.sleep(dt)
