import argparse
import json
import time

import FramesViewer.utils as fv_utils
import numpy as np
from FramesViewer.viewer import Viewer
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True)
parser.add_argument(
    "--hardware",
    action="store_true",
    help="use AMP_for_hardware format. If false, use IsaacGymEnvs format",
)
args = parser.parse_args()

fv = Viewer()
fv.start()

episode = json.load(open(args.file))

frame_duration = episode["FrameDuration"]

frames = episode["Frames"]
frame_offsets = episode["Frame_offset"][0]
period = episode["Placo"]["period"]

joint_names = episode["Joints"]

# Add slice for foot contacts if present in frame_offsets
if "foot_contacts" in frame_offsets:
    foot_contacts_slice = slice(frame_offsets["foot_contacts"], frame_offsets["foot_contacts"] + 2)
else:
    # Fallback if not explicitly defined
    print("Warning: foot_contacts not found in frame_offsets")
    foot_contacts_slice = None

root_pos_slice = slice(frame_offsets["root_pos"], frame_offsets["root_quat"])
root_quat_slice = slice(frame_offsets["root_quat"], frame_offsets["joints_pos"])

linear_vel_slice = slice(frame_offsets["world_linear_vel"], frame_offsets["world_angular_vel"])
angular_vel_slice = slice(frame_offsets["world_angular_vel"], frame_offsets["joints_vel"])
joint_vels_slice = slice(frame_offsets["joints_vel"], frame_offsets["left_toe_vel"])
joint_pos_slice = slice(frame_offsets["joints_pos"], frame_offsets["left_toe_pos"])

left_toe_pos_slice = slice(frame_offsets["left_toe_pos"], frame_offsets["right_toe_pos"])
right_toe_pos_slice = slice(frame_offsets["right_toe_pos"], frame_offsets["world_linear_vel"])

if "Debug_info" in episode:
    debug = episode["Debug_info"]
else:
    debug = None
pose = np.eye(4)
vels = {}
vels["linear_vel"] = []
vels["angular_vel"] = []
vels["joint_vels"] = []
left_foot_contacts = []
joint_pos = []
right_foot_contacts = []

for i, frame in enumerate(frames):
    root_position = frame[root_pos_slice]
    root_orientation_quat = frame[root_quat_slice]
    root_orientation_mat = R.from_quat(root_orientation_quat).as_matrix()

    pose[:3, 3] = root_position
    pose[:3, :3] = root_orientation_mat

    fv.pushFrame(pose, "aze")

    vels["linear_vel"].append(frame[linear_vel_slice])
    vels["angular_vel"].append(frame[angular_vel_slice])
    vels["joint_vels"].append(frame[joint_vels_slice])

    joint_pos.append(frame[joint_pos_slice])

    left_toe_pos = np.array(frame[left_toe_pos_slice]) #+ np.array(root_position)
    right_toe_pos = np.array(frame[right_toe_pos_slice]) #+ np.array(root_position)
    
    fv.pushFrame(fv_utils.make_pose(left_toe_pos, [0, 0, 0]), "left_toe")
    fv.pushFrame(fv_utils.make_pose(right_toe_pos, [0, 0, 0]), "right_toe")
    
    # Extract foot contacts if available
    if foot_contacts_slice:
        contacts = frame[foot_contacts_slice]
        left_foot_contacts.append(float(contacts[0]))
        right_foot_contacts.append(float(contacts[1]))
    #time.sleep(frame_duration)
    #input()

# plot vels
x_lin_vel = [vels["linear_vel"][i][0] for i in range(len(frames))]
y_lin_vel = [vels["linear_vel"][i][1] for i in range(len(frames))]
z_lin_vel = [vels["linear_vel"][i][2] for i in range(len(frames))]

joints_vel = [vels["joint_vels"][i] for i in range(len(frames))]
angular_vel_x = [vels["angular_vel"][i][0] for i in range(len(frames))]
angular_vel_y = [vels["angular_vel"][i][1] for i in range(len(frames))]
angular_vel_z = [vels["angular_vel"][i][2] for i in range(len(frames))]

# print means
print("Linear Velocity Mean:")
print(f"x: {np.mean(x_lin_vel):.2f}, y: {np.mean(y_lin_vel):.2f}, z: {np.mean(angular_vel_z):.2f}")

# Calculate fps from frame duration
fps = 1.0 / frame_duration
nb_steps_in_period = int(period * fps)

# Create cyclical period array using the number of steps in period
frame_indices = np.arange(len(frames))
period_array = (frame_indices % nb_steps_in_period) / nb_steps_in_period  # This will cycle from 0 to 1

plt.figure(figsize=(12, 6))

# Plot everything on the same graph
plt.plot(x_lin_vel, label="x_lin_vel")
plt.plot(y_lin_vel, label="y_lin_vel")
plt.plot(angular_vel_z, label="angular_vel_z")
plt.plot(period_array, label="period")

# Plot foot contacts if available
if foot_contacts_slice:
    plt.plot(left_foot_contacts, label="Left Foot Contact")
    plt.plot(right_foot_contacts, label="Right Foot Contact")

plt.legend()
plt.title("Motion Analysis")
plt.grid(True, alpha=0.3)
plt.show()

# plot joint positions (nx15)
joint_pos = np.array(joint_pos)
plt.figure(figsize=(12, 6))
for i in range(joint_pos.shape[1]):
    plt.plot(joint_pos[:, i], label=f"Joint {i+1}")
plt.title("Joint Positions")
plt.xlabel("Frame")
plt.ylabel("Position")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print max and min values for each joint
print("\nJoint Position Ranges:")
print("------------------------")

# Define original joint limits in degrees
original_joint_limits = {
    "head_yaw": (-180.0, 180.0),
    "head_pitch": (-60.0, 60.0),
    "neck_pitch": (-60.0, 90.0),
    "hind_tail": (-40.0, 40.0),
    "tail": (-20.0, 20.0),           # Updated from (-15.0, 15.0)
    "right_ankle": (-60.0, 60.0),
    "right_knee": (-40.0, 70.0),     # Updated from (-60.0, 30.0)
    "right_hip_pitch": (-90.0, 45.0),
    "right_hip_roll": (-25.0, 15.0), # Updated from (-30.0, 10.0)
    "right_hip_yaw": (-15.0, 20.0),
    "left_ankle": (-60.0, 60.0),
    "left_knee": (-70.0, 40.0),
    "left_hip_pitch": (-45.0, 90.0),
    "left_hip_roll": (-15.0, 25.0),  # Updated from (-10.0, 25.0)
    "left_hip_yaw": (-20.0, 15.0)    # Updated from (-20.0, 13.0)
}

print("\nJoint Limit Verification:")
print("------------------------")

for i in range(joint_pos.shape[1]):
    min_val = np.rad2deg(np.min(joint_pos[:, i]))
    max_val = np.rad2deg(np.max(joint_pos[:, i]))
    print(f"{joint_names[i]}: Min = {min_val:.4f}, Max = {max_val:.4f}, Range = {max_val - min_val:.4f}")
    
    # Check if this joint name exists in the original limits
    if joint_names[i] in original_joint_limits:
        orig_min, orig_max = original_joint_limits[joint_names[i]]
        
        # Check if observed values exceed original limits
        if min_val < orig_min or max_val > orig_max:
            print(f"  WARNING: {joint_names[i]} exceeds original limits ({orig_min:.2f}° to {orig_max:.2f}°)!")
            if min_val < orig_min:
                print(f"  - Min value {min_val:.2f}° is below limit {orig_min:.2f}°")
            if max_val > orig_max:
                print(f"  - Max value {max_val:.2f}° is above limit {orig_max:.2f}°")
    else:
        print(f"  NOTE: No original limits defined for {joint_names[i]}")

