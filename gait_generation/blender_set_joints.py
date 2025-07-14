import bpy
import math

# Define the joint names (should match Blender bone names exactly)
joint_names = [
    'left_hip_yaw.revolute.bone',
    'left_hip_roll.revolute.bone',
    'left_hip_pitch.revolute.bone',
    'left_knee.revolute.bone',
    'left_ankle.revolute.bone',
    'neck_pitch.revolute.bone',
    'head_pitch.revolute.bone',
    'head_yaw.revolute.bone',
    'head_roll.revolute.bone',
    'left_antenna.revolute.bone',
    'right_antenna.revolute.bone',
    'right_hip_yaw.revolute.bone',
    'right_hip_roll.revolute.bone',
    'right_hip_pitch.revolute.bone',
    'right_knee.revolute.bone',
    'right_ankle.revolute.bone',
]

# Example list of joint angles (in radians)
# Replace this list with your own values
input_angles = [
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]

# Name of the armature object
armature_obj = bpy.context.object  # Assumes the correct armature is selected

# Make sure we are in pose mode
bpy.ops.object.mode_set(mode='POSE')

# Apply each angle to the corresponding joint
for name, angle in zip(joint_names, input_angles):
    bone_name = name  # exact match to pose bone name
    bone = armature_obj.pose.bones.get(bone_name)
    if bone is None:
        print(f"Bone not found: {bone_name}")
        continue

    # Get local rotation mode
    bone.rotation_mode = 'XYZ'

    # Set the angle to the appropriate axis
    # Adjust per-joint axis mapping as needed (Y-axis assumed here)
    if "head_yaw" in bone_name:
        bone.rotation_euler.z = angle
    else:
        bone.rotation_euler.y = angle

print("Joint angles applied.")
