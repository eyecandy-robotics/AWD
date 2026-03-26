import bpy
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

# Set parameters
FPS = bpy.context.scene.render.fps
units = bpy.context.scene.unit_settings
unit_system = units.system
unit_scale = units.scale_length
foot_contact_height_thresh = 0.025

print(unit_scale)
episode = {
    "LoopMode": "Wrap",
    "FPS": FPS,
    "FrameDuration": np.around(1 / FPS, 4),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": False,
    "Joints": [],
    "Vel_x": [],
    "Vel_y": [],
    "Yaw": [],
    "Frame_offset": [{}],
    "Frame_size": [{}],
    "Frames": [],
    "MotionWeight": 1,
}

actual_joint_names = ['neck_pitch', 'head_pitch', 'head_yaw', 'tail', 'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 
                      'right_knee', 'right_ankle', 'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle']

joint_names = [joint_name + ".revolute.bone" for joint_name in actual_joint_names]

key_body_names = {
    "pelvis": "base",
    "left_toe": "left_foot_tpu",
    "right_toe": "right_foot_tpu",
}

body_names = ['base', 'body_front', 'headdown', 'front_tail', 'right_upper_cover', 'left_upper_cover', 'left_foot_tpu', 'right_foot_tpu'] 

default_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 0.0]

# Init storage
prev_joint_angles = None
prev_left_toe_pos = None
prev_right_toe_pos = None
prev_pelvis_pos = None
prev_pelvis_quat = None
prev_body_positions = {name: None for name in body_names}
prev_body_quats = {name: None for name in body_names}
pelvis_positions = []
frames_data = []

eyes = None
# mat = bpy.data.materials.get("eyes")
# if mat and mat.use_nodes:
#     for node in mat.node_tree.nodes:
#         if node.type == 'BSDF_PRINCIPLED':
#             eyes = node
#             break

# Helper: angular velocity
def compute_angular_velocity(quat, prev_quat, dt):
    if prev_quat is None:
        return [0.0, 0.0, 0.0]
    r1 = R.from_quat(quat)
    r0 = R.from_quat(prev_quat)
    r_rel = r0.inv() * r1
    axis, angle = r_rel.as_rotvec(), np.linalg.norm(r_rel.as_rotvec())
    angular_velocity = axis * (angle / dt)
    return list(angular_velocity)

# Frame range
start_frame = bpy.context.scene.frame_start
end_frame = bpy.context.scene.frame_end

# Loop through frames
for frame in range(start_frame, end_frame + 1):
    bpy.context.scene.frame_set(frame)
    
    frame_joint_angles = [0.0] * len(joint_names)
    frame_data = {}

    # Get joint angles
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = bpy.context.object.evaluated_get(depsgraph)
    for bone in obj_eval.pose.bones:
        if bone.name in joint_names:
            relative_matrix = bone.matrix
            parent_bone = bone.parent
            if parent_bone:
                relative_matrix = parent_bone.matrix.inverted() @ relative_matrix
            euler_angles = relative_matrix.to_euler()
            if "hind" in bone.name:
                joint_angle = euler_angles.x
            elif "head_yaw" in bone.name:
                joint_angle = euler_angles.z
            else:
                joint_angle = euler_angles.y
            idx = joint_names.index(bone.name)
            frame_joint_angles[idx] = round(joint_angle - default_angles[idx], 4)
    
    frame_joint_angles = np.unwrap(frame_joint_angles).tolist()

    # Pelvis and toes
    pelvis_obj = bpy.data.objects[key_body_names["pelvis"]]
    pelvis_position = pelvis_obj.matrix_world.translation * unit_scale
    pelvis_quat = pelvis_obj.matrix_world.to_quaternion()

    left_toe_pos = bpy.data.objects[key_body_names["left_toe"]].matrix_world.translation * unit_scale
    right_toe_pos = bpy.data.objects[key_body_names["right_toe"]].matrix_world.translation * unit_scale
    
    pelvis_positions.append([pelvis_position.x, pelvis_position.y, pelvis_position.z])

    # Velocities
    if prev_pelvis_pos is not None:
        world_linear_vel = list((np.array([pelvis_position.x, pelvis_position.y, pelvis_position.z]) - np.array(prev_pelvis_pos)) * FPS)
    else:
        world_linear_vel = [0.0, 0.0, 0.0]

    world_angular_vel = compute_angular_velocity(
        quat=[pelvis_quat.x, pelvis_quat.y, pelvis_quat.z, pelvis_quat.w],
        prev_quat=prev_pelvis_quat,
        dt=1 / FPS
    )

    joints_vel = [0.0] * len(frame_joint_angles) if prev_joint_angles is None else list(
        (np.array(frame_joint_angles) - np.array(prev_joint_angles)) * FPS
    )
    
    left_toe_vel = [0.0, 0.0, 0.0] if prev_left_toe_pos is None else list(
        (np.array([left_toe_pos.x, left_toe_pos.y, left_toe_pos.z]) - np.array(prev_left_toe_pos)) * FPS
    )
    
    right_toe_vel = [0.0, 0.0, 0.0] if prev_right_toe_pos is None else list(
        (np.array([right_toe_pos.x, right_toe_pos.y, right_toe_pos.z]) - np.array(prev_right_toe_pos)) * FPS
    )
    
    foot_contacts = [left_toe_pos.z < foot_contact_height_thresh, right_toe_pos.z < foot_contact_height_thresh]

    # === Body tracking for all bodies ===
    body_pos_w = []
    body_quat_w = []
    body_lin_vel_w = []
    body_ang_vel_w = []
    
    for body_name in body_names:
        body_obj = bpy.data.objects.get(body_name)
        if body_obj:
            pos = body_obj.matrix_world.translation * unit_scale
            quat = body_obj.matrix_world.to_quaternion()
            
            body_pos_w.extend([pos.x, pos.y, pos.z])
            body_quat_w.extend([quat.x, quat.y, quat.z, quat.w])
            
            # Linear velocity
            if prev_body_positions[body_name] is not None:
                lin_vel = list((np.array([pos.x, pos.y, pos.z]) - np.array(prev_body_positions[body_name])) * FPS)
            else:
                lin_vel = [0.0, 0.0, 0.0]
            body_lin_vel_w.extend(lin_vel)
            
            # Angular velocity
            ang_vel = compute_angular_velocity(
                quat=[quat.x, quat.y, quat.z, quat.w],
                prev_quat=prev_body_quats[body_name],
                dt=1 / FPS
            )
            body_ang_vel_w.extend(ang_vel)
            
            # Update prev
            prev_body_positions[body_name] = [pos.x, pos.y, pos.z]
            prev_body_quats[body_name] = [quat.x, quat.y, quat.z, quat.w]
        else:
            # Body not found, fill with zeros
            print(f"Warning: Body '{body_name}' not found.")

    # === Fill frame_data ===
    frame_data["root_pos"] = [pelvis_position.x, pelvis_position.y, pelvis_position.z-0.007]
    frame_data["root_quat"] = [pelvis_quat.x, pelvis_quat.y, pelvis_quat.z, pelvis_quat.w]
    frame_data["joints_pos"] = frame_joint_angles
    frame_data["left_toe_pos"] = [left_toe_pos.x, left_toe_pos.y, left_toe_pos.z+0.01]
    frame_data["right_toe_pos"] = [right_toe_pos.x, right_toe_pos.y, right_toe_pos.z+0.01]
    frame_data["world_linear_vel"] = world_linear_vel
    frame_data["world_angular_vel"] = world_angular_vel
    frame_data["joints_vel"] = joints_vel
    frame_data["left_toe_vel"] = left_toe_vel
    frame_data["right_toe_vel"] = right_toe_vel
    frame_data["foot_contacts"] = foot_contacts
    frame_data["body_pos_w"] = body_pos_w
    frame_data["body_quat_w"] = body_quat_w
    frame_data["body_lin_vel_w"] = body_lin_vel_w
    frame_data["body_ang_vel_w"] = body_ang_vel_w

    # Optional emission values
    if eyes:
        emission_color = eyes.inputs[27].default_value[:]
        emission_strength = eyes.inputs[28].default_value
        frame_data["eyes_color"] = list(emission_color)
        frame_data["eyes_strength"] = float(emission_strength)

    frames_data.append(frame_data)

    # Update prev
    prev_joint_angles = frame_joint_angles
    prev_left_toe_pos = [left_toe_pos.x, left_toe_pos.y, left_toe_pos.z+0.01]
    prev_right_toe_pos = [right_toe_pos.x, right_toe_pos.y, right_toe_pos.z+0.01]
    prev_pelvis_pos = [pelvis_position.x, pelvis_position.y, pelvis_position.z+0.01]
    prev_pelvis_quat = [pelvis_quat.x, pelvis_quat.y, pelvis_quat.z, pelvis_quat.w]

# Vel_x, Vel_y, Yaw
pelvis_positions = np.array(pelvis_positions)
velocities = np.diff(pelvis_positions, axis=0) * FPS
episode["Joints"] = actual_joint_names
episode["Vel_x"] = np.mean(velocities[:, 0]).tolist()
episode["Vel_y"] = np.mean(velocities[:, 1]).tolist()
episode["Yaw"] = np.arctan2(np.mean(velocities[:, 1]), np.mean(velocities[:, 0])).tolist()

# Frame offsets
offset = 0
for key, value in frames_data[0].items():
    episode["Frame_offset"][0][key] = offset
    size = len(value) if isinstance(value, list) else 1
    episode["Frame_size"][0][key] = size
    offset += size

# Collect frame arrays
for frame in frames_data:
    frame_array = (
        frame["root_pos"] + frame["root_quat"] + frame["joints_pos"] +
        frame["left_toe_pos"] + frame["right_toe_pos"] +
        frame["world_linear_vel"] + frame["world_angular_vel"] + frame["joints_vel"] +
        frame["left_toe_vel"] + frame["right_toe_vel"] + frame["foot_contacts"] +
        frame["body_pos_w"] + frame["body_quat_w"] + frame["body_lin_vel_w"] + frame["body_ang_vel_w"]
    )
    if eyes:
        frame_array += frame["eyes_color"] + [frame["eyes_strength"]]
    episode["Frames"].append(frame_array)

print("Episode data filled.")

# Save JSON
output_path = "blender.json"
with open(output_path, 'w') as f:
    json.dump(episode, f, indent=4)

print(f"Episode data saved to {output_path}")
