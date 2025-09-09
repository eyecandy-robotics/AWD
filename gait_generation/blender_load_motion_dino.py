import bpy
import json

# ==== PARAMETERS ====
json_file = "blender.json"   # Path to the motion file
fps = bpy.context.scene.render.fps

# Name of armature and pelvis bone
armature_obj = bpy.context.object  # Assumes your armature is selected
pelvis_bone_name = "root.bone"   # Root bone name

# ==== LOAD DATA ====
with open(json_file, 'r') as f:
    episode = json.load(f)

joint_names = [j + ".revolute.bone" for j in episode["Joints"]]

frame_offset = episode["Frame_offset"][0]
frame_size = episode["Frame_size"][0]
stride = sum(frame_size.values())

# ==== ENSURE POSE MODE ====
bpy.ops.object.mode_set(mode='POSE')

# ==== APPLY ANIMATION ====
for frame_idx, flat_data in enumerate(episode["Frames"], start=1):
    bpy.context.scene.frame_set(frame_idx)

    # --- Pelvis root position ---
    root_start = frame_offset["root_pos"]
    root_end = root_start + frame_size["root_pos"]
    root_pos = flat_data[root_start:root_end]

    pelvis_bone = armature_obj.pose.bones.get(pelvis_bone_name)
    if pelvis_bone:
        pelvis_bone.location = (root_pos[0]*100, root_pos[1]*100, root_pos[2]*100 - 13)
        pelvis_bone.keyframe_insert(data_path="location", index=-1)

    # --- Pelvis root orientation (quaternion) ---
    quat_start = frame_offset["root_quat"]
    quat_end = quat_start + frame_size["root_quat"]
    root_quat = flat_data[quat_start:quat_end]

    if pelvis_bone:
        pelvis_bone.rotation_mode = 'QUATERNION'
        pelvis_bone.rotation_quaternion = pelvis_bone.rotation_quaternion = (root_quat[3], root_quat[0], root_quat[1], root_quat[2])
        pelvis_bone.keyframe_insert(data_path="rotation_quaternion", index=-1)

    # --- Joint angles ---
    joints_start = frame_offset["joints_pos"]
    joints_end = joints_start + frame_size["joints_pos"]
    joint_angles = flat_data[joints_start:joints_end]

    for bone_name, angle in zip(joint_names, joint_angles):
        bone = armature_obj.pose.bones.get(bone_name)
        if bone is None:
            print(f"Bone not found: {bone_name}")
            continue
        bone.rotation_mode = 'XYZ' 
        bone.rotation_euler.y = angle
        bone.keyframe_insert(data_path="rotation_euler", index=-1)

print("✅ Full animation (pelvis + joints) applied from JSON.")
