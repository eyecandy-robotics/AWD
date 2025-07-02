import xml.etree.ElementTree as ET
import math
import bpy
import os

def rad2deg(rad):
    return math.degrees(float(rad))

def parse_urdf_joint_limits(urdf_path):
    limits = {}
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for joint in root.findall('joint'):
        name = joint.attrib.get('name')
        joint_type = joint.attrib.get('type')

        if joint_type != 'revolute':
            continue

        limit_elem = joint.find('limit')
        if limit_elem is not None:
            lower = limit_elem.attrib.get('lower')
            upper = limit_elem.attrib.get('upper')

            limits[name] = {
                "lower": rad2deg(lower) if lower is not None else None,
                "upper": rad2deg(upper) if upper is not None else None
            }

    return limits

def apply_limits_to_pose_bones(limits_dict):
    armature = bpy.context.object

    if armature.type != 'ARMATURE':
        raise RuntimeError("Selected object is not an armature")

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    for joint_name, data in limits_dict.items():
        bone_name = f"{joint_name}.revolute.bone"

        if bone_name not in armature.pose.bones:
            print(f"Bone not found: {bone_name}")
            continue

        pb = armature.pose.bones[bone_name]

        # Lock X and Z
        pb.lock_ik_x = True
        pb.lock_ik_z = True
        pb.lock_ik_y = False  # Y is allowed to move

        # Set Y-axis limits
        lower = data.get("lower")
        upper = data.get("upper")

        if lower is not None and upper is not None:
            pb.use_ik_limit_y = True
            pb.ik_min_y = math.radians(lower)
            pb.ik_max_y = math.radians(upper)
            print(f"[{bone_name}] Set Y limit: {lower:.1f}° to {upper:.1f}°")
        else:
            pb.use_ik_limit_y = False
            print(f"[{bone_name}] Warning: No Y limits found.")

    bpy.ops.object.mode_set(mode='OBJECT')

# ---------- USAGE ----------

urdf_path = "/home/mankaran/Desktop/rl/AWD/awd/data/assets/dino/dino.urdf"

try:
    joint_limits = parse_urdf_joint_limits(urdf_path)
    apply_limits_to_pose_bones(joint_limits)
    print("IK limits successfully applied.")
except Exception as e:
    print(f"Error: {e}")


