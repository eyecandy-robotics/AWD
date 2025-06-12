import bpy
import mathutils

# Central logic for FK/IK snapping
def move_effector_to_bone_tail(armature, source_bone_name, target_bone_name):
    pb_source = armature.pose.bones.get(source_bone_name)
    pb_target = armature.pose.bones.get(target_bone_name)

    mw = armature.matrix_world
    target_position_world = mw @ pb_source.tail
    target_matrix_world = mw @ pb_target.matrix

    new_matrix_world = target_matrix_world.copy()
    new_matrix_world.translation = target_position_world
    pb_target.matrix = mw.inverted() @ new_matrix_world

    bpy.context.view_layer.update()

def bake_ik_to_fk_at_current_frame(armature):
    """
    Bakes the IK-solved transforms into pose bones at the current frame.
    Only applies to the current frame and inserts keyframes.
    """
    scene = bpy.context.scene
    current_frame = scene.frame_current

    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_eval = armature.evaluated_get(depsgraph)

    for pb in armature.pose.bones:
        bone_eval = arm_eval.pose.bones[pb.name]
        pb.matrix = bone_eval.matrix

        # Insert keyframes for transform at current frame
        pb.keyframe_insert(data_path="location", frame=current_frame)
        pb.keyframe_insert(data_path="rotation_quaternion", frame=current_frame)
        pb.keyframe_insert(data_path="scale", frame=current_frame)

    bpy.context.view_layer.update()
    
def set_fk_ik_mode(armature, enable_ik: bool):
    """
    Enables IK or FK for both legs.
    If switching to FK, it bakes the IK-solved pose into bone matrices.
    """
    sides = ['left', 'right']
    ik_constraint_name = "IK"

    if not enable_ik:
        # Bake only at current frame
        bake_ik_to_fk_at_current_frame(armature)
        
        for side in sides:
            ankle = f"{side}_ankle.revolute.bone"
            pb = armature.pose.bones.get(ankle)
            if pb:
                ik_con = pb.constraints.get("IK")
                if ik_con and ik_con.type == 'IK':
                    ik_con.use_location = False

    else:
        for side in sides:
            ankle = f"{side}_ankle.revolute.bone"
            eff = f"{side}_eff"
            pb = armature.pose.bones.get(ankle)
            ik_con = pb.constraints.get(ik_constraint_name) if pb else None

            if pb and ik_con and ik_con.type == 'IK':
                # Snap effector to FK pose before enabling IK
                move_effector_to_bone_tail(armature, ankle, eff)
                ik_con.use_location = True


# Global property to track mode
def register_fk_ik_property():
    bpy.types.Scene.fk_ik_mode = bpy.props.EnumProperty(
        name="FK/IK Mode",
        description="Current control mode",
        items=[
            ('FK', "FK", ""),
            ('IK', "IK", "")
        ],
        default='IK'
    )


class OBJECT_OT_set_fk_ik_mode(bpy.types.Operator):
    bl_idname = "object.set_fk_ik_mode"
    bl_label = "Set FK/IK Mode"
    bl_description = "Switch between FK and IK"
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(
        items=[
            ('FK', "FK", ""),
            ('IK', "IK", "")
        ],
        name="Mode"
    )

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE' or context.mode != 'POSE':
            self.report({'ERROR'}, "Must be in Pose Mode with an Armature selected.")
            return {'CANCELLED'}

        context.scene.fk_ik_mode = self.mode
        set_fk_ik_mode(arm, enable_ik=(self.mode == 'IK'))

        return {'FINISHED'}


class VIEW3D_PT_fk_ik_toggle(bpy.types.Panel):
    bl_label = "FK/IK Switch"
    bl_idname = "VIEW3D_PT_fk_ik_toggle"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Rig Tools 1'

    def draw(self, context):
        layout = self.layout
        layout.label(text="Leg Mode:")

        row = layout.row(align=True)
        row.operator(
            "object.set_fk_ik_mode",
            text="FK",
            depress=(context.scene.fk_ik_mode == 'FK')
        ).mode = 'FK'

        row.operator(
            "object.set_fk_ik_mode",
            text="IK",
            depress=(context.scene.fk_ik_mode == 'IK')
        ).mode = 'IK'


def register():
    bpy.utils.register_class(OBJECT_OT_set_fk_ik_mode)
    bpy.utils.register_class(VIEW3D_PT_fk_ik_toggle)
    register_fk_ik_property()


def unregister():
    bpy.utils.unregister_class(OBJECT_OT_set_fk_ik_mode)
    bpy.utils.unregister_class(VIEW3D_PT_fk_ik_toggle)
    del bpy.types.Scene.fk_ik_mode


if __name__ == "__main__":
    register()

