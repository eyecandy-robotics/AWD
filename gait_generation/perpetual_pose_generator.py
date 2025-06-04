import argparse
import placo
import time
import threading
import webbrowser
from ischedule import schedule, run_loop
from placo_utils.visualization import *


def rotate_matrix(matrix, angle_degrees, axis='x'):
    # Convert angle to radians
    angle = np.radians(angle_degrees)

    # Define rotation matrices
    if axis == 'x':
        rotation = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        rotation = np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    # Apply the rotation by matrix multiplication
    return matrix @ rotation

def open_browser():
    try:
        webbrowser.open_new('http://127.0.0.1:7000/static/')
    except:
        print("Failed to open the default browser. Trying Google Chrome.")
        try:
            webbrowser.get('google-chrome').open_new('http://127.0.0.1:7000/static/')
        except:
            print("Failed to open Google Chrome. Make sure it's installed and accessible.")

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("path", help="Path to the URDF")
args = arg_parser.parse_args()

robot = placo.RobotWrapper(args.path, placo.Flags.ignore_collisions)
# Creating the solver
solver = placo.KinematicsSolver(robot)
target_frame_name = "trunk_assembly"

# Adding a custom regularization task
regularization_task = solver.add_regularization_task(1e-4)

T_world_trunk = tf.translation_matrix([0.0, 0.0, 0.16])
effector_task = solver.add_frame_task(target_frame_name, T_world_trunk)
effector_task.configure(target_frame_name, "soft", 2.0, 1.0)

solver.solve(True)

print("Frame names:")
print(list(robot.frame_names()))
        
joints = {
    "left_hip_yaw":0.002,
    "left_hip_roll":0.053,
    "left_hip_pitch":-0.63,
    "left_knee":1.368,
    "left_ankle":-0.784,
    "neck_pitch":0.5,
    "head_pitch":-0.5,
    "head_yaw":0,
    "head_roll":0,
    "left_antenna":0,
    "right_antenna":0,
    "right_hip_yaw":-0.003,
    "right_hip_roll":-0.065,
    "right_hip_pitch":0.635,
    "right_knee":1.379,
    "right_ankle":-0.796, 
}

for joint in joints:
    robot.set_joint(joint, joints[joint])
robot.update_kinematics()

viz = robot_viz(robot)
t = 0
dt = 0.01
solver.dt = dt
threading.Timer(1, open_browser).start()

default_trunk_transform = robot.get_T_world_frame(target_frame_name)

# Retrieving initial position of the feet, com and trunk orientation
T_world_left = robot.get_T_world_frame("left_foot")
T_world_right = robot.get_T_world_frame("right_foot")

# Keep left and right foot on the floor
left_foot_task = solver.add_frame_task("left_foot", T_world_left)
left_foot_task.configure("left_foot", "soft", 1.0, 1e-3)

right_foot_task = solver.add_frame_task("right_foot", T_world_right)
right_foot_task.configure("right_foot", "soft", 1.0, 1e-3)

@schedule(interval=dt)
def loop():
    global t
    t += dt
 
    target = rotate_matrix(T_world_trunk, np.clip(10*np.sin(t), -10, 10), "z")
    target = rotate_matrix(target, np.clip(10*np.sin(t), -10, 10), "y")
    target = rotate_matrix(target, np.clip(10*np.sin(t), -10, 10), "x")
    target[2, 3] += 0.01 * np.sin(t) 

    # head_target = rotate_matrix(head_target, np.clip(15*np.sin(t), linear_range[0], linear_range[1]), "y")
    # head_target = rotate_matrix(head_target, np.clip(15*np.sin(t), linear_range[0], linear_range[1]), "x")

    effector_task.T_world_frame = target

    solver.solve(True)
    robot.update_kinematics()
    
    # joints = []
    # for joint_name in robot.joint_names():
    #     joints.append(robot.get_joint(joint_name))
    
    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    robot_frame_viz(robot, target_frame_name)
    frame_viz("target", effector_task.T_world_frame)


run_loop()
