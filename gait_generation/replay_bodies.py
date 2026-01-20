import argparse
import json
import time

import FramesViewer.utils as fv_utils
import numpy as np
from FramesViewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True)
parser.add_argument("--loop", action="store_true", help="Loop the animation")
args = parser.parse_args()

# Body names in order
body_names = ['base', 'body_front', 'headdown', 'front_tail', 'right_upper_cover', 'left_upper_cover', 'left_foot_tpu', 'right_foot_tpu']

# Load episode
episode = json.load(open(args.file))
frame_duration = episode["FrameDuration"]
frames = episode["Frames"]
frame_offsets = episode["Frame_offset"][0]

# Get slices for body data
body_pos_start = frame_offsets["body_pos_w"]
body_quat_start = frame_offsets["body_quat_w"]

# Start viewer
fv = Viewer()
fv.start()

print(f"Loaded {len(frames)} frames at {1/frame_duration:.1f} FPS")
print(f"Bodies: {body_names}")

# Playback loop
while True:
    for i, frame in enumerate(frames):
        # Extract body positions (8 bodies * 3 coords = 24 values)
        body_pos_w = frame[body_pos_start:body_pos_start + len(body_names) * 3]
        # Extract body quaternions (8 bodies * 4 coords = 32 values)
        body_quat_w = frame[body_quat_start:body_quat_start + len(body_names) * 4]
        
        # Visualize each body
        for j, body_name in enumerate(body_names):
            pos = body_pos_w[j*3:(j+1)*3]
            quat = body_quat_w[j*4:(j+1)*4]  # x, y, z, w
            
            # Build 4x4 transform matrix
            pose = np.eye(4)
            pose[:3, 3] = pos
            pose[:3, :3] = R.from_quat(quat).as_matrix()
            
            fv.pushFrame(pose, body_name)
        
        time.sleep(frame_duration)
    
    if not args.loop:
        break

print("Playback finished. Viewer still running...")
input("Press Enter to exit...")
