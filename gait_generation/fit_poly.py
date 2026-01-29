import numpy as np
import json
from glob import glob
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--ref_motion", type=str, default="ref_motion")
args = parser.parse_args()

all_files = glob(f"{args.ref_motion}/*.json")


def process_ref_motion(file):
    data = json.load(open(file))
    Y_all = np.array(data["Frames"])
    period = data["Placo"]["period"]
    fps = data["FPS"]
    frame_offsets = data["Frame_offset"][0]
    startend_double_support_ratio = data["Placo"]["startend_double_support_ratio"]
    single_support_duration = data["Placo"]["single_support_duration"]
    double_support_ratio = data["Placo"]["double_support_ratio"]
    
    # Calculate period time for reference
    double_support_duration = single_support_duration * double_support_ratio
    startend_double_support_duration = single_support_duration * startend_double_support_ratio
    period_time = 2 * single_support_duration + 2 * double_support_duration
    nb_steps_in_period = round(period * fps)
    
    # Find warmup frames using foot contact data
    # foot_contacts: [left_contact, right_contact]
    # Left foot touches down when left_contact transitions from 0 to 1
    foot_contacts_offset = frame_offsets.get("foot_contacts")
    
    if foot_contacts_offset is not None:
        # Extract foot contact data from frames
        foot_contacts = Y_all[:, foot_contacts_offset:foot_contacts_offset + 2]
        left_contacts = foot_contacts[:, 0]
        
        # Find frames where left foot touches down (transition from no contact to contact)
        left_touchdown_frames = []
        for i in range(1, len(left_contacts)):
            if left_contacts[i-1] == 0 and left_contacts[i] == 1:
                left_touchdown_frames.append(i)
        
        # Start at the 3rd left foot touchdown for stable gait
        # (1st: startup, 2nd: stabilizing, 3rd: stable)
        num_touchdowns_to_skip = 2
        if len(left_touchdown_frames) > num_touchdowns_to_skip:
            warmup_frames = left_touchdown_frames[num_touchdowns_to_skip]
        else:
            # Fallback if not enough touchdowns found
            warmup_frames = int(round((startend_double_support_duration + 2 * period_time) * fps))
        
        # Compute ACTUAL period from foot contact data (not theoretical)
        # Measure average frames between consecutive left foot touchdowns after warmup
        post_warmup_touchdowns = [f - warmup_frames for f in left_touchdown_frames if f >= warmup_frames]
        
        if len(post_warmup_touchdowns) >= 2:
            # Calculate actual period from consecutive touchdown intervals
            touchdown_intervals = np.diff(post_warmup_touchdowns)
            actual_nb_steps_in_period = np.mean(touchdown_intervals)
            actual_period = actual_nb_steps_in_period / fps
        else:
            # Fallback to theoretical
            actual_nb_steps_in_period = period * fps
            actual_period = period
    else:
        # Fallback: calculate from timing parameters
        warmup_frames = int(round((startend_double_support_duration + 2 * period_time) * fps))
        actual_nb_steps_in_period = period * fps
        actual_period = period
    
    nb_steps_in_period = actual_nb_steps_in_period
    
    warmup_time = warmup_frames / fps
    
    # Start offset skips the warmup phase
    start_offset = warmup_frames
    
    meta = {}
    meta["Joints"] = data["Joints"]
    meta["Frame_offset"] = data["Frame_offset"]
    meta["Frame_size"] = data["Frame_size"]

    # Store the FULL reference data, not just one period
    # Also store period-only data for backward compatibility
    # Y_period = Y_all[start_offset : start_offset + int(nb_steps_in_period)]
    Y_all = Y_all[start_offset:]
    
    # Store the original motion data
    ret_data = {
        # "motion_data": Y_period.tolist(),  # One period of motion (for backward compatibility)
        "motion_data": Y_all.tolist(),  # Full reference motion data
        "period": period,  # Use actual measured period
        "period_time": period_time,
        "fps": fps,
        "frame_offsets": frame_offsets,
        "start_offset": start_offset,
        "warmup_frames": warmup_frames,
        "warmup_time": warmup_time,
        "nb_steps_in_period": nb_steps_in_period,
        "total_frames": len(Y_all),
        "startend_double_support_ratio": startend_double_support_ratio,
        "double_support_ratio": double_support_ratio,
        "single_support_duration": single_support_duration,
        "meta": meta,
    }

    return ret_data

all_motion_data = {}
n = 0
for file in all_files:
    name = os.path.basename(file).strip(".json")
    tmp = name.split("_")
    name = f"{tmp[1]}_{tmp[2]}_{tmp[3]}"

    all_motion_data[name] = process_ref_motion(file)
    n += 1

print(f"Processed {n} reference motions.")
pickle.dump(all_motion_data, open("original_motion_data.pkl", "wb"))