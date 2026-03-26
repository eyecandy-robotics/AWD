"""
Visualize phase vs foot contact/swing periods for reference motions.
Helps verify that phase alignment is consistent across different commands.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob


def _get_joint_index(joints, joint_name):
    try:
        return joints.index(joint_name)
    except ValueError:
        return None


def _estimate_ref_from_phase(signal, base_wave):
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.array([])

    center = float(np.mean(signal))
    centered = signal - center
    amp = float(0.5 * (np.percentile(signal, 95) - np.percentile(signal, 5)))
    if amp < 1e-6:
        amp = float(np.std(centered) * np.sqrt(2.0))
    if amp < 1e-6:
        amp = 0.0

    corr = float(np.dot(centered, base_wave))
    sign = 1.0 if abs(corr) < 1e-9 else np.sign(corr)
    return center + sign * amp * base_wave


def load_and_analyze_motion(file):
    """Load a motion file and extract phase/contact info."""
    data = json.load(open(file))
    Y_all = np.array(data["Frames"])
    
    period = data["Placo"]["period"]
    fps = data["FPS"]
    frame_offsets = data["Frame_offset"][0]
    frame_sizes = data.get("Frame_size", [{}])[0]
    joints = data.get("Joints", [])
    
    dx = data["Placo"]["dx"]
    dy = data["Placo"]["dy"]
    dtheta = data["Placo"]["dtheta"]
    
    single_support_duration = data["Placo"]["single_support_duration"]
    double_support_ratio = data["Placo"]["double_support_ratio"]
    startend_double_support_ratio = data["Placo"]["startend_double_support_ratio"]
    
    double_support_duration = single_support_duration * double_support_ratio
    startend_double_support_duration = single_support_duration * startend_double_support_ratio
    period_time = 2 * single_support_duration + 2 * double_support_duration
    
    foot_contacts_offset = frame_offsets.get("foot_contacts")
    joints_pos_offset = frame_offsets.get("joints_pos")
    joints_pos_size = frame_sizes.get("joints_pos", 0)
    
    if foot_contacts_offset is None:
        print(f"No foot_contacts data in {file}")
        return None

    if joints_pos_offset is None or joints_pos_size < 1:
        print(f"No joints_pos data in {file}")
        return None
    
    # Extract foot contact data
    foot_contacts = Y_all[:, foot_contacts_offset:foot_contacts_offset + 2]
    left_contacts = foot_contacts[:, 0]
    right_contacts = foot_contacts[:, 1]

    neck_pitch = None

    neck_idx = _get_joint_index(joints, "neck_pitch")

    if neck_idx is not None:
        neck_pitch = Y_all[:, joints_pos_offset + neck_idx]
    
    # Find all left foot touchdown frames for warmup detection AND period measurement
    left_touchdown_frames = []
    for i in range(1, len(left_contacts)):
        if left_contacts[i-1] == 0 and left_contacts[i] == 1:
            left_touchdown_frames.append(i)
    
    num_touchdowns_to_skip = 2
    if len(left_touchdown_frames) > num_touchdowns_to_skip:
        warmup_frames = left_touchdown_frames[num_touchdowns_to_skip]
    else:
        warmup_frames = int(round((startend_double_support_duration + 2 * period_time) * fps))
    
    # Compute ACTUAL period from foot contact data (not theoretical)
    # Measure average frames between consecutive left foot touchdowns after warmup
    post_warmup_touchdowns = [f - warmup_frames for f in left_touchdown_frames if f >= warmup_frames]
    
    if len(post_warmup_touchdowns) >= 2:
        # Calculate actual period from consecutive touchdown intervals
        touchdown_intervals = np.diff(post_warmup_touchdowns)
        actual_nb_steps_in_period = np.mean(touchdown_intervals)
        actual_period = actual_nb_steps_in_period / fps
        print(f"  Theoretical period: {period:.4f}s ({period * fps:.1f} frames)")
        print(f"  Actual period:      {actual_period:.4f}s ({actual_nb_steps_in_period:.1f} frames)")
    else:
        # Fallback to theoretical
        actual_nb_steps_in_period = period * fps
        actual_period = period
    
    # Trim to start from warmup
    left_contacts = left_contacts[warmup_frames:]
    right_contacts = right_contacts[warmup_frames:]
    if neck_pitch is not None:
        neck_pitch = neck_pitch[warmup_frames:]
    
    num_frames = len(left_contacts)
    
    # Create time and phase arrays using ACTUAL measured period
    time = np.arange(num_frames) / fps
    phase = (np.arange(num_frames) % actual_nb_steps_in_period) / actual_nb_steps_in_period
    
    return {
        "file": os.path.basename(file),
        "dx": dx,
        "dy": dy,
        "dtheta": dtheta,
        "time": time,
        "phase": phase,
        "left_contacts": left_contacts,
        "right_contacts": right_contacts,
        "neck_pitch": neck_pitch,
        "period": actual_period,
        "theoretical_period": period,
        "period_time": period_time,
        "fps": fps,
        "warmup_frames": warmup_frames,
        "nb_steps_in_period": actual_nb_steps_in_period,
    }


def plot_single_motion(motion_data, ax=None, num_periods=2):
    """Plot foot contacts vs phase for a single motion."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    dx, dy, dtheta = motion_data["dx"], motion_data["dy"], motion_data["dtheta"]
    time = motion_data["time"]
    phase = motion_data["phase"]
    left = motion_data["left_contacts"]
    right = motion_data["right_contacts"]
    neck_pitch = motion_data["neck_pitch"]
    period = motion_data["period"]
    nb_steps = motion_data["nb_steps_in_period"]
    
    # Limit to num_periods
    max_frames = int(num_periods * nb_steps)
    time = time[:max_frames]
    phase = phase[:max_frames]
    left = left[:max_frames]
    right = right[:max_frames]
    if neck_pitch is not None:
        neck_pitch = neck_pitch[:max_frames]
    
    # Create x-axis as phase (with period count)
    x = np.arange(len(time)) / nb_steps  # This gives us 0, 1, 2, ... periods
    phase_in_period = x % 1.0

    # Reference matching fit_poly.py neck bob regeneration logic
    bob_wave = np.sin(4 * np.pi * phase_in_period)
    neck_ref = _estimate_ref_from_phase(neck_pitch, bob_wave) if neck_pitch is not None else None
    
    # Plot contacts as filled regions
    ax.fill_between(x, 0, left * 0.4, alpha=0.7, color='blue', label='Left contact', step='mid')
    ax.fill_between(x, 0.5, 0.5 + right * 0.4, alpha=0.7, color='red', label='Right contact', step='mid')
    
    # Add phase markers
    for i in range(int(num_periods) + 1):
        ax.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
        ax.text(i, 1.0, f'φ=0', ha='center', va='bottom', fontsize=8)
    
    # Add half-period markers
    for i in range(int(num_periods)):
        ax.axvline(x=i + 0.5, color='gray', linestyle=':', alpha=0.3)
        ax.text(i + 0.5, 1.0, f'φ=0.5', ha='center', va='bottom', fontsize=8, alpha=0.5)
    
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, num_periods)
    ax.set_xlabel('Period')
    ax.set_ylabel('Foot Contact')
    ax.set_yticks([0.2, 0.7])
    ax.set_yticklabels(['Left', 'Right'])
    ax.set_title(f'dx={dx:.3f}, dy={dy:.3f}, dθ={dtheta:.3f} (period={period:.3f}s)')
    ax.grid(True, alpha=0.3)

    # Neck pitch overlay on secondary axis
    neck_ax = ax.twinx()

    if neck_pitch is not None:
        neck_line = neck_ax.plot(x, neck_pitch, color='green', linewidth=1.8, label='Neck pitch')[0]
    if neck_ref is not None:
        neck_ref_line = neck_ax.plot(x, neck_ref, color='green', linestyle='--', linewidth=1.2, alpha=0.8,
                                     label='Neck ref (2x/period)')[0]

    neck_ax.set_ylabel('Neck Pitch (rad)', color='green')
    neck_ax.tick_params(axis='y', labelcolor='green')

    # Combined legend from both axes
    handles_main, labels_main = ax.get_legend_handles_labels()
    handles_neck, labels_neck = neck_ax.get_legend_handles_labels()
    ax.legend(handles_main + handles_neck, labels_main + labels_neck, loc='upper right')
    
    return ax


def plot_phase_comparison(motion_files, num_periods=2):
    """Plot multiple motions for comparison."""
    motions = []
    for f in motion_files:
        m = load_and_analyze_motion(f)
        if m is not None:
            motions.append(m)
    
    if not motions:
        print("No valid motions to plot")
        return
    
    n = len(motions)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    
    for ax, motion in zip(axes, motions):
        plot_single_motion(motion, ax, num_periods)
    
    plt.tight_layout()
    plt.savefig('phase_visualization.png', dpi=150)
    print("Saved phase_visualization.png")
    plt.show()


def plot_phase_grid(ref_motion_dir, num_periods=2):
    """Plot a grid comparing +/- directions."""
    # Find specific motions for comparison
    all_files = glob(f"{ref_motion_dir}/*.json")
    
    # Categorize by direction
    left_motions = []  # +dy
    right_motions = []  # -dy
    forward_motions = []  # +dx
    backward_motions = []  # -dx
    
    for f in all_files:
        m = load_and_analyze_motion(f)
        if m is None:
            continue
        
        if abs(m["dy"]) > 0.01 and abs(m["dx"]) < 0.01:
            if m["dy"] > 0:
                left_motions.append(m)
            else:
                right_motions.append(m)
        elif abs(m["dx"]) > 0.01 and abs(m["dy"]) < 0.01:
            if m["dx"] > 0:
                forward_motions.append(m)
            else:
                backward_motions.append(m)
    
    # Pick one from each category
    selected = []
    if forward_motions:
        selected.append(forward_motions[0])
    if backward_motions:
        selected.append(backward_motions[0])
    if left_motions:
        selected.append(left_motions[0])
    if right_motions:
        selected.append(right_motions[0])
    
    if not selected:
        print("No directional motions found. Plotting first 4 available.")
        for f in all_files[:4]:
            m = load_and_analyze_motion(f)
            if m:
                selected.append(m)
    
    if not selected:
        print("No valid motions to plot")
        return
    
    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    
    for ax, motion in zip(axes, selected):
        plot_single_motion(motion, ax, num_periods)
    
    plt.suptitle('Phase vs Foot Contact Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('phase_comparison.png', dpi=150)
    print("Saved phase_comparison.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize phase vs foot contacts")
    parser.add_argument("--ref_motion", type=str, default="ref_motion", 
                        help="Directory containing reference motion JSON files")
    parser.add_argument("--files", type=str, nargs="+", default=None,
                        help="Specific files to plot")
    parser.add_argument("--periods", type=float, default=2,
                        help="Number of periods to show")
    parser.add_argument("--drift", action="store_true",
                        help="Show longer time (10 periods) to check for phase drift")
    parser.add_argument("--grid", action="store_true",
                        help="Show comparison grid of +/- directions")
    parser.add_argument("-n", "--num_files", type=int, default=6,
                        help="Number of random files to plot")
    args = parser.parse_args()
    
    # Use more periods for drift check
    num_periods = 20 if args.drift else args.periods
    
    if args.files:
        plot_phase_comparison(args.files, num_periods)
    elif args.grid:
        plot_phase_grid(args.ref_motion, num_periods)
    else:
        # Plot random files for variety
        all_files = glob(f"{args.ref_motion}/*.json")
        if all_files:
            # Randomly sample files to get diverse commands
            import random
            num_to_plot = min(args.num_files, len(all_files))
            random_files = random.sample(all_files, num_to_plot)
            plot_phase_comparison(random_files, num_periods)
        else:
            print(f"No JSON files found in {args.ref_motion}/")
