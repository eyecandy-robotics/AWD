import json
import os
import subprocess
import numpy as np
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import re
from gait.gait_blending import blend_gait_parameters, vel_to_step, gait_sample_data, gait_sample_data_med_only

def run_command_with_logging(cmd_log_tuple):
    cmd, log_file = cmd_log_tuple
    if log_file is None:
        print(f"cmd: {cmd}")
        subprocess.run(cmd)
    else:
        with open(log_file, "w") as outfile:
            subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT)

def numeric_prefix_sort_key(item):
    total_speed, preset_name = item
    match = re.match(r"(\d+)(.*)", preset_name)
    if match:
        number_part = int(match.group(1))
        rest_part = match.group(2)
        return (number_part, rest_part)
    return (float("inf"), preset_name)

def load_sample_presets(bdx_type, static_gait=False):
    
    # Create nine dummy samples with (x, y, theta) and a parameter dict G
    if bdx_type == "dino":
        sample_data = [
            {'x':  0.0, 'y':  0.0, 'theta':  0.0, 'G': {'walk_trunk_pitch': -4.0, 'single_support_duration': 0.22,
                                                        'head_bob_amplitude':0.12, 'neck_pitch':0.0, 'head_pitch':-0.0,
                                                        'double_support_ratio':0.5}}, # in-place
            {'x':  0.05, 'y':  0.0, 'theta':  0.0, 'G': {'walk_trunk_pitch': -4.0, 'single_support_duration': 0.22,
                                                        'head_bob_amplitude':0.12, 'neck_pitch':0.0, 'head_pitch':-0.0,
                                                        'double_support_ratio':0.5}}, # forward slow
            {'x':  0.1, 'y':  0.0, 'theta':  0.0, 'G': {'walk_trunk_pitch': -3.0, 'single_support_duration': 0.20,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.0, 'head_pitch':-0.0,
                                                        'double_support_ratio':0.5}}, # forward med
            {'x':  0.15, 'y':  0.0, 'theta':  0.0, 'G': {'walk_trunk_pitch': 5.0, 'single_support_duration': 0.18,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.0, 'head_pitch':-0.0,
                                                        'double_support_ratio':0.5}}, # forward fast
            {'x':  0.0, 'y':  0.15, 'theta':  0.0, 'G': {'walk_trunk_pitch': 5.0, 'single_support_duration': 0.18,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.0, 'head_pitch':-0.0,
                                                        'double_support_ratio':0.5}}, # strafe right
            {'x':  0.0, 'y': -0.15, 'theta':  0.0, 'G': {'walk_trunk_pitch': 5.0, 'single_support_duration': 0.18,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.0, 'head_pitch':-0.0,
                                                        'double_support_ratio':0.5}}, # strafe left
            {'x':  0.0, 'y':  0.0, 'theta':  0.5, 'G': {'walk_trunk_pitch': 5.0, 'single_support_duration': 0.18,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.0, 'head_pitch':-0.0,
                                                        'double_support_ratio':0.5}}, # turn right
            {'x':  0.0, 'y':  0.0, 'theta': -0.5, 'G': {'walk_trunk_pitch': 5.0, 'single_support_duration': 0.18,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.0, 'head_pitch':-0.0,
                                                        'double_support_ratio':0.5}}, # turn left
            {'x': -0.1, 'y':  0.0, 'theta':  0.0, 'G': {'walk_trunk_pitch': 0.0, 'single_support_duration': 0.20,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.0, 'head_pitch':-0.0,
                                                        'double_support_ratio':0.5}}, # backward med
                    ]
    elif bdx_type in ["go_bdx", "mini_bdx", "mini2_bdx"]:
        sample_data = [
            {'x':  0.0, 'y':  0.0, 'theta':  0.0, 'G': {'walk_trunk_pitch': -4.0, 'single_support_duration': 0.22,
                                                        'head_bob_amplitude':0.15, 'neck_pitch':0.5, 'head_pitch':-0.4,
                                                        'double_support_ratio':0.5}}, # in-place
            {'x':  0.05, 'y':  0.0, 'theta':  0.0, 'G': {'walk_trunk_pitch': -4.0, 'single_support_duration': 0.21,
                                                        'head_bob_amplitude':0.15, 'neck_pitch':0.5, 'head_pitch':-0.4,
                                                        'double_support_ratio':0.5}}, # forward slow
            {'x':  0.1, 'y':  0.0, 'theta':  0.0, 'G': {'walk_trunk_pitch': -3.0, 'single_support_duration': 0.20,
                                                        'head_bob_amplitude':0.125, 'neck_pitch':0.25, 'head_pitch':-0.25,
                                                        'double_support_ratio':0.5}}, # forward med
            {'x':  0.15, 'y':  0.0, 'theta':  0.0, 'G': {'walk_trunk_pitch': 3.0, 'single_support_duration': 0.18,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.1, 'head_pitch':-0.1,
                                                        'double_support_ratio':0.5}}, # forward fast
            {'x':  0.0, 'y':  0.15, 'theta':  0.0, 'G': {'walk_trunk_pitch': 3.0, 'single_support_duration': 0.18,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.1, 'head_pitch':-0.1,
                                                        'double_support_ratio':0.5}}, # strafe right
            {'x':  0.0, 'y': -0.15, 'theta':  0.0, 'G': {'walk_trunk_pitch': 3.0, 'single_support_duration': 0.18,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.1, 'head_pitch':-0.1,
                                                        'double_support_ratio':0.5}}, # strafe left
            {'x':  0.0, 'y':  0.0, 'theta':  0.5, 'G': {'walk_trunk_pitch': 3.0, 'single_support_duration': 0.18,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.1, 'head_pitch':-0.1,
                                                        'double_support_ratio':0.5}}, # turn right
            {'x':  0.0, 'y':  0.0, 'theta': -0.5, 'G': {'walk_trunk_pitch': 3.0, 'single_support_duration': 0.18,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.1, 'head_pitch':-0.1,
                                                        'double_support_ratio':0.5}}, # turn left
            {'x': -0.1, 'y':  0.0, 'theta':  0.0, 'G': {'walk_trunk_pitch': 0.0, 'single_support_duration': 0.20,
                                                        'head_bob_amplitude':0.1, 'neck_pitch':0.15, 'head_pitch':-0.15,
                                                        'double_support_ratio':0.5}}, # backward med
                    ]
    if static_gait:
        # Use only the medium preset for static gait
        sample_data = gait_sample_data_med_only
                    
    return sample_data

def main(args):
    start_time = time.time()

    if args.bdx_type == "go_bdx":
        slow = 0.221
        medium = 0.336
        fast = 0.568
        # Define velocity limits instead of step size
        x_vel_max = 0.12  # m/s
        y_vel_max = 0.06  # m/s
        theta_vel_max = 0.3  # rad/s
        # Hardcoded sweep parameters for go_bdx (now in velocity space)
        min_sweep_x_vel = -0.08
        max_sweep_x_vel = 0.12
        min_sweep_y_vel = -0.06
        max_sweep_y_vel = 0.06
        min_sweep_theta_vel = -0.3
        max_sweep_theta_vel = 0.3
        sweep_xy_vel_granularity = 0.04
        sweep_theta_vel_granularity = 0.1
    elif args.bdx_type == "mini_bdx" or args.bdx_type == "mini2_bdx" or args.bdx_type == "dino":
        slow = 0.05
        medium = 0.1
        fast = 0.15
        # Define velocity limits instead of step size
        x_vel_max = 0.15  # m/s
        y_vel_max = 0.15  # m/s
        theta_vel_max = 0.5  # rad/s
        # Hardcoded sweep parameters in velocity space
        min_sweep_x_vel = -0.1
        max_sweep_x_vel = 0.15
        min_sweep_y_vel = -0.15
        max_sweep_y_vel = 0.15
        min_sweep_theta_vel = -0.5
        max_sweep_theta_vel = 0.5
        sweep_xy_vel_granularity = 0.05
        sweep_theta_vel_granularity = 0.2
    else:
        raise ValueError("Invalid bdx_type. Choose either 'go_bdx', 'mini_bdx', or 'mini2_bdx'.")

    presets_dir = f"../awd/data/assets/{args.bdx_type}/placo_presets"
    tmp_dir = os.path.join(presets_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Create log directory
    script_path = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_path, "../recordings")
    log_dir = os.path.join(default_output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    # Load sample presets for gait blending
    sample_data = load_sample_presets(args.bdx_type, args.static_gait)
    
    # Use medium as base preset
    base_preset = "medium"

    if args.sweep:
        # Create sweep in velocity space
        x_vels = np.arange(min_sweep_x_vel, max_sweep_x_vel + sweep_xy_vel_granularity, sweep_xy_vel_granularity)
        y_vels = np.arange(min_sweep_y_vel, max_sweep_y_vel + sweep_xy_vel_granularity, sweep_xy_vel_granularity)
        theta_vels = np.arange(min_sweep_theta_vel, max_sweep_theta_vel + sweep_theta_vel_granularity, sweep_theta_vel_granularity)
        all_n = len(x_vels) * len(y_vels) * len(theta_vels)
    else:
        all_n = args.num

    nb_moves_message = f"=== GENERATING {all_n} MOVES WITH GAIT BLENDING ==="
    spacer = "=" * len(nb_moves_message)
    print(spacer)
    print(nb_moves_message)
    print(spacer)

    commands = []
    for i in range(all_n):
        # Load the base preset
        with open(os.path.join(presets_dir, f"{base_preset}.json")) as file:
            data = json.load(file)

        if args.sweep:
            x_idx = i % len(x_vels)
            y_idx = (i // len(x_vels)) % len(y_vels)
            theta_idx = (i // (len(x_vels) * len(y_vels))) % len(theta_vels)

            # Use velocities directly for blending
            x_vel = round(x_vels[x_idx], 2)
            y_vel = round(y_vels[y_idx], 2)
            theta_vel = round(theta_vels[theta_idx], 2)
        else:
            # Generate random velocities within limits
            x_vel = round(np.random.uniform(-x_vel_max, x_vel_max), 2)
            y_vel = round(np.random.uniform(-y_vel_max, y_vel_max), 2)
            theta_vel = round(np.random.uniform(-theta_vel_max, theta_vel_max), 2)

        # Blend gait parameters based on velocities
        blended_params = blend_gait_parameters(sample_data, x_vel, y_vel, theta_vel, x_vel_max, y_vel_max, theta_vel_max)
        
        # Update preset with blended parameters
        data['walk_trunk_pitch'] = blended_params['walk_trunk_pitch']
        data['single_support_duration'] = blended_params['single_support_duration']
        data['head_bob_amplitude'] = blended_params['head_bob_amplitude']
        data['double_support_ratio'] = blended_params['double_support_ratio']
        
        if 'joint_angles' not in data:
            data['joint_angles'] = {}
        data['joint_angles']['neck_pitch'] = blended_params['neck_pitch']
        data['joint_angles']['head_pitch'] = blended_params['head_pitch']
        
        # Convert velocities to step sizes
        x_step, y_step, theta_step, period = vel_to_step(x_vel, y_vel, theta_vel, blended_params)
        
        # Update preset with step sizes
        data['dx'] = round(x_step, 3)
        data['dy'] = round(y_step, 3)
        data['dtheta'] = round(theta_step, 3)

        # Add extra debugging data to keep track of original velocities
        data['x_vel'] = x_vel
        data['y_vel'] = y_vel
        data['theta_vel'] = theta_vel

        tmp_preset = os.path.join(tmp_dir, f"{i}_{base_preset}_blend.json")
        with open(tmp_preset, 'w') as file:
            json.dump(data, file, indent=4)

        if args.bdx_type in ["mini_bdx", "mini2_bdx"]:
            cmd = ['python', "gait_generator.py", "--preset", f"{tmp_preset}", 
                   "--name", f"{i}", f"--{args.bdx_type.split('_')[0]}"]
        elif args.bdx_type in ["dino"]:
            cmd = ['python', "gait_generator.py", "--preset", f"{tmp_preset}", 
                   "--name", f"{i}", "--dino"]
        else:
            cmd = ['python', "gait_generator.py", "--preset", f"{tmp_preset}", 
                   "--name", f"{i}"]
                           
        log_file = None if args.verbose else os.path.join(log_dir, f"{i}.log")
        commands.append((cmd, log_file))

    # Execute commands either sequentially or in parallel
    if args.jobs > 1:
        print(f"Running {args.jobs} parallel jobs")
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            executor.map(run_command_with_logging, commands)
    else:
        for cmd in commands:
            run_command_with_logging(cmd)

    # Check and filter recordings
    totals = []
    for filename in os.listdir(default_output_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(default_output_dir, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                
            # Extract the relevant information
            placo_data = data.get("Placo", {})
            avg_x_vel = placo_data.get("avg_x_lin_vel", 0)
            avg_y_vel = placo_data.get("avg_y_lin_vel", 0)
            preset_name = placo_data.get("preset_name", "unknown")
            
            # Calculate the total speed
            total_speed = np.sqrt(avg_x_vel**2 + avg_y_vel**2)
            totals.append((total_speed, preset_name))


    # Sort and display the results
    totals = sorted(totals, key=numeric_prefix_sort_key)
    for speed, preset_name in totals:
        print(f"Preset: {preset_name}, Total Speed: {speed:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMP data with gait blending")
    parser.add_argument("--bdx_type", choices=["go_bdx", "mini_bdx", "mini2_bdx", "dino"], required=True, help="Type of BDX to generate data for")
    parser.add_argument("--num", type=int, default=100, help="Number of motion files to generate.")
    parser.add_argument("--sweep", action="store_true", help="Sweep through the velocity values.")
    parser.add_argument("--static_gait", action="store_true", help="use med_only gait sample data instead of blending")
    parser.add_argument("-j", "--jobs", nargs="?", type=int, const=os.cpu_count(), default=1,
                       help="Number of parallel jobs. If -j is provided without a number, "
                            "uses the number of CPU cores available. Default is 1.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    main(args)