import json
import os
import subprocess
import argparse
import time
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from gait.gait_blending import blend_gait_parameters, vel_to_step, gait_sample_data, gait_sample_data_med_only

# Robot configurations: velocity limits and sweep parameters
ROBOT_CONFIGS = {
    "go_bdx": {
        "vel_max": (0.12, 0.06, 0.3),
        "sweep_min": (-0.08, -0.06, -0.3),
        "sweep_max": (0.12, 0.06, 0.3),
        "sweep_granularity": (0.04, 0.04, 0.1),
    },
    "mini_bdx": {
        "vel_max": (0.15, 0.15, 0.5),
        "sweep_min": (-0.1, -0.15, -0.5),
        "sweep_max": (0.15, 0.15, 0.5),
        "sweep_granularity": (0.05, 0.05, 0.2),
    },
}
ROBOT_CONFIGS["mini2_bdx"] = ROBOT_CONFIGS["mini_bdx"]
ROBOT_CONFIGS["dino"] = ROBOT_CONFIGS["mini_bdx"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../recordings")
LOG_DIR = os.path.join(OUTPUT_DIR, "log")


def run_command(cmd, log_file=None):
    """Run a command, optionally redirecting output to a log file."""
    if log_file:
        with open(log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    else:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)


def generate_velocities(args, config):
    """Generate list of (x_vel, y_vel, theta_vel) tuples based on args."""
    x_max, y_max, theta_max = config["vel_max"]
    
    # Single velocity mode
    if any(v is not None for v in [args.x_vel, args.y_vel, args.theta_vel]):
        return [(args.x_vel or 0.0, args.y_vel or 0.0, args.theta_vel or 0.0)]
    
    # Sweep mode
    if args.sweep:
        x_min, y_min, theta_min = config["sweep_min"]
        x_max, y_max, theta_max = config["sweep_max"]
        dx, dy, dtheta = config["sweep_granularity"]
        
        x_vals = np.arange(x_min, x_max + dx, dx)
        y_vals = np.arange(y_min, y_max + dy, dy)
        theta_vals = np.arange(theta_min, theta_max + dtheta, dtheta)
        
        return [(round(x, 2), round(y, 2), round(t, 2)) 
                for t in theta_vals for y in y_vals for x in x_vals]
    
    # Random mode
    return [(round(np.random.uniform(-x_max, x_max), 2),
             round(np.random.uniform(-y_max, y_max), 2),
             round(np.random.uniform(-theta_max, theta_max), 2))
            for _ in range(args.num)]


def create_blended_preset(base_data, velocities, sample_data, vel_max):
    """Create a blended preset from base data and target velocities."""
    x_vel, y_vel, theta_vel = velocities
    x_max, y_max, theta_max = vel_max
    
    params = blend_gait_parameters(sample_data, x_vel, y_vel, theta_vel, x_max, y_max, theta_max)
    x_step, y_step, theta_step, _ = vel_to_step(x_vel, y_vel, theta_vel, params)
    
    data = base_data.copy()
    data.update({
        'walk_trunk_pitch': params['walk_trunk_pitch'],
        'single_support_duration': params['single_support_duration'],
        'head_bob_amplitude': params['head_bob_amplitude'],
        'double_support_ratio': params['double_support_ratio'],
        'dx': round(x_step, 3),
        'dy': round(y_step, 3),
        'dtheta': round(theta_step, 3),
        'x_vel': x_vel,
        'y_vel': y_vel,
        'theta_vel': theta_vel,
        'joint_angles': {
            **data.get('joint_angles', {}),
            'neck_pitch': params['neck_pitch'],
            'head_pitch': params['head_pitch'],
        }
    })
    return data


def build_command(bdx_type, preset_path, index):
    """Build the gait_generator command for the given robot type."""
    cmd = ['python', 'gait_generator.py', '--preset', preset_path, '--name', str(index)]
    
    if bdx_type in ["mini_bdx", "mini2_bdx"]:
        cmd.append(f"--{bdx_type.split('_')[0]}")
    elif bdx_type == "dino":
        cmd.append("--dino")
    
    return cmd


def print_results():
    """Print speed results from generated recordings."""
    results = []
    for filename in os.listdir(OUTPUT_DIR):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(OUTPUT_DIR, filename)) as f:
            data = json.load(f)
        placo = data.get("Placo", {})
        speed = np.hypot(placo.get("avg_x_lin_vel", 0), placo.get("avg_y_lin_vel", 0))
        results.append((speed, placo.get("preset_name", "unknown")))
    
    # Sort by numeric prefix in preset name
    def sort_key(item):
        match = re.match(r"(\d+)(.*)", item[1])
        return (int(match.group(1)), match.group(2)) if match else (float("inf"), item[1])
    
    for speed, name in sorted(results, key=sort_key):
        print(f"Preset: {name}, Speed: {speed:.4f}")


def main(args):
    start_time = time.time()
    
    config = ROBOT_CONFIGS[args.bdx_type]
    sample_data = gait_sample_data_med_only if args.static_gait else gait_sample_data
    
    # Setup directories
    presets_dir = f"../awd/data/assets/{args.bdx_type}/placo_presets"
    tmp_dir = os.path.join(presets_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Load base preset
    with open(os.path.join(presets_dir, "medium.json")) as f:
        base_data = json.load(f)
    
    # Generate velocities
    velocities_list = generate_velocities(args, config)
    print(f"{'=' * 50}\n=== GENERATING {len(velocities_list)} MOVES WITH GAIT BLENDING ===\n{'=' * 50}")
    
    # Build commands
    commands = []
    for i, velocities in enumerate(velocities_list):
        preset_data = create_blended_preset(base_data, velocities, sample_data, config["vel_max"])
        preset_path = os.path.join(tmp_dir, f"{i}_medium_blend.json")
        
        with open(preset_path, 'w') as f:
            json.dump(preset_data, f, indent=4)
        
        cmd = build_command(args.bdx_type, preset_path, i)
        log_file = None if args.verbose else os.path.join(LOG_DIR, f"{i}.log")
        commands.append((cmd, log_file))
    
    # Execute commands
    if args.jobs > 1:
        print(f"Running {args.jobs} parallel jobs")
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            executor.map(lambda c: run_command(*c), commands)
    else:
        for cmd, log_file in commands:
            run_command(cmd, log_file)
    
    print_results()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMP data with gait blending")
    parser.add_argument("--bdx_type", choices=list(ROBOT_CONFIGS.keys()), required=True)
    parser.add_argument("--num", type=int, default=100, help="Number of motions to generate")
    parser.add_argument("--sweep", action="store_true", help="Sweep through velocity values")
    parser.add_argument("--static_gait", action="store_true", help="Use medium-only gait data")
    parser.add_argument("--x_vel", type=float, help="Single x velocity (m/s)")
    parser.add_argument("--y_vel", type=float, help="Single y velocity (m/s)")
    parser.add_argument("--theta_vel", type=float, help="Single theta velocity (rad/s)")
    parser.add_argument("-j", "--jobs", nargs="?", type=int, const=os.cpu_count(), default=1,
                        help="Parallel jobs (default: 1, -j alone uses all cores)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    main(parser.parse_args())