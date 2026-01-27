#!/usr/bin/env python3
"""
Simple fast joystick control for robot gaits.
Joystick 1: Forward/Backward + Strafe (X/Y)
Joystick 2: Rotation (Theta)

Performance Optimizations:
- Frame skipping: Updates viz every 3rd frame (3x faster rendering)
- Minimal viz: Only robot mesh, no footsteps or coordinate frames
- Smart parameter updates: Only updates when values change (reduces CPU load)
- High FPS: 60-120 FPS target for smooth visualization
"""

import argparse
from flask import Flask, render_template, request, jsonify
import numpy as np
import webbrowser
import threading
import json
import os
import time
import logging

import placo
from placo_utils.visualization import footsteps_viz, robot_frame_viz, robot_viz
from scipy.spatial.transform import Rotation as R

from gait.placo_walk_engine import PlacoWalkEngine
from gait.gait_blending import blend_gait_parameters, gait_sample_data, vel_to_step

parser = argparse.ArgumentParser()
parser.add_argument("--dino", action="store_true", default=True, help="Use dino robot (default)")
parser.add_argument("--go_bdx", action="store_true", help="Use go_bdx robot")
parser.add_argument("--mini_bdx", action="store_true", help="Use mini_bdx robot")
parser.add_argument("--mini2_bdx", action="store_true", help="Use mini2_bdx robot")
args = parser.parse_args()

app = Flask(__name__)

# Suppress Flask's default logging for status endpoint (reduces spam)
class StatusFilter(logging.Filter):
    def filter(self, record):
        return '/api/status' not in record.getMessage()

log = logging.getLogger('werkzeug')
log.addFilter(StatusFilter())

# Control parameters - FAST response
CONTROL_UPDATE_RATE = 60  # Hz - faster updates
DT = 0.001  # Simulation timestep
MESHCAT_FPS = 100  # Faster visualization

# Script path for finding assets
script_path = os.path.dirname(os.path.abspath(__file__))

class RobotController:
    """Manages robot configuration and gait parameters"""
    def __init__(self, robot_type='dino'):
        self.robot_type = robot_type
        self.update_robot_paths(robot_type)
        
        # Current commanded velocities (from joystick)
        self.dx = 0.0
        self.dy = 0.0
        self.dtheta = 0.0
        
        # Maximum velocities for normalization
        self.x_max = 0.15
        self.y_max = 0.15
        self.theta_max = 0.5
        
        # Load gait samples (you can customize these per robot)
        self.gait_samples = gait_sample_data
        
        # Current blended parameters
        self.current_G = None
        
        # Walk engine
        self.pwe = None
        self.viz = None
        
    def update_robot_paths(self, robot_type):
        """Update asset paths based on robot type"""
        self.robot_type = robot_type
        if robot_type == 'mini_bdx':
            self.robot_urdf = "urdf/bdx.urdf"
            self.asset_path = os.path.join(script_path, "../awd/data/assets/mini_bdx")
        elif robot_type == 'mini2_bdx':
            self.robot_urdf = "mini2_bdx.urdf"
            self.asset_path = os.path.join(script_path, "../awd/data/assets/mini2_bdx")
        elif robot_type == 'dino':
            self.robot_urdf = "dino.urdf"
            self.asset_path = os.path.join(script_path, "../awd/data/assets/dino")
        else:  # go_bdx
            self.robot_urdf = "go_bdx.urdf"
            self.asset_path = os.path.join(script_path, "../awd/data/assets/go_bdx")
        
        # Load default parameters for this robothttp://127.0.0.1:7000/static/
        defaults_file = os.path.join(self.asset_path, "placo_defaults.json")
        if os.path.exists(defaults_file):
            with open(defaults_file, 'r') as f:
                self.default_params = json.load(f)
        else:
            self.default_params = {}
    
    def initialize_walk_engine(self):
        """Initialize the Placo walk engine"""
        self.pwe = PlacoWalkEngine(self.asset_path, self.robot_urdf, self.default_params)
        self.pwe.set_traj(0, 0, 0)
        
        # Setup visualization (lightweight for speed)
        self.viz = robot_viz(self.pwe.robot)
        self.viz.display(self.pwe.robot.state.q)
        
        print(f"✓ Walk engine initialized for {self.robot_type}")
    
    def update_velocity_from_joystick(self, joy_x, joy_y, joy_theta):
        """
        Update desired velocities from joystick input.
        joy_x, joy_y, joy_theta are in range [-1, 1]
        """
        self.dx = joy_x * self.x_max
        self.dy = joy_y * self.y_max
        self.dtheta = joy_theta * self.theta_max
    
    def blend_parameters(self):
        """Use gait blending to compute smooth parameters"""
        self.current_G = blend_gait_parameters(
            self.gait_samples,
            self.dx, self.dy, self.dtheta,
            self.x_max, self.y_max, self.theta_max,
            use_quintic=False  # Smoother C2 continuity
        )
        return self.current_G
    
    def apply_parameters_to_walk_engine(self):
        """Apply the blended parameters to the walk engine (optimized to skip redundant updates)"""
        if self.current_G is None or self.pwe is None:
            return
        
        # Initialize cache on first call
        if not hasattr(self, '_last_G'):
            self._last_G = None
            self._last_velocities = (None, None, None)
        
        # Check if parameters or velocities actually changed
        params_changed = self._last_G != self.current_G
        velocity_changed = (self.dx, self.dy, self.dtheta) != self._last_velocities
        
        # Skip update if nothing changed (saves CPU when joystick is idle)
        if not params_changed and not velocity_changed:
            return
        
        # Apply parameters only if they changed
        if params_changed:
            if 'walk_trunk_pitch' in self.current_G:
                self.pwe.parameters.walk_trunk_pitch = np.deg2rad(self.current_G['walk_trunk_pitch'])
            if 'single_support_duration' in self.current_G:
                self.pwe.parameters.single_support_duration = self.current_G['single_support_duration']
            if 'double_support_ratio' in self.current_G:
                self.pwe.parameters.double_support_ratio = self.current_G['double_support_ratio']
            if 'head_bob_amplitude' in self.current_G:
                # Could apply head bob if walk engine supports it
                pass
            self._last_G = self.current_G.copy()
        
        # Update trajectory only if velocities changed
        if velocity_changed:
            # Convert velocities to step sizes using gait parameters
            x_step, y_step, theta_step, period = vel_to_step(
                self.dx, self.dy, self.dtheta, self.current_G
            )
            # Set trajectory with computed step sizes (not raw velocities)
            self.pwe.set_traj(x_step, y_step, theta_step)
            self._last_velocities = (self.dx, self.dy, self.dtheta)
    
    def get_status(self):
        """Get current controller status"""
        return {
            'robot': self.robot_type,
            'dx': round(self.dx, 4),
            'dy': round(self.dy, 4),
            'dtheta': round(self.dtheta, 4),
            'x_max': self.x_max,
            'y_max': self.y_max,
            'theta_max': self.theta_max,
            'parameters': self.current_G if self.current_G else {}
        }

# Global controller instance
controller = RobotController(
    robot_type='dino' if args.dino else 
                'go_bdx' if args.go_bdx else 
                'mini_bdx' if args.mini_bdx else 
                'mini2_bdx' if args.mini2_bdx else 
                'dino'
)

# Threading control
run_loop = False
control_thread_running = False
control_lock = threading.Lock()

def open_browser():
    """Open browser to the joystick control interface"""
    try:
        webbrowser.open_new('http://127.0.0.1:5000/')
    except:
        print("Failed to open the default browser. Trying Google Chrome.")
        try:
            webbrowser.get('google-chrome').open_new('http://127.0.0.1:5000/')
        except:
            print("Failed to open Google Chrome. Make sure it's installed and accessible.")

# Flask routes
@app.route('/')
def index():
    """Main joystick control page"""
    return render_template('joystick.html')

@app.route('/api/joystick', methods=['POST'])
def update_joystick():
    """Update velocity from joystick input"""
    data = request.get_json()
    joy_x = float(data.get('x', 0))
    joy_y = float(data.get('y', 0))
    joy_theta = float(data.get('theta', 0))
    
    with control_lock:
        controller.update_velocity_from_joystick(joy_x, joy_y, joy_theta)
    
    return jsonify({'status': 'ok'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current controller status (high-frequency polling, logging suppressed)"""
    with control_lock:
        status = controller.get_status()
    return jsonify(status)

@app.route('/api/start', methods=['POST'])
def start_control():
    """Start the control loop"""
    global run_loop
    run_loop = True
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
def stop_control():
    """Stop the control loop"""
    global run_loop
    run_loop = False
    with control_lock:
        controller.dx = 0
        controller.dy = 0
        controller.dtheta = 0
    return jsonify({'status': 'stopped'})

@app.route('/api/set_limits', methods=['POST'])
def set_limits():
    """Update velocity limits"""
    data = request.get_json()
    with control_lock:
        if 'x_max' in data:
            controller.x_max = float(data['x_max'])
        if 'y_max' in data:
            controller.y_max = float(data['y_max'])
        if 'theta_max' in data:
            controller.theta_max = float(data['theta_max'])
    return jsonify({'status': 'ok'})

def control_loop_thread():
    """Main control loop that runs the walk engine"""
    global run_loop, control_thread_running
    
    print("Initializing walk engine...")
    controller.initialize_walk_engine()
    print("✓ Ready! Open browser to control the robot.")
    
    control_thread_running = True
    last_update = time.time()
    last_viz_update = time.time()
    frame_counter = 0  # Frame skipping for faster rendering
    
    while control_thread_running:
        if not run_loop:
            time.sleep(0.01)
            continue
        
        # Blend parameters based on current velocities
        with control_lock:
            controller.blend_parameters()
            controller.apply_parameters_to_walk_engine()
        
        # Tick the walk engine
        controller.pwe.tick(DT)
        
        # Update visualization - OPTIMIZED for speed:
        # 1. Skip every 3rd frame (frame skipping)
        # 2. Only show robot, no footsteps or frames
        current_time = time.time()
        frame_counter += 1
        
        if frame_counter % 3 == 0 and current_time - last_viz_update >= 1.0 / MESHCAT_FPS:
            controller.viz.display(controller.pwe.robot.state.q)
            last_viz_update = current_time
        
        # Small sleep to prevent CPU hogging
        time.sleep(DT)

if __name__ == '__main__':
    print("="*60)
    print("  Robot Joystick Control with Gait Blending")
    print("="*60)
    print(f"Robot: {controller.robot_type}")
    print(f"Max velocities: x={controller.x_max}, y={controller.y_max}, θ={controller.theta_max}")
    print("\nStarting control thread...")
    
    # Start control thread
    thread = threading.Thread(target=control_loop_thread, daemon=True)
    thread.start()
    
    # Wait a bit for initialization
    time.sleep(2)
    
    # Open browser after a delay
    threading.Timer(1, open_browser).start()
    
    print("\n✓ Server starting on http://127.0.0.1:5000")
    print("✓ MeshCat viewer: http://127.0.0.1:7000/static/")
    print("\nPress Ctrl+C to exit")
    
    # Start Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)
