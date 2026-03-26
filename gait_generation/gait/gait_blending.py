import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import json
from pathlib import Path


def f_soft(alpha, clamp=True):
    """
    Cubic easing function (Hermite smoothstep): f_soft(alpha) = -2*alpha^3 + 3*alpha^2
    Ensures C1 continuity with zero derivative at alpha=0 and alpha=1.

    Args:
        alpha: Interpolation parameter (should be in [0, 1])
        clamp: If True, clamps alpha to [0, 1] for numerical safety

    Returns:
        Smoothly interpolated value in [0, 1]
    """
    if clamp:
        alpha = np.clip(alpha, 0.0, 1.0)
    return -2 * alpha**3 + 3 * alpha**2


def f_smooth(alpha, clamp=True):
    """
    Quintic easing function (smoother step): 6*alpha^5 - 15*alpha^4 + 10*alpha^3
    Ensures C2 continuity with zero first AND second derivatives at endpoints.
    Provides even smoother transitions than f_soft.

    Args:
        alpha: Interpolation parameter (should be in [0, 1])
        clamp: If True, clamps alpha to [0, 1] for numerical safety

    Returns:
        Smoothly interpolated value in [0, 1]
    """
    if clamp:
        alpha = np.clip(alpha, 0.0, 1.0)
    return 6 * alpha**5 - 15 * alpha**4 + 10 * alpha**3


def blend_dicts(GA, GB, alpha):
    """
    Linearly blend two parameter dictionaries GA and GB using weight alpha.
    Supports both scalar values and numpy arrays.

    Args:
        GA: First parameter dictionary
        GB: Second parameter dictionary (must have same keys as GA)
        alpha: Blend weight in [0, 1]. alpha=0 returns GA, alpha=1 returns GB

    Returns:
        New dictionary with blended values: (1-alpha)*GA + alpha*GB
    """
    G_interp = {}
    one_minus_alpha = 1.0 - alpha
    for key in GA:
        val_a = GA[key]
        val_b = GB[key]
        if isinstance(val_a, np.ndarray) or isinstance(val_b, np.ndarray):
            G_interp[key] = one_minus_alpha * np.asarray(val_a) + alpha * np.asarray(val_b)
        else:
            G_interp[key] = one_minus_alpha * val_a + alpha * val_b
    return G_interp


def find_bracketing_samples_1D(samples, key, target):
    """
    Given a list of sample dicts with numeric attribute 'key', find two samples A, B
    such that sample_A[key] <= target <= sample_B[key], assuming samples sorted by 'key'.
    Returns indices i, i+1. If target is outside range, clamps to endpoints.
    """
    values = np.array([s[key] for s in samples])
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]

    if target <= sorted_vals[0]:
        return sorted_idx[0], sorted_idx[0]
    if target >= sorted_vals[-1]:
        return sorted_idx[-1], sorted_idx[-1]

    for i in range(len(sorted_vals) - 1):
        if sorted_vals[i] <= target <= sorted_vals[i + 1]:
            return sorted_idx[i], sorted_idx[i + 1]
    return sorted_idx[-2], sorted_idx[-1]


def find_circular_bracket(betas, beta, eps=1e-9):
    """
    Given a list of betas in [0, 2π), find indices i, j so that
    beta ∈ [betas[i], betas[j]] in a circular sense.

    Args:
        betas: List of angles in [0, 2π)
        beta: Target angle to bracket
        eps: Small tolerance for numerical comparison

    Returns:
        Tuple (i, j) of indices into the original betas list
    """
    betas = np.array(betas)
    n = len(betas)

    if n == 0:
        raise ValueError("betas list cannot be empty")
    if n == 1:
        return 0, 0

    beta = beta % (2 * np.pi)

    sorted_idx = np.argsort(betas)
    sorted_betas = betas[sorted_idx]

    for k in range(n - 1):
        if sorted_betas[k] - eps <= beta <= sorted_betas[k + 1] + eps:
            return sorted_idx[k], sorted_idx[k + 1]

    return sorted_idx[-1], sorted_idx[0]


def blend_gait_parameters(samples, x, y, theta, x_max, y_max, theta_max, use_quintic=False):
    """
    Performs three-step blending on gait samples following the paper's methodology.

    The blending operates in a normalized velocity space and uses smooth easing
    functions to ensure continuous parameter transitions.

    Args:
        samples: List of sample dicts, each with keys:
            - 'x': forward speed
            - 'y': strafe speed
            - 'theta': turn rate
            - 'G': dict of gait parameters
        x: Desired forward velocity
        y: Desired strafe velocity
        theta: Desired turn rate
        x_max: Maximum forward velocity (for normalization)
        y_max: Maximum strafe velocity (for normalization)
        theta_max: Maximum turn rate (for normalization)
        use_quintic: If True, use C2-continuous quintic smoothing instead of cubic.

    Returns:
        Final blended parameter dictionary G_interp
    """
    ease_fn = f_smooth if use_quintic else f_soft

    x_axis_samples = [s for s in samples if abs(s['y']) < 1e-6 and abs(s['theta']) < 1e-6]

    Ax_idx, Bx_idx = find_bracketing_samples_1D(x_axis_samples, 'x', x)
    xA, xB = x_axis_samples[Ax_idx]['x'], x_axis_samples[Bx_idx]['x']

    if abs(xB - xA) < 1e-9:
        alpha_x = 0.0
    else:
        alpha_lin_x = (x - xA) / (xB - xA)
        alpha_x = ease_fn(alpha_lin_x)

    G_Ax = x_axis_samples[Ax_idx]['G']
    G_Bx = x_axis_samples[Bx_idx]['G']
    G_fwd = blend_dicts(G_Ax, G_Bx, alpha_x)

    theta_n = theta / theta_max if abs(theta_max) > 1e-9 else 0.0
    y_n = y / y_max if abs(y_max) > 1e-9 else 0.0

    rho_n = min(np.sqrt(theta_n**2 + y_n**2), 1.0)
    beta = np.arctan2(y_n, theta_n)
    if beta < 0:
        beta += 2 * np.pi

    in_place_sample = None
    for s in samples:
        if abs(s['x']) < 1e-6 and abs(s['y']) < 1e-6 and abs(s['theta']) < 1e-6:
            in_place_sample = s
            break

    cardinal_samples = []
    cardinal_betas = []
    for s in samples:
        if abs(s['x']) < 1e-6:
            tn = s['theta'] / theta_max if abs(theta_max) > 1e-9 else 0.0
            yn = s['y'] / y_max if abs(y_max) > 1e-9 else 0.0
            is_cardinal = (abs(abs(tn) - 1) < 1e-6 and abs(yn) < 1e-6) or \
                          (abs(abs(yn) - 1) < 1e-6 and abs(tn) < 1e-6)
            if is_cardinal:
                cardinal_samples.append(s)
                b = np.arctan2(yn, tn)
                if b < 0:
                    b += 2 * np.pi
                cardinal_betas.append(b)

    if len(cardinal_samples) == 0:
        G_cardinal = in_place_sample['G'] if in_place_sample else G_fwd
    elif len(cardinal_samples) == 1:
        G_cardinal = cardinal_samples[0]['G']
    else:
        A_beta_idx, B_beta_idx = find_circular_bracket(cardinal_betas, beta)
        betaA = cardinal_betas[A_beta_idx]
        betaB = cardinal_betas[B_beta_idx]

        if betaB < betaA:
            betaB += 2 * np.pi
            if beta < betaA:
                beta += 2 * np.pi

        delta_beta = betaB - betaA
        if abs(delta_beta) < 1e-9:
            alpha_beta = 0.0
        else:
            alpha_lin_beta = (beta - betaA) / delta_beta
            alpha_beta = ease_fn(alpha_lin_beta)

        G_Ab = cardinal_samples[A_beta_idx]['G']
        G_Bb = cardinal_samples[B_beta_idx]['G']
        G_cardinal = blend_dicts(G_Ab, G_Bb, alpha_beta)

    if in_place_sample is not None:
        G_in_place = in_place_sample['G']
        alpha_rho = ease_fn(rho_n)
        G_ts = blend_dicts(G_in_place, G_cardinal, alpha_rho)
    else:
        G_ts = G_cardinal

    x_n = x / x_max if abs(x_max) > 1e-9 else 0.0
    alpha_fwd = ease_fn(abs(x_n))

    G_final = blend_dicts(G_ts, G_fwd, alpha_fwd)
    return G_final


def compute_phase_period(single_support_duration, double_support_ratio):
    """
    Computes the phase period based on gait parameters.
    single_support_duration: Duration of single support phase in seconds.
    double_support_ratio: Ratio of double support phase to total step duration.
    Returns the total step duration.
    """
    double_support_duration = single_support_duration * double_support_ratio
    period = 2*single_support_duration + 2*double_support_duration
    return period


def steps_to_vel(step_size, period):
    return (step_size * 2) / period


def vel_to_step(x_vel, y_vel, theta_vel, G):
    """
    Computes the step size based on velocities and gait parameters.
    x_vel: Forward velocity.
    y_vel: Strafe velocity.
    theta_vel: Turn rate.
    G: Dictionary of gait parameters.

    Returns the step size as a tuple (x_step, y_step).
    """
    single_support_duration = G['single_support_duration']
    double_support_ratio = G['double_support_ratio']

    period = compute_phase_period(single_support_duration, double_support_ratio)

    x_step = period * x_vel / 2
    y_step = period * y_vel / 2
    theta_step = period * theta_vel / 2

    return x_step, y_step, theta_step, period


def load_gait_blending_config(robot="mini_bdx", defaults_path=None):
    """Load gait blending samples/config from a robot placo_defaults.json file."""
    if defaults_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        defaults_path = repo_root / "awd" / "data" / "assets" / robot / "placo_defaults.json"
    else:
        defaults_path = Path(defaults_path)

    with open(defaults_path, "r") as f:
        defaults = json.load(f)

    blend = defaults.get("gait_blending")
    if not blend:
        raise ValueError(f"Missing gait_blending in {defaults_path}")

    samples_dynamic = blend.get("samples_dynamic")
    samples_static = blend.get("samples_static")
    if not samples_dynamic or not samples_static:
        raise ValueError(f"Missing samples_dynamic/samples_static in {defaults_path}")

    vel_max = blend.get(
        "vel_max",
        [
            defaults.get("walk_max_dx_forward", 0.1),
            defaults.get("walk_max_dy", 0.1),
            defaults.get("walk_max_dtheta", 1.0),
        ],
    )

    backward_max = defaults.get("walk_max_dx_backward", vel_max[0])
    sweep_min = blend.get("sweep_min", [-backward_max, -vel_max[1], -vel_max[2]])
    sweep_max = blend.get("sweep_max", [vel_max[0], vel_max[1], vel_max[2]])

    return {
        "samples_dynamic": samples_dynamic,
        "samples_static": samples_static,
        "vel_max": tuple(vel_max),
        "sweep_min": tuple(sweep_min),
        "sweep_max": tuple(sweep_max),
        "defaults_path": str(defaults_path),
    }


def plot_gait_cycle_vs_velocity(samples, x_min, x_max, y_min, y_max, theta_min, theta_max, num_points=100):
    """
    Plot all gait parameters vs x, y, and theta velocities.

    Args:
        samples: List of gait sample dicts
        x_min, x_max: Range for forward velocity
        y_min, y_max: Range for strafe velocity
        theta_min, theta_max: Range for turn rate
        num_points: Number of points to sample
    """

    param_names = list(samples[0]['G'].keys())
    n_params = len(param_names)

    x_vals = np.linspace(x_min, x_max, num_points)
    x_params = {name: [] for name in param_names}

    for x in x_vals:
        G = blend_gait_parameters(samples, x, 0.0, 0.0, x_max, y_max, theta_max)
        for name in param_names:
            x_params[name].append(G[name])

    y_vals = np.linspace(y_min, y_max, num_points)
    y_params = {name: [] for name in param_names}

    for y in y_vals:
        G = blend_gait_parameters(samples, 0.0, y, 0.0, x_max, y_max, theta_max)
        for name in param_names:
            y_params[name].append(G[name])

    theta_vals = np.linspace(theta_min, theta_max, num_points)
    theta_params = {name: [] for name in param_names}

    for theta in theta_vals:
        G = blend_gait_parameters(samples, 0.0, 0.0, theta, x_max, y_max, theta_max)
        for name in param_names:
            theta_params[name].append(G[name])

    fig, axes = plt.subplots(n_params, 3, figsize=(16, 3 * n_params))

    colors = plt.cm.tab10(np.linspace(0, 1, n_params))

    for i, name in enumerate(param_names):
        axes[i, 0].plot(x_vals, x_params[name], color=colors[i], linewidth=2)
        axes[i, 0].set_ylabel(name.replace('_', ' ').title(), fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            axes[i, 0].set_title('vs Forward Velocity (y=0, θ=0)', fontsize=12)
        if i == n_params - 1:
            axes[i, 0].set_xlabel('Forward Velocity (m/s)', fontsize=10)

        axes[i, 1].plot(y_vals, y_params[name], color=colors[i], linewidth=2)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            axes[i, 1].set_title('vs Strafe Velocity (x=0, θ=0)', fontsize=12)
        if i == n_params - 1:
            axes[i, 1].set_xlabel('Strafe Velocity (m/s)', fontsize=10)

        axes[i, 2].plot(theta_vals, theta_params[name], color=colors[i], linewidth=2)
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].axvline(0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            axes[i, 2].set_title('vs Turn Rate (x=0, y=0)', fontsize=12)
        if i == n_params - 1:
            axes[i, 2].set_xlabel('Turn Rate (rad/s)', fontsize=10)

    plt.suptitle('Gait Parameters vs Velocity', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('gait_params_vs_velocity.png', dpi=150, bbox_inches='tight')
    plt.show()

    x_periods = [2 * x_params['single_support_duration'][i] * (1 + x_params['double_support_ratio'][i])
                 for i in range(num_points)]
    y_periods = [2 * y_params['single_support_duration'][i] * (1 + y_params['double_support_ratio'][i])
                 for i in range(num_points)]
    theta_periods = [2 * theta_params['single_support_duration'][i] * (1 + theta_params['double_support_ratio'][i])
                     for i in range(num_points)]

    return {
        'x': (x_vals, x_periods),
        'y': (y_vals, y_periods),
        'theta': (theta_vals, theta_periods)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect and plot gait blending continuity")
    parser.add_argument("--robot", default="mini_bdx", choices=["go_bdx", "mini_bdx", "mini2_bdx", "dino", "dinoJr"])
    parser.add_argument("--defaults", type=str, default=None, help="Optional explicit path to placo_defaults.json")
    parser.add_argument("--static_gait", action="store_true", help="Use samples_static instead of samples_dynamic")
    parser.add_argument("--x", type=float, default=0.1, help="Target x velocity for final blend print")
    parser.add_argument("--y", type=float, default=0.1, help="Target y velocity for final blend print")
    parser.add_argument("--theta", type=float, default=0.4, help="Target theta velocity for final blend print")
    args = parser.parse_args()

    cfg = load_gait_blending_config(robot=args.robot, defaults_path=args.defaults)
    sample_data = cfg["samples_static"] if args.static_gait else cfg["samples_dynamic"]
    x_max, y_max, theta_max = cfg["vel_max"]
    x_min, y_min, theta_min = cfg["sweep_min"]
    x_plot_max, y_plot_max, theta_plot_max = cfg["sweep_max"]

    x_in, y_in, theta_in = args.x, args.y, args.theta
    print(f"Loaded blending config from: {cfg['defaults_path']}")

    print("Generating gait cycle vs velocity plots...")
    results = plot_gait_cycle_vs_velocity(
        sample_data,
        x_min=x_min, x_max=x_plot_max,
        y_min=y_min, y_max=y_plot_max,
        theta_min=theta_min, theta_max=theta_plot_max
    )

    for axis, (vals, periods) in results.items():
        print(f"{axis}: velocity range [{vals[0]:.3f}, {vals[-1]:.3f}], period range [{min(periods):.3f}, {max(periods):.3f}] s")

    print("\nGenerating 2D continuity heatmaps...")

    n_grid = 50
    x_range = np.linspace(x_min, x_plot_max, n_grid)
    y_range = np.linspace(y_min, y_plot_max, n_grid)
    theta_range = np.linspace(theta_min, theta_plot_max, n_grid)

    period_xy = np.zeros((n_grid, n_grid))
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            G = blend_gait_parameters(sample_data, x, y, 0.0, x_max, y_max, theta_max)
            ssd = G['single_support_duration']
            dsr = G['double_support_ratio']
            period_xy[i, j] = 2 * ssd * (1 + dsr)

    period_xtheta = np.zeros((n_grid, n_grid))
    for i, theta in enumerate(theta_range):
        for j, x in enumerate(x_range):
            G = blend_gait_parameters(sample_data, x, 0.0, theta, x_max, y_max, theta_max)
            ssd = G['single_support_duration']
            dsr = G['double_support_ratio']
            period_xtheta[i, j] = 2 * ssd * (1 + dsr)

    period_ytheta = np.zeros((n_grid, n_grid))
    for i, theta in enumerate(theta_range):
        for j, y in enumerate(y_range):
            G = blend_gait_parameters(sample_data, 0.0, y, theta, x_max, y_max, theta_max)
            ssd = G['single_support_duration']
            dsr = G['double_support_ratio']
            period_ytheta[i, j] = 2 * ssd * (1 + dsr)

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes2[0].imshow(period_xy, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
                          origin='lower', aspect='auto', cmap='viridis')
    axes2[0].set_xlabel('X velocity (m/s)')
    axes2[0].set_ylabel('Y velocity (m/s)')
    axes2[0].set_title('Gait Period: X-Y plane (θ=0)')
    axes2[0].axhline(0, color='white', linestyle='--', alpha=0.5)
    axes2[0].axvline(0, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(im0, ax=axes2[0], label='Period (s)')

    im1 = axes2[1].imshow(period_xtheta, extent=[x_range[0], x_range[-1], theta_range[0], theta_range[-1]],
                          origin='lower', aspect='auto', cmap='viridis')
    axes2[1].set_xlabel('X velocity (m/s)')
    axes2[1].set_ylabel('Theta (rad/s)')
    axes2[1].set_title('Gait Period: X-θ plane (y=0)')
    axes2[1].axhline(0, color='white', linestyle='--', alpha=0.5)
    axes2[1].axvline(0, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(im1, ax=axes2[1], label='Period (s)')

    im2 = axes2[2].imshow(period_ytheta, extent=[y_range[0], y_range[-1], theta_range[0], theta_range[-1]],
                          origin='lower', aspect='auto', cmap='viridis')
    axes2[2].set_xlabel('Y velocity (m/s)')
    axes2[2].set_ylabel('Theta (rad/s)')
    axes2[2].set_title('Gait Period: Y-θ plane (x=0)')
    axes2[2].axhline(0, color='white', linestyle='--', alpha=0.5)
    axes2[2].axvline(0, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(im2, ax=axes2[2], label='Period (s)')

    plt.tight_layout()
    plt.savefig('gait_2d_heatmaps.png', dpi=150)
    plt.show()
    print("2D heatmaps saved to gait_2d_heatmaps.png")

    G_output = blend_gait_parameters(sample_data, x_in, y_in, theta_in, x_max, y_max, theta_max)
    print("\nFinal blended G:")
    for param, value in G_output.items():
        print(f"  {param}: {value:.4f}")

    start_time = time.time()
    x_step, y_step, theta_step, period = vel_to_step(x_in, y_in, theta_in, G_output)
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    print(f"vel_to_step execution time: {elapsed_ms:.3f} ms")
    print(f"\nComputed step sizes from velocities:")

    num_iterations = 1000
    times = []

    for _ in range(num_iterations):
        start_time = time.time()
        x_step, y_step, theta_step, period = vel_to_step(x_in, y_in, theta_in, G_output)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"vel_to_step benchmarks ({num_iterations} iterations):")
    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    print(f"  Max: {max_time:.3f} ms")
    print(f"  x_step: {x_step:.4f}, y_step: {y_step:.4f}, theta_step: {theta_step:.4f}, period: {period:.4f}")
