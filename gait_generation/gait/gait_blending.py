import numpy as np

def f_soft(alpha):
    """
    Cubic easing function f_soft(alpha) = -2*alpha^3 + 3*alpha^2
    Ensures zero derivative at alpha=0 and alpha=1.
    """
    return -2 * alpha**3 + 3 * alpha**2

def blend_dicts(GA, GB, alpha):
    """
    Linearly blend two parameter dictionaries GA and GB using weight alpha.
    Assumes both GA and GB have identical keys with scalar values.
    Returns a new dictionary with blended values.
    """
    G_interp = {}
    for key in GA:
        G_interp[key] = (1 - alpha) * GA[key] + alpha * GB[key]
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
    
    # If target is below or above range, clamp to endpoints
    if target <= sorted_vals[0]:
        return sorted_idx[0], sorted_idx[0]
    if target >= sorted_vals[-1]:
        return sorted_idx[-1], sorted_idx[-1]
    
    # Otherwise find the interval
    for i in range(len(sorted_vals) - 1):
        if sorted_vals[i] <= target <= sorted_vals[i + 1]:
            return sorted_idx[i], sorted_idx[i + 1]
    # Fallback (shouldn't happen)
    return sorted_idx[-2], sorted_idx[-1]

def find_circular_bracket(betas, beta):
    """
    Given a sorted list of betas in [0, 2π), find indices i, j so that
    beta ∈ [betas[i], betas[j]] in a circular sense. Assumes len(betas)=4 sorted ascending.
    Returns indices (i, j) into the original betas list.
    """
    # Ensure sorted with corresponding indices
    sorted_idx = np.argsort(betas)
    sorted_betas = np.array(betas)[sorted_idx]
    # Check each interval [sorted_betas[i], sorted_betas[i+1]]
    for k in range(len(sorted_betas) - 1):
        if sorted_betas[k] <= beta <= sorted_betas[k + 1]:
            return sorted_idx[k], sorted_idx[k + 1]
    # Check wrap-around interval [sorted_betas[-1], sorted_betas[0] + 2π]
    if beta >= sorted_betas[-1] or beta <= sorted_betas[0]:
        return sorted_idx[-1], sorted_idx[0]
    # Fallback
    return sorted_idx[-2], sorted_idx[-1]

def blend_gait_parameters(samples, x, y, theta, x_max, y_max, theta_max, eta=0.5):
    """
    Performs three-step blending on nine gait samples, each with keys:
      - 'x': forward speed
      - 'y': strafe speed
      - 'theta': turn rate
      - 'G': dict of gait parameters (e.g. double-stand-duration, trunk-pitch, etc.)
    
    Steps:
      1) Blend forward/reverse using x.
      2) Blend turn/strafe using (y, theta).
      3) Blend the two intermediate results based on normalized x and radial norm of (y, theta).
    
    Returns final G_interp dict.
    """
    # ===== Step 1: Forward/Reverse by x =====
    Ax_idx, Bx_idx = find_bracketing_samples_1D(samples, 'x', x)
    xA, xB = samples[Ax_idx]['x'], samples[Bx_idx]['x']
    
    if xB - xA == 0:
        alpha_x = 0.0
    else:
        alpha_lin_x = (x - xA) / (xB - xA)
        alpha_x = f_soft(alpha_lin_x)
    
    G_Ax = samples[Ax_idx]['G']
    G_Bx = samples[Bx_idx]['G']
    G_fwd = blend_dicts(G_Ax, G_Bx, alpha_x)
    
    # ===== Step 2: Turn/Strafe in (theta_n, y_n) plane =====
    # Normalize
    theta_n = theta / theta_max
    y_n = y / y_max
    rho_n = min(np.sqrt(theta_n**2 + y_n**2), 1.0)
    beta = np.arctan2(y_n, theta_n)
    if beta < 0:
        beta += 2 * np.pi
    
    # Identify the four cardinal samples
    cardinal_samples = []
    cardinal_betas = []
    for s in samples:
        if abs(s['x']) < 1e-6:
            tn = s['theta'] / theta_max
            yn = s['y'] / y_max
            # Check (tn, yn) on unit circle
            if (abs(abs(tn) - 1) < 1e-6 and abs(yn) < 1e-6) or \
               (abs(abs(yn) - 1) < 1e-6 and abs(tn) < 1e-6):
                cardinal_samples.append(s)
                b = np.arctan2(yn, tn)
                if b < 0:
                    b += 2 * np.pi
                cardinal_betas.append(b)
    
    # Find the two cardinal samples whose betas bracket beta
    A_beta_idx, B_beta_idx = find_circular_bracket(cardinal_betas, beta)
    betaA = cardinal_betas[A_beta_idx]
    betaB = cardinal_betas[B_beta_idx]
    # If wrap-around, add 2π for correct interpolation
    if betaB < betaA:
        betaB += 2 * np.pi
    
    alpha_lin_beta = (beta - betaA) / (betaB - betaA)
    alpha_beta = f_soft(alpha_lin_beta)
    
    G_Ab = cardinal_samples[A_beta_idx]['G']
    G_Bb = cardinal_samples[B_beta_idx]['G']
    G_turn = blend_dicts(G_Ab, G_Bb, alpha_beta)
    
    # ===== Step 3: Fuse G_fwd and G_turn =====
    x_n = x / x_max
    alpha_inner = f_soft(abs(x_n))
    scale = 1 - eta * alpha_inner
    alpha_ts = f_soft(rho_n * scale)
    
    G_final = blend_dicts(G_fwd, G_turn, alpha_ts)
    return G_final

# --- Example Usage with Dummy Sample Data ---

# Create nine dummy samples with (x, y, theta) and a parameter dict G
sample_data = [
    {'x':  0.0, 'y':  0.0, 'theta':  0.0, 'G': {'double_stand': 0.30, 'trunk_pitch': 5.0}}, # in-place
    {'x':  0.5, 'y':  0.0, 'theta':  0.0, 'G': {'double_stand': 0.28, 'trunk_pitch': 4.5}}, # forward slow
    {'x':  1.0, 'y':  0.0, 'theta':  0.0, 'G': {'double_stand': 0.25, 'trunk_pitch': 4.0}}, # forward med
    {'x':  1.5, 'y':  0.0, 'theta':  0.0, 'G': {'double_stand': 0.22, 'trunk_pitch': 3.5}}, # forward fast
    {'x':  0.0, 'y':  1.0, 'theta':  0.0, 'G': {'double_stand': 0.29, 'trunk_pitch': 4.2}}, # strafe right
    {'x':  0.0, 'y': -1.0, 'theta':  0.0, 'G': {'double_stand': 0.29, 'trunk_pitch': 4.2}}, # strafe left
    {'x':  0.0, 'y':  0.0, 'theta':  1.0, 'G': {'double_stand': 0.32, 'trunk_pitch': 5.5}}, # turn right
    {'x':  0.0, 'y':  0.0, 'theta': -1.0, 'G': {'double_stand': 0.32, 'trunk_pitch': 5.5}}, # turn left
    {'x': -0.5, 'y':  0.0, 'theta':  0.0, 'G': {'double_stand': 0.35, 'trunk_pitch': 6.0}}, # backward med
]

x_max = 1.5
y_max = 1.0
theta_max = 1.0

x_in, y_in, theta_in = 0.75, 0.3, 0.2

G_output = blend_gait_parameters(sample_data, x_in, y_in, theta_in, x_max, y_max, theta_max)
print("Final blended G:")
for param, value in G_output.items():
    print(f"  {param}: {value:.4f}")
