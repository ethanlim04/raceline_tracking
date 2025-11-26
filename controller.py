import numpy as np
from numpy.typing import ArrayLike
from racetrack import RaceTrack

# ============================================================================
# TUNING PARAMETERS
# ============================================================================

# Path following
LOOKAHEAD_DISTANCE = 25.0      # Base lookahead distance (meters)
BLEND_FACTOR = 0.5             # Raceline vs centerline (0=center, 1=race)

# Speed control
MAX_LATERAL_ACCEL = 45.0       # Maximum cornering acceleration (m/s²)
CURVATURE_LOOKAHEAD = 15       # Distance ahead to check for braking (meters)
STEERING_EFFORT_WINDOW = 40    # Points to analyze for difficulty

# Lower-level control gains
VELOCITY_KP = 40.0             # Velocity P gain
STEERING_KP = 12.0             # Steering P gain
STEERING_KI = 16.0             # Steering I gain
STEERING_KD = 0.15             # Steering D gain

# Control loop timing
DT = 0.1                       # Time step (seconds)

# ============================================================================
# GLOBAL STATE (for integral/derivative terms)
# ============================================================================

prev_steering_error = 0.0
steering_error_integral = 0.0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def resample_path(path: np.ndarray, n_points: int) -> np.ndarray:
    """
    Resample a path to have exactly n_points using linear interpolation.
    Ensures centerline and raceline have matching lengths for blending.
    """
    if len(path) == n_points:
        return path
    
    indices = np.linspace(0, len(path) - 1, n_points)
    x = np.interp(indices, np.arange(len(path)), path[:, 0])
    y = np.interp(indices, np.arange(len(path)), path[:, 1])
    
    return np.column_stack((x, y))


def find_lookahead_point(path: np.ndarray, start_idx: int, distance: float) -> int:
    """
    Walk along the path from start_idx until we've traveled 'distance' meters.
    Returns the index of the point we reach.
    
    This follows the path's actual geometry rather than assuming uniform spacing.
    """
    path_length = len(path)
    distance_traveled = 0.0
    current_idx = start_idx
    max_iterations = min(path_length, 1000)  # Safety limit
    
    for _ in range(max_iterations):
        next_idx = (current_idx + 1) % path_length
        segment_length = np.linalg.norm(path[next_idx] - path[current_idx])
        
        if segment_length == 0:
            break
            
        distance_traveled += segment_length
        
        if distance_traveled >= distance:
            return next_idx
            
        current_idx = next_idx
    
    return current_idx


def calculate_steering_effort(path: np.ndarray, width: np.ndarray, 
                              center_idx: int, window: int = 10) -> float:
    """
    Calculate steering difficulty metric around a point on the path.
    
    This metric captures:
    - How much steering angle changes (rate of direction change)
    - Effect of track width (narrower = more difficult)
    - Variability in steering demands
    
    Returns a normalized metric where higher values = more difficult section.
    """
    half_window = window // 2
    start_idx = max(center_idx - half_window, 1)
    end_idx = min(center_idx + half_window, len(path) - 3)
    
    steering_changes = []
    segment_distances = []
    
    # Analyze steering requirements across the window
    for i in range(start_idx, end_idx + 1):
        # Look at three points to calculate angle change
        p1, p2, p3 = path[i - 2], path[i], path[i + 2]
        
        v1 = p2 - p1
        v2 = p3 - p2
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue
        
        # Calculate angle between segments
        cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Use cross product to get signed angle (left vs right turn)
        cross = np.cross(v1, v2)
        if cross < 0:
            theta = -theta
        
        # Amplify sharp angles, normalize by track width
        # Narrower tracks make the same curvature more difficult
        if theta < 0.77:  # ~44 degrees
            normalized_theta = np.tan(theta * 2) / 2
        else:
            normalized_theta = 100  # Very sharp corner
        
        # Scale by track width (10m is reference width)
        effort = normalized_theta / width[i] * 10
        
        steering_changes.append(effort)
        segment_distances.append(norm1)
    
    if len(steering_changes) < 2:
        return 0.0
    
    steering_changes = np.array(steering_changes)
    segment_distances = np.array(segment_distances)
    
    # Calculate rate of steering change (how quickly angle changes per meter)
    delta_steering = np.diff(steering_changes)
    delta_steering = (delta_steering + np.pi) % (2 * np.pi) - np.pi  # Normalize angles
    
    avg_segment_distance = (segment_distances[1:] + segment_distances[:-1]) / 2
    steering_rate = np.abs(delta_steering) / np.maximum(avg_segment_distance, 1e-6)
    
    # Take average of top-5 largest steering rates
    # This focuses on the most demanding parts of the section
    top_k_rates = np.sort(steering_rate)[-5:]
    avg_effort = np.mean(top_k_rates)
    
    return avg_effort


# ============================================================================
# MAIN CONTROLLER
# ============================================================================

def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    """
    High-level controller: determines desired steering angle and velocity.
    
    Uses Pure Pursuit algorithm with:
    - Dynamic lookahead based on current speed
    - Steering effort prediction for speed planning
    - Raceline/centerline blending for safety
    
    Args:
        state: [x, y, steering_angle, velocity, heading, ...]
        parameters: [wheelbase, delta_min, v_min, ..., delta_max, v_max, ...]
        racetrack: Track object with centerline, raceline, boundaries
    
    Returns:
        [desired_steering_angle, desired_velocity]
    """
    # Parse inputs
    state = np.asarray(state, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    
    position = state[:2]
    heading = state[4]
    velocity = state[3]
    
    wheelbase = parameters[0]
    delta_min = parameters[1]  # Minimum steering angle
    delta_max = parameters[4]  # Maximum steering angle
    v_max = parameters[5]      # Maximum velocity
    
    # ========================================================================
    # PATH SELECTION AND PREPARATION
    # ========================================================================
    
    # Blend raceline with centerline for safety margin
    if hasattr(racetrack, 'raceline') and racetrack.raceline is not None:
        raceline = racetrack.raceline
        centerline = racetrack.centerline
        
        # Ensure same length for blending
        if len(raceline) != len(centerline):
            centerline = resample_path(centerline, len(raceline))
        
        # Blend: 0.5 = halfway between aggressive raceline and safe centerline
        path = BLEND_FACTOR * raceline + (1 - BLEND_FACTOR) * centerline
    else:
        path = racetrack.centerline
    
    # Get track width at each point (for steering effort calculation)
    track_width = np.linalg.norm(
        racetrack.right_boundary - racetrack.left_boundary, 
        axis=1
    )
    
    # ========================================================================
    # FIND CURRENT POSITION ON PATH
    # ========================================================================
    
    distances_to_path = np.linalg.norm(path - position, axis=1)
    nearest_idx = int(np.argmin(distances_to_path))
    cross_track_error = distances_to_path[nearest_idx]
    
    # Detect off-track situation (more than 5 meters from path)
    OFF_TRACK_THRESHOLD = 5.0
    off_track = cross_track_error > OFF_TRACK_THRESHOLD
    
    # ========================================================================
    # SPEED PLANNING: Look ahead and brake for difficult sections
    # ========================================================================
    
    if off_track:
        # RECOVERY MODE: Slow down significantly when off-track
        # This prevents spinning and allows controlled return to path
        desired_velocity = min(v_max * 0.4, 30.0)  # Max 40% speed or 30 m/s
    else:
        # NORMAL MODE: Use steering effort prediction
        # Find point ahead where we need to check curvature for braking
        speed_lookahead_idx = find_lookahead_point(
            path, nearest_idx, LOOKAHEAD_DISTANCE
        )
        
        # Calculate steering effort at future point
        # Multiply by 2 for more conservative braking
        future_steering_effort = 2 * calculate_steering_effort(
            path, 
            track_width, 
            speed_lookahead_idx + CURVATURE_LOOKAHEAD,
            window=STEERING_EFFORT_WINDOW
        )
        
        # Calculate safe speed based on curvature
        # v² = a_lateral / curvature (basic vehicle dynamics)
        if abs(future_steering_effort) > 0.001:
            safe_speed = np.sqrt(MAX_LATERAL_ACCEL / abs(future_steering_effort))
        else:
            safe_speed = v_max
        
        desired_velocity = min(safe_speed, v_max)
    
    # ========================================================================
    # ADAPTIVE LOOKAHEAD: Adjust based on current speed
    # ========================================================================
    
    if off_track:
        # RECOVERY MODE: Use very short, fixed lookahead
        # Aim for nearest point on CENTERLINE (safest path)
        center_distances = np.linalg.norm(racetrack.centerline - position, axis=1)
        center_nearest_idx = int(np.argmin(center_distances))
        
        # Short lookahead for tight control during recovery
        recovery_lookahead = 8.0  # Very short for precise correction
        lookahead_idx = find_lookahead_point(
            racetrack.centerline, 
            center_nearest_idx, 
            recovery_lookahead
        )
        lookahead_point = racetrack.centerline[lookahead_idx]
        
        # Check if lookahead point is behind us (causes spinning)
        to_lookahead = lookahead_point - position
        forward_direction = np.array([np.cos(heading), np.sin(heading)])
        
        # If lookahead is behind (dot product < 0), aim for nearest point instead
        if np.dot(to_lookahead, forward_direction) < 0:
            lookahead_point = racetrack.centerline[center_nearest_idx]
    else:
        # NORMAL MODE: Speed-adaptive lookahead
        # Reduce lookahead at lower speeds for tighter control
        # (100 - speed) / 8 creates a penalty that decreases with speed
        lookahead_adjustment = int((100 - velocity) / 8)
        adaptive_lookahead = max(
            LOOKAHEAD_DISTANCE - lookahead_adjustment,
            10.0  # Minimum lookahead
        )
        
        lookahead_idx = find_lookahead_point(path, nearest_idx, adaptive_lookahead)
        lookahead_point = path[lookahead_idx]
    
    # ========================================================================
    # PURE PURSUIT STEERING CALCULATION
    # ========================================================================
    
    # Vector from current position to lookahead point
    lookahead_vector = lookahead_point - position
    lookahead_distance = np.linalg.norm(lookahead_vector)
    
    # Desired heading to reach lookahead point
    desired_heading = np.arctan2(lookahead_vector[1], lookahead_vector[0])
    
    # Heading error (wrapped to [-π, π])
    heading_error = (desired_heading - heading + np.pi) % (2 * np.pi) - np.pi
    
    # Pure Pursuit curvature formula:
    # κ = 2 * sin(α) / L_d
    # where α is heading error, L_d is lookahead distance
    if lookahead_distance > 0.001:
        path_curvature = 2 * np.sin(heading_error) / lookahead_distance
    else:
        path_curvature = 0.0
    
    # Convert curvature to steering angle using bicycle model
    # δ = arctan(wheelbase * κ)
    desired_steering = np.arctan(wheelbase * path_curvature)
    
    # Apply steering limits
    if off_track:
        # Extra limit when off-track to prevent spinning
        # Reduce maximum steering to 60% of normal range
        reduced_max = delta_max * 0.6
        reduced_min = delta_min * 0.6
        desired_steering = np.clip(desired_steering, reduced_min, reduced_max)
    
    desired_steering = np.clip(desired_steering, delta_min, delta_max)
    
    return np.array([desired_steering, desired_velocity], dtype=float)


# ============================================================================
# LOWER-LEVEL CONTROLLER
# ============================================================================

def lower_controller(state: ArrayLike, desired: ArrayLike, 
                     parameters: ArrayLike) -> ArrayLike:
    """
    Low-level controller: converts desired steering/velocity to control rates.
    
    Uses:
    - PID control for steering (eliminates steady-state error)
    - P control for velocity (simple and effective)
    
    Args:
        state: Current vehicle state
        desired: [desired_steering_angle, desired_velocity] from controller()
        parameters: Vehicle parameters including control limits
    
    Returns:
        [steering_rate, acceleration]
    """
    global prev_steering_error, steering_error_integral
    
    state = np.asarray(state, dtype=float)
    desired = np.asarray(desired, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    
    # Current state
    current_steering = state[2]
    current_velocity = state[3]
    
    # Desired state
    desired_steering = desired[0]
    desired_velocity = desired[1]
    
    # Control limits
    steering_rate_min = parameters[7]
    steering_rate_max = parameters[9]
    accel_min = parameters[8]
    accel_max = parameters[10]
    
    # ========================================================================
    # PID CONTROL FOR STEERING
    # ========================================================================
    
    # Calculate error (wrapped to [-π, π])
    steering_error = (desired_steering - current_steering + np.pi) % (2 * np.pi) - np.pi
    
    # Derivative term (rate of change of error)
    steering_error_rate = (steering_error - prev_steering_error) / DT
    
    # Integral term (accumulated error over time)
    steering_error_integral += steering_error * DT
    
    # Anti-windup: prevent integral term from growing too large
    # This prevents overshooting when limits are hit
    MAX_INTEGRAL = 0.5
    steering_error_integral = np.clip(
        steering_error_integral, 
        -MAX_INTEGRAL, 
        MAX_INTEGRAL
    )
    
    # Update previous error for next iteration
    prev_steering_error = steering_error
    
    # PID formula: u = Kp*e + Ki*∫e + Kd*de/dt
    steering_rate = (
        STEERING_KP * steering_error +
        STEERING_KI * steering_error_integral +
        STEERING_KD * steering_error_rate
    )
    
    # Apply rate limits
    steering_rate = np.clip(steering_rate, steering_rate_min, steering_rate_max)
    
    # ========================================================================
    # P CONTROL FOR VELOCITY
    # ========================================================================
    
    velocity_error = desired_velocity - current_velocity
    acceleration = VELOCITY_KP * velocity_error
    
    # Apply acceleration limits
    acceleration = np.clip(acceleration, accel_min, accel_max)
    
    return np.array([steering_rate, acceleration], dtype=float)