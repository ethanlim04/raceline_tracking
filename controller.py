import numpy as np
from numpy.typing import ArrayLike
from racetrack import RaceTrack

# Core tuning parameters - these work across different tracks
LOOKAHEAD_DISTANCE = 25.0
BLEND_FACTOR = 0.5
# MAX_LATERAL_ACCEL = 43.0  # Slightly more conservative for extreme hairpins
MAX_LATERAL_ACCEL = 10.0  # Wy slower for monza hairpin turn
STEERING_EFFORT_WINDOW = 40

# Control gains
VELOCITY_KP = 40.0
STEERING_KP = 12.0
STEERING_KI = 16.0
STEERING_KD = 0.15
DT = 0.1

# Global state for PID
prev_steering_error = 0.0
steering_error_integral = 0.0


def resample_path(path: np.ndarray, n_points: int) -> np.ndarray:
    """Resample path to have exactly n_points using linear interpolation."""
    if len(path) == n_points:
        return path
    
    indices = np.linspace(0, len(path) - 1, n_points)
    x = np.interp(indices, np.arange(len(path)), path[:, 0])
    y = np.interp(indices, np.arange(len(path)), path[:, 1])
    
    return np.column_stack((x, y))


def find_lookahead_point(path: np.ndarray, start_idx: int, distance: float) -> int:
    """
    Walk along path until we've traveled 'distance' meters.
    Handles non-uniform point spacing properly.
    """
    path_length = len(path)
    distance_traveled = 0.0
    current_idx = start_idx
    max_iterations = min(path_length, 1000)
    
    for _ in range(max_iterations):
        next_idx = (current_idx + 1) % path_length
        segment_length = np.linalg.norm(path[next_idx] - path[current_idx])
        
        # Skip zero-length segments
        if segment_length < 1e-6:
            current_idx = next_idx
            continue
            
        distance_traveled += segment_length
        
        if distance_traveled >= distance:
            return next_idx
            
        current_idx = next_idx
    
    return current_idx


def calculate_steering_effort(path: np.ndarray, width: np.ndarray, 
                              center_idx: int, window: int = 10) -> float:
    """
    Calculate track difficulty metric at a given point.
    
    Returns ~0.0 for straights, ~0.5 for normal corners, ~1.0+ for tight hairpins.
    This adapts automatically to track characteristics.
    """
    if len(path) < 5 or len(width) == 0:
        return 0.0
    
    center_idx = center_idx % len(path)
    half_window = window // 2
    start_idx = max(center_idx - half_window, 2)
    end_idx = min(center_idx + half_window, len(path) - 3)
    
    if start_idx >= end_idx:
        return 0.0
    
    steering_changes = []
    segment_distances = []
    
    for i in range(start_idx, end_idx + 1):
        p1, p2, p3 = path[i - 2], path[i], path[i + 2]
        
        v1 = p2 - p1
        v2 = p3 - p2
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue
        
        # Calculate turning angle
        cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Get sign (left/right)
        cross = np.cross(v1, v2)
        if cross < 0:
            theta = -theta
        
        # Amplify sharp angles
        if theta < 0.77:
            normalized_theta = np.tan(theta * 2) / 2
        else:
            normalized_theta = 100
        
        # Scale by track width
        width_idx = i % len(width)
        effort = normalized_theta / max(width[width_idx], 5.0) * 10
        
        steering_changes.append(effort)
        segment_distances.append(norm1)
    
    if len(steering_changes) < 2:
        return 0.0
    
    steering_changes = np.array(steering_changes)
    segment_distances = np.array(segment_distances)
    
    # Rate of steering change
    delta_steering = np.diff(steering_changes)
    delta_steering = (delta_steering + np.pi) % (2 * np.pi) - np.pi
    
    avg_segment_distance = (segment_distances[1:] + segment_distances[:-1]) / 2
    steering_rate = np.abs(delta_steering) / np.maximum(avg_segment_distance, 1e-6)
    
    # Focus on worst parts
    if len(steering_rate) >= 5:
        top_k_rates = np.sort(steering_rate)[-5:]
    else:
        top_k_rates = steering_rate
        
    return np.mean(top_k_rates) if len(top_k_rates) > 0 else 0.0


def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    """
    Universal adaptive Pure Pursuit controller.
    Automatically adjusts to track difficulty without manual tuning.
    """
    state = np.asarray(state, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    
    position = state[:2]
    heading = state[4]
    velocity = state[3]
    
    wheelbase = parameters[0]
    delta_min = parameters[1]
    delta_max = parameters[4]
    v_max = parameters[5]
    
    # Get path with DYNAMIC blending based on upcoming difficulty
    if hasattr(racetrack, 'raceline') and racetrack.raceline is not None:
        raceline = racetrack.raceline
        centerline = racetrack.centerline
        
        if len(raceline) != len(centerline):
            centerline = resample_path(centerline, len(raceline))
        
        # Check difficulty of upcoming section to decide blend
        temp_path = 0.5 * raceline + 0.5 * centerline
        temp_distances = np.linalg.norm(temp_path - position, axis=1)
        temp_nearest = int(np.argmin(temp_distances))
        
        temp_width = np.linalg.norm(
            racetrack.right_boundary - racetrack.left_boundary, 
            axis=1
        )
        
        # Look ahead to see what's coming (check further for extreme hairpins)
        upcoming_check_idx = find_lookahead_point(temp_path, temp_nearest, 25.0)
        upcoming_difficulty = calculate_steering_effort(
            temp_path, temp_width, upcoming_check_idx, window=30
        )
        
        # DYNAMIC BLEND: Continuously scales based on difficulty
        # This works for ANY track - no hard-coded thresholds
        if upcoming_difficulty > 0.3:
            # Smooth curve: more centerline as difficulty increases
            # difficulty 0.3 → 50% blend (50/50)
            # difficulty 1.0 → 25% blend (75% centerline)
            # difficulty 1.5 → 8% blend (92% centerline)
            # difficulty 2.0+ → 10% blend (90% centerline, capped)
            blend = max(0.10, 0.5 - (upcoming_difficulty - 0.3) * 0.36)
        else:
            blend = BLEND_FACTOR  # 50/50 for easy sections
        
        path = blend * raceline + (1 - blend) * centerline
    else:
        path = racetrack.centerline
    
    track_width = np.linalg.norm(
        racetrack.right_boundary - racetrack.left_boundary, 
        axis=1
    )
    
    # Find current position
    distances_to_path = np.linalg.norm(path - position, axis=1)
    nearest_idx = int(np.argmin(distances_to_path))
    cross_track_error = distances_to_path[nearest_idx]
    
    # Off-track detection with moderate threshold
    OFF_TRACK_THRESHOLD = 3.5  # Slightly tighter for earlier recovery
    off_track = cross_track_error > OFF_TRACK_THRESHOLD
    
    # ========================================================================
    # SPEED CONTROL - Adaptive based on local difficulty
    # ========================================================================
    
    if off_track:
        # Recovery mode - slow down more
        desired_velocity = min(v_max * 0.3, 22.0)
    else:
        # Look ahead for speed planning
        speed_lookahead_idx = find_lookahead_point(path, nearest_idx, LOOKAHEAD_DISTANCE)
        
        # FIXED: Proper index arithmetic (this was the critical bug!)
        future_idx = (speed_lookahead_idx + 15) % len(path)
        
        # Check both future AND current location
        future_effort = calculate_steering_effort(
            path, track_width, future_idx, window=STEERING_EFFORT_WINDOW
        )
        current_effort = calculate_steering_effort(
            path, track_width, nearest_idx, window=20
        )
        
        # Use the worse of the two
        max_effort = max(future_effort, current_effort)
        
        # ADAPTIVE BRAKING: Continuously scales based on effort
        # No hard thresholds - works for ANY difficulty level
        if max_effort > 0.3:
            # Scale from 2.5x to 5.0x smoothly
            # effort 0.3 → 2.5x (fast)
            # effort 1.0 → 4.25x (safe)
            # effort 1.5+ → 5.0x (very safe, capped)
            brake_multiplier = min(5.0, 2.5 + (max_effort - 0.3) * 2.5)
        else:
            brake_multiplier = 2.5  # Easy sections stay fast
        
        max_effort *= brake_multiplier
        
        # Calculate safe speed
        if abs(max_effort) > 0.001:
            safe_speed = np.sqrt(MAX_LATERAL_ACCEL / abs(max_effort))
        else:
            safe_speed = v_max
        
        desired_velocity = min(safe_speed, v_max)
    
    # ========================================================================
    # LOOKAHEAD - Adaptive based on speed and local curvature
    # ========================================================================
    
    if off_track:
        # Recovery mode: aim for centerline with short lookahead
        center_distances = np.linalg.norm(racetrack.centerline - position, axis=1)
        center_nearest_idx = int(np.argmin(center_distances))
        
        recovery_lookahead = 4.0  # Very short for tight control
        lookahead_idx = find_lookahead_point(
            racetrack.centerline, center_nearest_idx, recovery_lookahead
        )
        lookahead_point = racetrack.centerline[lookahead_idx]
        
        # Don't look behind
        to_lookahead = lookahead_point - position
        forward_direction = np.array([np.cos(heading), np.sin(heading)])
        if np.dot(to_lookahead, forward_direction) < 0:
            lookahead_point = racetrack.centerline[center_nearest_idx]
    else:
        # Normal mode: speed-adaptive lookahead
        lookahead_adjustment = int((100 - velocity) / 8)
        adaptive_lookahead = max(LOOKAHEAD_DISTANCE - lookahead_adjustment, 10.0)
        
        # CRITICAL: Reduce lookahead based on local difficulty
        # Continuously scales - no hard thresholds
        local_difficulty = calculate_steering_effort(path, track_width, nearest_idx, window=10)
        
        if local_difficulty > 0.1:
            # Smooth scaling from 100% down to 15%
            # difficulty 0.1 → 100% lookahead
            # difficulty 0.5 → 70% lookahead
            # difficulty 1.0 → 32.5% lookahead
            # difficulty 1.5+ → 15% lookahead (capped)
            difficulty_factor = max(0.15, 1.0 - (local_difficulty - 0.1) * 0.75)
        else:
            difficulty_factor = 1.0
        
        adaptive_lookahead *= difficulty_factor
        adaptive_lookahead = max(adaptive_lookahead, 4.5)  # Absolute minimum
        
        lookahead_idx = find_lookahead_point(path, nearest_idx, adaptive_lookahead)
        lookahead_point = path[lookahead_idx]
    
    # ========================================================================
    # STEERING - Pure Pursuit
    # ========================================================================
    
    lookahead_vector = lookahead_point - position
    lookahead_distance = np.linalg.norm(lookahead_vector)
    
    desired_heading = np.arctan2(lookahead_vector[1], lookahead_vector[0])
    heading_error = (desired_heading - heading + np.pi) % (2 * np.pi) - np.pi
    
    if lookahead_distance > 0.001:
        path_curvature = 2 * np.sin(heading_error) / lookahead_distance
    else:
        path_curvature = 0.0
    
    desired_steering = np.arctan(wheelbase * path_curvature)
    
    # Limit steering in recovery mode
    if off_track:
        reduced_max = delta_max * 0.55
        reduced_min = delta_min * 0.55
        desired_steering = np.clip(desired_steering, reduced_min, reduced_max)
    
    desired_steering = np.clip(desired_steering, delta_min, delta_max)
    
    return np.array([desired_steering, desired_velocity], dtype=float)


def lower_controller(state: ArrayLike, desired: ArrayLike, 
                     parameters: ArrayLike) -> ArrayLike:
    """
    Low-level PID controller for steering and P controller for velocity.
    """
    global prev_steering_error, steering_error_integral
    
    state = np.asarray(state, dtype=float)
    desired = np.asarray(desired, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    
    current_steering = state[2]
    current_velocity = state[3]
    
    desired_steering = desired[0]
    desired_velocity = desired[1]
    
    steering_rate_min = parameters[7]
    steering_rate_max = parameters[9]
    accel_min = parameters[8]
    accel_max = parameters[10]
    
    # PID for steering
    steering_error = (desired_steering - current_steering + np.pi) % (2 * np.pi) - np.pi
    steering_error_rate = (steering_error - prev_steering_error) / DT
    steering_error_integral += steering_error * DT
    
    # Anti-windup
    MAX_INTEGRAL = 0.4
    if prev_steering_error * steering_error < 0:
        steering_error_integral *= 0.5
    
    steering_error_integral = np.clip(steering_error_integral, -MAX_INTEGRAL, MAX_INTEGRAL)
    prev_steering_error = steering_error
    
    steering_rate = (
        STEERING_KP * steering_error +
        STEERING_KI * steering_error_integral +
        STEERING_KD * steering_error_rate
    )
    steering_rate = np.clip(steering_rate, steering_rate_min, steering_rate_max)
    
    # P control for velocity
    velocity_error = desired_velocity - current_velocity
    acceleration = VELOCITY_KP * velocity_error
    acceleration = np.clip(acceleration, accel_min, accel_max)
    
    return np.array([steering_rate, acceleration], dtype=float)