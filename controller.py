import numpy as np
from numpy.typing import ArrayLike
from racetrack import RaceTrack

# Base control gains (will be adapted)
KP_VELOCITY = 20.0
KP_STEERING = 15.0
KD_STEERING = 0.3

# Base lookahead parameters (will be adapted)
BASE_LOOKAHEAD = 30.0
MIN_LOOKAHEAD = 12.0

# Base speed tuning (will be adapted)
MAX_LATERAL_ACCEL = 48.0

# Global variables
prev_steering_error = 0.0
dt = 0.1

# Cache for track characteristics (computed once per track)
track_characteristics = {
    'analyzed': False,
    'avg_width': 0.0,
    'min_width': 0.0,
    'max_width': 0.0,
    'avg_curvature': 0.0,
    'max_curvature': 0.0,
    'track_type': 'unknown',  # 'street', 'road', 'oval'
    'blend_factor': 0.80,
    'safety_margin': 2.5,
    'lateral_accel': 48.0
}


def analyze_track(racetrack: RaceTrack):
    """
    Analyze track characteristics once and cache the results.
    Determines optimal parameters based on track geometry.
    """
    global track_characteristics
    
    if track_characteristics['analyzed']:
        return  # Already analyzed
    
    # Calculate track widths
    widths = []
    for i in range(len(racetrack.left_boundary)):
        width = np.linalg.norm(racetrack.left_boundary[i] - racetrack.right_boundary[i])
        widths.append(width)
    
    widths = np.array(widths)
    avg_width = np.mean(widths)
    min_width = np.min(widths)
    max_width = np.max(widths)
    width_variation = np.std(widths)
    
    # Calculate curvatures along centerline
    curvatures = []
    path = racetrack.centerline
    num_points = len(path)
    
    for i in range(num_points):
        idx_before = (i - 5) % num_points
        idx_after = (i + 5) % num_points
        
        p1 = path[idx_before]
        p2 = path[i]
        p3 = path[idx_after]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 > 0.01 and len2 > 0.01:
            cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            arc_length = (len1 + len2) / 2.0
            curvature = angle / max(arc_length, 0.1)
            curvatures.append(curvature)
    
    curvatures = np.array(curvatures)
    avg_curvature = np.mean(curvatures)
    max_curvature = np.percentile(curvatures, 95)  # 95th percentile to avoid outliers
    
    # Classify track type based on characteristics
    # Street circuit: narrow, high curvature variation, sharp corners
    # Road course: medium width, moderate curvature
    # Oval: wide, low curvature, consistent width
    
    if avg_width < 11.0 and max_curvature > 0.15:
        # Narrow with sharp corners → Street circuit (like Montreal)
        track_type = 'street'
        blend_factor = 0.75  # Conservative - more centerline
        safety_margin = 2.2  # Tight safety margin
        lateral_accel = 45.0  # More conservative cornering
        
    elif avg_width > 14.0 and avg_curvature < 0.05:
        # Wide with gentle curves → Oval (like IMS)
        track_type = 'oval'
        blend_factor = 0.90  # Aggressive - mostly raceline
        safety_margin = 3.0  # Wider safety margin (track is wide)
        lateral_accel = 52.0  # Aggressive cornering
        
    else:
        # Medium characteristics → Road course (like Monza)
        track_type = 'road'
        blend_factor = 0.82  # Balanced
        safety_margin = 2.6  # Balanced
        lateral_accel = 48.0  # Balanced
    
    # Fine-tune based on width variation
    # High variation = more technical track = more conservative
    if width_variation > 2.0:
        blend_factor *= 0.95  # Slightly more conservative
        lateral_accel *= 0.95
    
    # Store results
    track_characteristics['analyzed'] = True
    track_characteristics['avg_width'] = avg_width
    track_characteristics['min_width'] = min_width
    track_characteristics['max_width'] = max_width
    track_characteristics['avg_curvature'] = avg_curvature
    track_characteristics['max_curvature'] = max_curvature
    track_characteristics['track_type'] = track_type
    track_characteristics['blend_factor'] = blend_factor
    track_characteristics['safety_margin'] = safety_margin
    track_characteristics['lateral_accel'] = lateral_accel
    
    # Debug output (optional - comment out in production)
    print(f"\n=== TRACK ANALYSIS ===")
    print(f"Track Type: {track_type.upper()}")
    print(f"Average Width: {avg_width:.1f}m")
    print(f"Width Range: {min_width:.1f}m - {max_width:.1f}m")
    print(f"Average Curvature: {avg_curvature:.4f}")
    print(f"Max Curvature: {max_curvature:.4f}")
    print(f"Optimal Blend: {blend_factor:.2f}")
    print(f"Safety Margin: {safety_margin:.1f}m")
    print(f"Max Lateral Accel: {lateral_accel:.1f} m/s²")
    print(f"======================\n")


def find_lookahead_point(path: np.ndarray, start_idx: int, target_distance: float) -> int:
    """Walk along path until we've covered target_distance meters."""
    num_points = len(path)
    accumulated_dist = 0.0
    current_idx = start_idx
    max_iterations = min(num_points, 500)
    
    for _ in range(max_iterations):
        next_idx = (current_idx + 1) % num_points
        segment_vec = path[next_idx] - path[current_idx]
        segment_length = np.linalg.norm(segment_vec)
        accumulated_dist += segment_length
        
        if accumulated_dist >= target_distance:
            return next_idx
        current_idx = next_idx
    
    return current_idx


def calculate_path_curvature(path: np.ndarray, idx: int, window: int = 5) -> float:
    """Calculate curvature at a point by looking at nearby points."""
    num_points = len(path)
    
    idx_before = (idx - window) % num_points
    idx_after = (idx + window) % num_points
    
    p1 = path[idx_before]
    p2 = path[idx]
    p3 = path[idx_after]
    
    v1 = p2 - p1
    v2 = p3 - p2
    
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    
    if len1 < 0.01 or len2 < 0.01:
        return 0.0
    
    cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    arc_length = (len1 + len2) / 2.0
    curvature = angle / max(arc_length, 0.1)
    
    return curvature


def blend_raceline_with_centerline(raceline: np.ndarray, centerline: np.ndarray, 
                                    blend_factor: float) -> np.ndarray:
    """Blend raceline with centerline for safer path."""
    if len(raceline) != len(centerline):
        indices = np.linspace(0, len(centerline) - 1, len(raceline))
        center_x = np.interp(indices, np.arange(len(centerline)), centerline[:, 0])
        center_y = np.interp(indices, np.arange(len(centerline)), centerline[:, 1])
        centerline_resampled = np.column_stack((center_x, center_y))
    else:
        centerline_resampled = centerline
    
    blended = blend_factor * raceline + (1 - blend_factor) * centerline_resampled
    return blended


def get_local_track_width(nearest_idx: int, left_boundary: np.ndarray, 
                          right_boundary: np.ndarray) -> float:
    """Get track width at current position."""
    left_pt = left_boundary[nearest_idx]
    right_pt = right_boundary[nearest_idx]
    return np.linalg.norm(left_pt - right_pt)


def get_boundary_distance(position: np.ndarray, nearest_idx: int,
                          left_boundary: np.ndarray, 
                          right_boundary: np.ndarray) -> float:
    """Get distance to nearest boundary in meters."""
    left_pt = left_boundary[nearest_idx]
    right_pt = right_boundary[nearest_idx]
    
    dist_to_left = np.linalg.norm(position - left_pt)
    dist_to_right = np.linalg.norm(position - right_pt)
    
    return min(dist_to_left, dist_to_right)


def get_normalized_margin(position: np.ndarray, nearest_idx: int, 
                          left_boundary: np.ndarray, right_boundary: np.ndarray) -> float:
    """Get normalized margin: 0 (at boundary) to 1 (at center)."""
    left_pt = left_boundary[nearest_idx]
    right_pt = right_boundary[nearest_idx]
    
    dist_to_left = np.linalg.norm(position - left_pt)
    dist_to_right = np.linalg.norm(position - right_pt)
    track_width = np.linalg.norm(left_pt - right_pt)
    
    min_boundary_dist = min(dist_to_left, dist_to_right)
    normalized = min_boundary_dist / (track_width / 2.0) if track_width > 0 else 1.0
    
    return np.clip(normalized, 0.0, 1.0)


def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    """
    Universal adaptive Pure Pursuit controller.
    Automatically analyzes track and adjusts parameters for optimal performance.
    Works on street circuits, road courses, and ovals.
    """
    # Analyze track on first call
    analyze_track(racetrack)
    
    # Get track-specific parameters
    blend_factor = track_characteristics['blend_factor']
    safety_margin_base = track_characteristics['safety_margin']
    lateral_accel = track_characteristics['lateral_accel']
    track_type = track_characteristics['track_type']
    
    # Parse state
    x_pos = state[0]
    y_pos = state[1]
    velocity = state[3]
    heading = state[4]
    
    # Parse parameters
    wheelbase = parameters[0]
    delta_min = parameters[1]
    v_min = parameters[2]
    delta_max = parameters[4]
    v_max = parameters[5]
    
    position = np.array([x_pos, y_pos])
    
    # Get path with track-adaptive blending
    if hasattr(racetrack, 'raceline') and racetrack.raceline is not None:
        path = blend_raceline_with_centerline(racetrack.raceline, 
                                              racetrack.centerline, 
                                              blend_factor)
    else:
        path = racetrack.centerline
    
    # Find nearest point
    distances_to_path = np.linalg.norm(path - position, axis=1)
    nearest_idx = np.argmin(distances_to_path)
    nearest_dist = distances_to_path[nearest_idx]
    
    # Local track characteristics
    local_width = get_local_track_width(nearest_idx, 
                                        racetrack.left_boundary, 
                                        racetrack.right_boundary)
    boundary_dist = get_boundary_distance(position, nearest_idx,
                                         racetrack.left_boundary,
                                         racetrack.right_boundary)
    track_margin = get_normalized_margin(position, nearest_idx,
                                        racetrack.left_boundary,
                                        racetrack.right_boundary)
    
    # Adaptive safety distance based on local width
    # Narrower sections get tighter margins
    adaptive_safety = min(safety_margin_base, local_width * 0.22)
    
    # Emergency mode if too close to boundary
    emergency_mode = boundary_dist < adaptive_safety
    
    if emergency_mode:
        # EMERGENCY: Aim for centerline
        center_distances = np.linalg.norm(racetrack.centerline - position, axis=1)
        center_nearest_idx = np.argmin(center_distances)
        
        # Very short lookahead for aggressive correction
        emergency_lookahead = MIN_LOOKAHEAD * 0.6
        lookahead_idx = find_lookahead_point(racetrack.centerline, 
                                            center_nearest_idx, 
                                            emergency_lookahead)
        lookahead_point = racetrack.centerline[lookahead_idx]
        
        use_path = racetrack.centerline
        use_idx = center_nearest_idx
        
    else:
        # NORMAL MODE: Follow racing line
        use_path = path
        use_idx = nearest_idx
        
        # Adaptive lookahead based on speed and track type
        if nearest_dist > 5.0:
            # Off path - minimum lookahead for correction
            lookahead_dist = MIN_LOOKAHEAD
        else:
            # Base lookahead on speed
            if track_type == 'oval':
                # Ovals: longer lookahead for high-speed smoothness
                speed_factor = max(0.8, min(2.5, velocity / 30.0))
                base_la = BASE_LOOKAHEAD * 1.2
            elif track_type == 'street':
                # Street circuits: shorter lookahead for tight corners
                speed_factor = max(0.5, min(1.8, velocity / 25.0))
                base_la = BASE_LOOKAHEAD * 0.9
            else:
                # Road courses: balanced
                speed_factor = max(0.6, min(2.0, velocity / 25.0))
                base_la = BASE_LOOKAHEAD
            
            lookahead_dist = max(MIN_LOOKAHEAD, base_la * speed_factor)
            
            # Reduce if approaching boundary
            if track_margin < 0.5:
                boundary_factor = 0.7 + 0.3 * (track_margin / 0.5)
                lookahead_dist *= boundary_factor
        
        lookahead_idx = find_lookahead_point(use_path, use_idx, lookahead_dist)
        lookahead_point = use_path[lookahead_idx]
    
    # Calculate steering using Pure Pursuit
    dx = lookahead_point[0] - x_pos
    dy = lookahead_point[1] - y_pos
    actual_lookahead = np.sqrt(dx**2 + dy**2)
    
    angle_to_target = np.arctan2(dy, dx)
    heading_error = angle_to_target - heading
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    
    # Pure Pursuit curvature formula
    if actual_lookahead > 0.1:
        path_curvature = 2.0 * np.sin(heading_error) / actual_lookahead
    else:
        path_curvature = 0.0
    
    # Convert to steering angle
    desired_steering = np.arctan(wheelbase * path_curvature)
    desired_steering = np.clip(desired_steering, delta_min, delta_max)
    
    # Speed control - look ahead for upcoming curvature
    if emergency_mode:
        speed_lookahead_dist = MIN_LOOKAHEAD * 0.8
        speed_lookahead_idx = find_lookahead_point(racetrack.centerline,
                                                   center_nearest_idx,
                                                   speed_lookahead_dist)
        future_curvature = calculate_path_curvature(racetrack.centerline,
                                                    speed_lookahead_idx,
                                                    window=5)
    else:
        # Adaptive speed lookahead based on track type
        if track_type == 'oval':
            speed_lookahead_dist = min(lookahead_dist + 20.0, 50.0)
        elif track_type == 'street':
            speed_lookahead_dist = min(lookahead_dist + 8.0, 35.0)
        else:
            speed_lookahead_dist = min(lookahead_dist + 12.0, 42.0)
        
        speed_lookahead_idx = find_lookahead_point(use_path, use_idx, speed_lookahead_dist)
        future_curvature = calculate_path_curvature(use_path, speed_lookahead_idx, window=6)
    
    max_curvature = max(abs(path_curvature), future_curvature)
    
    # Physics-based speed using track-adaptive lateral acceleration
    if max_curvature > 0.002:
        physics_based_speed = np.sqrt(lateral_accel / max_curvature)
    else:
        physics_based_speed = v_max
    
    # Multi-layer speed reduction for safety
    
    # Layer 1: Emergency boundary proximity
    if emergency_mode:
        emergency_factor = max(0.55, boundary_dist / adaptive_safety)
        physics_based_speed *= emergency_factor
    
    # Layer 2: Warning zone (approaching boundary)
    elif track_margin < 0.5:
        warning_factor = 0.78 + 0.22 * (track_margin / 0.5)
        physics_based_speed *= warning_factor
    
    # Layer 3: Off racing line penalty
    if nearest_dist > 3.5:
        distance_penalty = max(0.75, 1.0 - (nearest_dist - 3.5) / 12.0)
        physics_based_speed *= distance_penalty
    
    # Apply speed limits
    desired_velocity = min(physics_based_speed, v_max)
    desired_velocity = max(desired_velocity, v_min * 2.0)
    
    return np.array([desired_steering, desired_velocity], dtype=float)


def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    """
    Lower-level PD controller for steering and P controller for velocity.
    Converts high-level commands to actual control inputs.
    """
    global prev_steering_error
    
    current_steering = state[2]
    current_velocity = state[3]
    
    desired_steering = desired[0]
    desired_velocity = desired[1]
    
    v_delta_min = parameters[7]
    a_min = parameters[8]
    v_delta_max = parameters[9]
    a_max = parameters[10]
    
    # PD control for steering
    steering_error = desired_steering - current_steering
    steering_error = np.arctan2(np.sin(steering_error), np.cos(steering_error))
    
    steering_error_rate = (steering_error - prev_steering_error) / dt
    prev_steering_error = steering_error
    
    steering_velocity = KP_STEERING * steering_error + KD_STEERING * steering_error_rate
    steering_velocity = np.clip(steering_velocity, v_delta_min, v_delta_max)
    
    # P control for velocity
    velocity_error = desired_velocity - current_velocity
    acceleration = KP_VELOCITY * velocity_error
    acceleration = np.clip(acceleration, a_min, a_max)
    
    return np.array([steering_velocity, acceleration], dtype=float)