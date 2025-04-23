import numpy as np
from swing import Swing
import json
import os

def save_keypoints_to_json(keypoints, json_path):
    """
    Save keypoints to a JSON file.

    Parameters:
    - keypoints (list): List of keypoints with (x, y, z) coordinates or None for missing landmarks.
    - json_path (str): Path to save the JSON file.
    """
    with open(json_path, 'w') as f:
        json.dump(keypoints, f)
    print(f"Keypoints saved to {json_path}")
    

def save_detected_keypoints_to_json(keypoints, json_path):
    """
    Save only detected keypoints (non-None) to JSON file in readable format.
    Skips missing landmarks and frames with all missing data.
    """
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_heel", "right_heel", "left_foot_index", "right_foot_index"
    ]

    readable = []
    for i, frame in enumerate(keypoints):
        landmarks_dict = {}
        for idx, point in enumerate(frame):
            if point is None or len(point) != 3 or None in point:
                continue  # skip undetected landmarks
            name = landmark_names[idx]
            x, y, z = point
            landmarks_dict[name] = {"x": x, "y": y, "z": z}
        
        # Only save frames that have at least one detected landmark
        if landmarks_dict:
            readable.append({
                "frame": i,
                "landmarks": landmarks_dict
            })

    with open(json_path, 'w') as f:
        json.dump(readable, f, indent=2)

    print(f"Detected keypoints saved to {json_path}")
    
def load_keypoints_from_json(json_path):
    """
    Load readable-format keypoints from JSON and return them as a list of (x, y, z) tuples per frame.
    Missing frames are filled with placeholder values to preserve correct frame timing.

    Returns:
    - List of frames, each containing a list of 33 (x, y, z) tuples.
    """
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_heel", "right_heel", "left_foot_index", "right_foot_index"
    ]

    name_to_index = {name: i for i, name in enumerate(landmark_names)}
    keypoints_array = []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Create a mapping of frame index to landmark data
        frame_map = {frame["frame"]: frame["landmarks"] for frame in data}
        total_frames = max(frame_map.keys()) + 1

        for i in range(total_frames):
            landmarks = [(None, None, None)] * 33
            if i in frame_map:
                for name, coords in frame_map[i].items():
                    idx = name_to_index[name]
                    landmarks[idx] = (coords["x"], coords["y"], coords["z"])
            keypoints_array.append(landmarks)

        print(f"Successfully loaded {len(keypoints_array)} frames of keypoints from {json_path}")
        return keypoints_array

    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None
    

def save_swings_to_json(swings, filename):
    """Save a list of Swing objects to a JSON file."""
    with open(filename, "w") as f:
        json.dump([swing.__dict__ for swing in swings], f, indent=4)
 
def load_swings_from_json(filename):
    """Load Swing objects from a JSON file if available; otherwise, return an empty list."""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Returning an empty list.")
        return []  # Return an empty list instead of failing
    
    with open(filename, "r") as f:
        swing_dicts = json.load(f)
    return [Swing(**swing_dict) for swing_dict in swing_dicts]
       
def calculate_angle(shoulder, elbow, wrist):
    """
    Calculate the counterclockwise angle of the elbow bend formed by the shoulder, elbow, and wrist in 2D.

    Parameters:
    - shoulder: (x, y) coordinates of the shoulder.
    - elbow: (x, y) coordinates of the elbow.
    - wrist: (x, y) coordinates of the wrist.

    Returns:
    - The angle in degrees (0° to 180°, always counterclockwise).
    """
    # Compute vectors
    v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])  # Shoulder to elbow
    v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])        # Wrist to elbow

    # Dot product and magnitudes
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0  # Avoid division by zero

    # Angle calculation using dot product
    angle_radians = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def calculate_contact_point(orientation):
    if orientation['rw_y'] < orientation['rs_y']:
        return "high"
    if orientation['rw_y'] > orientation['rh_y']:
        return "low"
    
    return 'middle'


def calculate_forehand_stance(orientation):
    body = orientation['shoulder_hip_distance']
    x_distance = (orientation['ra_x'] - orientation['la_x']) / orientation['shoulder_hip_distance']
    z_distance = (orientation['la_z'] - orientation['ra_z']) / orientation['shoulder_hip_distance']
    
    #print(f"x: {x_distance:.2f}, z: {z_distance:.2f} {body}")
    if x_distance < 0.35:
        return "closed"
    if z_distance < 1.1:
        return "open"
    return "semi-open"
    
def calculate_backhand_stance(orientation):
    body = orientation['shoulder_hip_distance']
    x_distance = (orientation['ra_x'] - orientation['la_x']) / orientation['shoulder_hip_distance']
    z_distance = (orientation['ra_z'] - orientation['la_z']) / orientation['shoulder_hip_distance']
    
    #print(f"x: {x_distance:.2f}, z: {z_distance:.2f} {body}")
    if x_distance < 0.35:
        return "closed"
    if z_distance < 1.1:
        return "open"
    return "semi-open"
    
# Function to calculate body orientation and distances for each frame
# called by find_most_parallel_and_furthest_segment
def calculate_body_orientation(keypoints):
    orientations = []
    for keypoint in keypoints:
        if keypoint[15][0] is not None and keypoint[16][0] is not None and keypoint[11][0] is not None and keypoint[12][0] is not None and keypoint[24][0] is not None and keypoint[23][0] is not None:
            # Validate if the body detection is plausible
            #if is_body_detection_invalid(keypoint):
             #   orientations.append(None)
              #  continue
            try:
                # Key body points (x, y, z coordinates are now included)
                nose_x, nose_y, nose_z = keypoint[0][:3]
                rs_x, rs_y, rs_z = keypoint[12][:3]  # Right shoulder
                ls_x, ls_y, ls_z = keypoint[11][:3]  # Left shoulder
                rh_x, rh_y, rh_z = keypoint[24][:3]  # Right hip
                lh_x, lh_y, lh_z = keypoint[23][:3]  # Left hip
                lw_x, lw_y, lw_z = keypoint[15][:3]  # Left wrist
                rw_x, rw_y, rw_z = keypoint[16][:3]  # Right wrist
                le_x, le_y, le_z = keypoint[13][:3]  # Left elbow
                re_x, re_y, re_z = keypoint[14][:3]  # Right elbow
                lk_x, lk_y, lk_z = keypoint[25][:3]  # Left knee
                rk_x, rk_y, rk_z = keypoint[26][:3]  # Right knee
                la_x, la_y, la_z = keypoint[27][:3]  # Left ankle
                ra_x, ra_y, ra_z = keypoint[28][:3]  # Right ankle
                # Calculate shoulder and hip rotation angles
                shoulder_rotation = np.arctan2(ls_x - rs_x, ls_y - rs_y)  # Angle in radians
                hip_rotation = np.arctan2(lh_x - rh_x, lh_y - rh_y)       # Angle in radians

                # Convert to degrees for easier interpretation
                shoulder_rotation_deg = np.degrees(shoulder_rotation)
                hip_rotation_deg = np.degrees(hip_rotation)

                # Normalize angles: forward-facing body as 0 degrees
                shoulder_rotation_deg_normalized = shoulder_rotation_deg + 90
                hip_rotation_deg_normalized = hip_rotation_deg + 90

                # Average shoulder and hip rotation for body rotation
                body_rotation = (shoulder_rotation_deg_normalized + hip_rotation_deg_normalized) / 2

                # Calculate alignment score (based on alignment of shoulders and hips along the y-axis)
                alignment_score = abs(rs_y - ls_y) + abs(rh_y - lh_y)

                # Calculate left hand and right wrist distances
                left_hand_position = lw_x - rs_x

                # Calculate the angle between left shoulder, left elbow, and left wrist (to check for elbow bend)
                elbow_angle_left = calculate_angle([ls_x, ls_y, ls_z], [le_x, le_y, le_z], [lw_x, lw_y, lw_z])

                # Calculate the angle between right shoulder, right elbow, and right wrist (to check for elbow bend)
                elbow_angle_right = calculate_angle([rs_x, rs_y], [re_x, re_y], [rw_x, rw_y])

                # Calculate left hand and right wrist distances
                shoulder_hip_distance = abs(rs_y - rh_y)
                wrist_distance = abs(lw_y - rw_y) + abs(lw_x - rw_x)
                wrist_distance = wrist_distance / shoulder_hip_distance
                both_hands_left_of_body = lw_x < ls_x and rw_x < ls_x
                both_hands_right_of_body = lw_x > rs_x and rw_x > rs_x 

                orientations.append({
                    "body_rotation": np.degrees(body_rotation),  # Rotation around the y-axis in degrees
                    "alignment_score": alignment_score,
                    "left_hand_position": left_hand_position,
                    "rw_x": rw_x,  # x-coordinate of the right wrist
                    "rs_x": rs_x,
                    "rw_y": rw_y,
                    "lw_y": lw_y,
                    "rh_x": rh_x,
                    "re_x": re_x,
                    "lw_x": lw_x,
                    "rs_x": rs_x,
                    "ls_x": ls_x,
                    "rs_y": rs_y,
                    "ls_y": ls_y,
                    "la_x": la_x,
                    "ra_x": ra_x,
                    "la_y": la_y,
                    "ra_y": ra_y,
                    "le_x": le_x,
                    "rs_z": rs_z,
                    "ls_z": ls_z,
                    "lw_z": lw_z,
                    "la_z": la_z,
                    "ra_z": ra_z,
                    "rw_z": rw_z,
                    "le_x": le_x,
                    "lh_x": lh_x,
                    "rh_y": rh_y,
                    "re_z": re_z,
                    "rh_z": rh_z,
                    "nose_x": nose_x,
                    "nose_z": nose_z,
                    "left_wrist": (lw_x, lw_y, lw_z),
                    "right_wrist": (rw_x, rw_y, rw_z),
                    "shoulder_hip_distance": shoulder_hip_distance,
                    "both_hands_left_of_body": both_hands_left_of_body,
                    "both_hands_right_of_body": both_hands_right_of_body,
                    "wrist_distance": wrist_distance,
                    "elbow_angle_left": elbow_angle_left,  # Angle of the left arm at the elbow
                    "elbow_angle_right": elbow_angle_right  # Angle of the right arm at the elbow
                })
            except Exception as e:
                orientations.append(None)
        else:
            orientations.append(None)
    return orientations

def calculate_peak_wrist_velocity_at_contact(right_wrist_coords, fps, shoulder_to_hip_fraction, player_height_m):
    """
    Compute the peak wrist velocity over any 0.5-second window, output in meters per second (m/s).
    Uses real player height and observed shoulder-hip segment to scale unitless coords into meters.
    """
    if shoulder_to_hip_fraction <= 0 or player_height_m <= 0:
        raise ValueError("Both shoulder-to-hip fraction and player height must be positive.")

    # Based on anthropometric estimates: shoulder-to-hip is ~27% of body height
    expected_shoulder_hip_proportion = 0.27
    real_shoulder_to_hip_m = expected_shoulder_hip_proportion * player_height_m

    # This is the scale multiplier that maps unitless coordinates to meters
    scale = real_shoulder_to_hip_m / shoulder_to_hip_fraction

    window_size = int(fps * 0.19)
    if len(right_wrist_coords) < window_size:
        return 0.0, None

    peak_velocity_mps = 0.0
    peak_start_index = None
    frame_interval = window_size / fps  # duration in seconds

    for i in range(len(right_wrist_coords) - window_size):
        start = right_wrist_coords[i]
        end = right_wrist_coords[i + window_size]
        if start is None or end is None:
            continue

        displacement_unitless = np.linalg.norm(np.array(end) - np.array(start))
        displacement_meters = displacement_unitless * scale
        velocity_mps = displacement_meters / frame_interval

        if velocity_mps > peak_velocity_mps:
            peak_velocity_mps = velocity_mps
            peak_start_index = i

    return peak_velocity_mps, peak_start_index



def calculate_average_wrist_velocity(right_wrist_coords, fps, shoulder_to_hip_length):
    """
    Calculate the highest wrist velocity where z increases, normalized by shoulder-to-hip length.
    Also determines if the wrist generally moves to the right (positive x direction) and whether both y and z increase overall.

    Parameters:
        right_wrist_coords (list): List of right wrist coordinates [(x, y, z), ...] for each frame.
        fps (int): Frames per second of the video.
        shoulder_to_hip_length (float): Distance between the shoulder and hip, used for normalization.

    Returns:
        tuple: (float, bool, bool)
            - The maximum weighted wrist velocity where z increases, normalized by shoulder-to-hip length.
            - A boolean indicating whether the wrist moves predominantly to the right (True) or left (False).
            - A boolean indicating whether both y and z values increase overall (True) or not (False).
    """
    if shoulder_to_hip_length <= 0:
        raise ValueError("Shoulder-to-hip length must be greater than 0 for normalization.")

    weighted_velocities = []
    cumulative_x_movement = 0
    cumulative_y_movement = 0
    cumulative_z_change = 0
    time_interval = 1 / fps  # Time between consecutive frames
    normalization_factor = 1 / shoulder_to_hip_length  # Normalize motion by body proportion

    for i in range(1, len(right_wrist_coords)):  # Iterate through frames sequentially
        if right_wrist_coords[i] is None or right_wrist_coords[i - 1] is None:
            continue  # Skip missing frames

        # Extract coordinates for the current and previous frames
        x1, y1, z1 = right_wrist_coords[i - 1]
        x2, y2, z2 = right_wrist_coords[i]

        # Track cumulative movement in x, y, and z directions
        cumulative_x_movement += (x2 - x1)
        cumulative_y_movement += (y2 - y1)
        cumulative_z_change += (z2 - z1)

        # Only consider frames where the wrist moves forward in depth (z increases)
        if z2 > z1:
            # Compute displacement (3D distance between two consecutive positions)
            displacement = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

            # Compute velocity (displacement per unit time)
            velocity = (displacement * normalization_factor) / time_interval

            # Weight velocity by its displacement magnitude
            weighted_velocity = velocity * displacement
            weighted_velocities.append(weighted_velocity)

    # Determine if wrist movement is generally towards the right
    moves_right = cumulative_x_movement > 0

    # Check if both y and z have a net increase
    #z_and_y_increase = (cumulative_y_movement > 0) and (cumulative_z_change > 0)

    # Return the highest recorded weighted velocity, movement direction, and overall y/z increase
    max_velocity = np.max(weighted_velocities) if weighted_velocities else 0.0
    return max_velocity, moves_right


