import numpy as np

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


