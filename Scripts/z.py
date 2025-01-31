import numpy as np
from mediapipe import solutions as mp_solutions
import cv2

"""
--------------------- EXTRA UNUSED FUNCTIONS FILE FOR ARCHIVE PURPOSES --------------------
"""

mp_pose = mp_solutions.pose
pose = mp_pose.Pose()
# Function to extract keypoints and detect the start of the swing
def detect_swing_start(keypoints, threshold=0.5):
    """
    Detects the start of the swing based on the movement of the wrist (right wrist as an example).
    
    :param keypoints: List of keypoints for a frame.
    :param threshold: The movement threshold that signifies the start of the swing.
    :return: True if swing start is detected, False otherwise.
    """
    if keypoints is None or len(keypoints) <= 28:
        return False
    
    # Assuming right wrist is at index 28
    wrist_x, wrist_y = keypoints[28]  # Right wrist keypoint

    if wrist_x is not None and wrist_y is not None:
        # Check for a significant movement in the wrist (you can adjust this based on what you consider a "swing start")
        if wrist_y > threshold:  # You can use any condition that identifies the swing start
            return True
    return False


def create_aligned_overlay_video(video1_path, video2_path, annotated1_path, annotated2_path, overlay_output_path, side_by_side_annotated_path, side_by_side_original_path):
    # Extract keypoints and scales
    keypoints1, scales1 = extract_keypoints_and_scale(video1_path)
    keypoints2, scales2 = extract_keypoints_and_scale(video2_path)

    # Find the first valid frame for both players
    first_valid_frame_idx = None
    for idx in range(len(keypoints1)):
        if keypoints1[idx] is not None and keypoints2[idx] is not None:
            first_valid_frame_idx = idx
            break

    if first_valid_frame_idx is None:
        print("Error: Could not find a valid frame with keypoints for both players.")
        return

    # Initialize video capture for all videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    cap_annot1 = cv2.VideoCapture(annotated1_path)
    cap_annot2 = cv2.VideoCapture(annotated2_path)

    # Video properties
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

    # Output video writers
    overlay_out = cv2.VideoWriter(overlay_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps1, (width, height))
    side_by_side_annot_out = cv2.VideoWriter(side_by_side_annotated_path, cv2.VideoWriter_fourcc(*'mp4v'), fps2, (width * 2, height))
    side_by_side_orig_out = cv2.VideoWriter(side_by_side_original_path, cv2.VideoWriter_fourcc(*'mp4v'), max(fps1, fps2), (width * 2, height))

    frame_idx = first_valid_frame_idx
    target_center = (0.5, 0.5)  # Normalize to the center of the frame

    # Initialize scale ratio based on the first valid frame
    scale_ratio = scales1[first_valid_frame_idx] / scales2[first_valid_frame_idx]

    while cap1.isOpened() and cap2.isOpened() and cap_annot1.isOpened() and cap_annot2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret_annot1, frame_annot1 = cap_annot1.read()
        ret_annot2, frame_annot2 = cap_annot2.read()

        if not ret1 or not ret2 or not ret_annot1 or not ret_annot2 or frame_idx >= len(keypoints1) or frame_idx >= len(keypoints2):
            break

        # Create a blank frame for the overlay video
        blank_frame = np.zeros_like(frame1)

        # Align keypoints and draw overlay
        if keypoints1[frame_idx] and keypoints2[frame_idx]:
            # Keep the first player's size fixed
            fixed_scale1 = scales1[frame_idx]
            adjusted_scale2 = scales2[frame_idx] * scale_ratio

            aligned_kp1 = align_keypoints(keypoints1[frame_idx], fixed_scale1, fixed_scale1, target_center)
            aligned_kp2 = align_keypoints(keypoints2[frame_idx], scales2[frame_idx], adjusted_scale2, target_center)

            blank_frame = draw_overlay(blank_frame, aligned_kp1, aligned_kp2)

        # Write to overlay video
        overlay_out.write(blank_frame)

        # Create side-by-side videos
        side_by_side_annot = np.hstack((frame_annot1, frame_annot2))
        side_by_side_orig = np.hstack((frame1, frame2))

        side_by_side_annot_out.write(side_by_side_annot)
        side_by_side_orig_out.write(side_by_side_orig)

        frame_idx += 1

    # Release all resources
    cap1.release()
    cap2.release()
    cap_annot1.release()
    cap_annot2.release()
    overlay_out.release()
    side_by_side_annot_out.release()
    side_by_side_orig_out.release()
    
    
    
def extract_swing_features(keypoints):
    """Extract features from keypoints for the given swing segment."""
    wrist_positions = []
    elbow_positions = []
    shoulder_positions = []
    hip_positions = []
    
    for frame_keypoints in keypoints:
        if frame_keypoints is None:
            continue

        # Get the keypoints for the wrists, elbows, shoulders, and hips
        right_wrist = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value][:2])
        left_wrist = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value][:2])
        right_elbow = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value][:2])
        left_elbow = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value][:2])
        right_shoulder = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][:2])
        left_shoulder = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value][:2])
        right_hip = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value][:2])
        left_hip = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_HIP.value][:2])

        # Store the keypoints for analysis
        wrist_positions.append((right_wrist, left_wrist))
        elbow_positions.append((right_elbow, left_elbow))
        shoulder_positions.append((right_shoulder, left_shoulder))
        hip_positions.append((right_hip, left_hip))

    # Return the features as a tuple of lists
    return wrist_positions, elbow_positions, shoulder_positions, hip_positions

def compare_swing_features(user_features, pro_features):
    """Compare the swing features of user and pro."""
    wrist_distance = 0
    elbow_distance = 0
    shoulder_distance = 0
    hip_distance = 0
    
    # Assuming the swing features are lists of positions
    # Compare each feature (wrist, elbow, etc.) by calculating Euclidean distance
    for user_wrist, pro_wrist in zip(user_features[0], pro_features[0]):
        wrist_distance += np.linalg.norm(user_wrist[0] - pro_wrist[0]) + np.linalg.norm(user_wrist[1] - pro_wrist[1])
    
    for user_elbow, pro_elbow in zip(user_features[1], pro_features[1]):
        elbow_distance += np.linalg.norm(user_elbow[0] - pro_elbow[0]) + np.linalg.norm(user_elbow[1] - pro_elbow[1])
    
    for user_shoulder, pro_shoulder in zip(user_features[2], pro_features[2]):
        shoulder_distance += np.linalg.norm(user_shoulder[0] - pro_shoulder[0]) + np.linalg.norm(user_shoulder[1] - pro_shoulder[1])
    
    for user_hip, pro_hip in zip(user_features[3], pro_features[3]):
        hip_distance += np.linalg.norm(user_hip[0] - pro_hip[0]) + np.linalg.norm(user_hip[1] - pro_hip[1])

    total_distance = wrist_distance + elbow_distance + shoulder_distance + hip_distance

    # Provide feedback based on distance
    feedback = ""
    if total_distance < 1000:  # This threshold can be adjusted based on how far you expect the swings to be
        feedback = "Your swing is very similar to the pro swing!"
    else:
        feedback = "There are significant differences between your swing and the pro swing. Work on improving your wrist and elbow movements."

    return total_distance, feedback



"""Extract and annotate pose keypoints from a video."""
def extract_and_annotate(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video extract_and_annotate.")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        blank_frame = np.zeros((height, width, 3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x, y, z, visibility = landmark.x, landmark.y, landmark.z, landmark.visibility
                cv2.circle(blank_frame, (int(x * width), int(y * height)), 5, (0, 255, 0), -1)

        out.write(blank_frame)
        cv2.imshow('Annotated Video', blank_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    '''
# Function to overlay keypoints onto a blank frame. 
# called by sync_video_timings
def draw_overlay(frame, keypoints1, keypoints2, color1=(0, 255, 0), color2=(0, 0, 255)):
    """
    Draws the keypoints of two sets of landmarks onto a given frame.
    
    :param frame: The blank frame onto which keypoints are drawn.
    :param keypoints1: Aligned keypoints for the first person.
    :param keypoints2: Aligned keypoints for the second person.
    :param color1: Color for the first person's keypoints.
    :param color2: Color for the second person's keypoints.
    """
    height, width, _ = frame.shape

    # Draw keypoints for the first person
    if keypoints1:
        for x1, y1 in keypoints1:
            px1, py1 = int(x1 * width), int(y1 * height)
            cv2.circle(frame, (px1, py1), 5, color1, -1)

    # Draw keypoints for the second person
    if keypoints2:
        for x2, y2 in keypoints2:
            px2, py2 = int(x2 * width), int(y2 * height)
            cv2.circle(frame, (px2, py2), 5, color2, -1)

    return frame
    
def crop_to_square(frame):
    """Crops the frame to a square, centered on the middle of the frame."""
    original_height, original_width = frame.shape[:2]

    # Calculate the side length for the square crop
    side_length = min(original_width, original_height)

    # Calculate starting points to crop the center
    crop_x_start = (original_width - side_length) // 2
    crop_x_end = crop_x_start + side_length
    crop_y_start = (original_height - side_length) // 2
    crop_y_end = crop_y_start + side_length

    # Crop the frame to the square
    cropped_frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    return cropped_frame
    
# Function to align keypoints by scaling and translating
# called by sync_video_timings
def align_keypoints(keypoints, scale, target_scale, target_center):
    if keypoints is None:
        return None

    # Calculate the centroid of the keypoints
    centroid = np.mean(keypoints, axis=0)

    # Scale and translate keypoints
    aligned_keypoints = [
        ((x - centroid[0]) * target_scale / scale + target_center[0],
         (y - centroid[1]) * target_scale / scale + target_center[1])
        for x, y, z in keypoints
    ]
    return aligned_keypoints
    
def extract_keypoints_and_scale(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    scales = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            # Extract keypoints
            landmarks = result.pose_landmarks.landmark
            keypoints.append([(lm.x, lm.y, lm.z) for lm in landmarks])

            # Calculate reference scale using shoulders and hips
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            shoulder_distance = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 +
                                        (left_shoulder.y - right_shoulder.y) ** 2)
            hip_distance = np.sqrt((left_hip.x - right_hip.x) ** 2 +
                                   (left_hip.y - right_hip.y) ** 2)

            # Use the average of shoulder and hip distances
            average_scale = (shoulder_distance + hip_distance) / 2
            scales.append(average_scale)
        else:
            keypoints.append(None)
            scales.append(None)

    cap.release()
    return keypoints, scales, fps   
    
def resize_and_pad(frame, target_width, target_height, pad_color=(0, 0, 0)):
    """
    Resize the frame to fit within the target dimensions, adding padding or cropping only if necessary.

    Parameters:
    - frame: Input image (numpy array).
    - target_width: Target width of the frame.
    - target_height: Target height of the frame.
    - pad_color: Color for padding (default: black).

    Returns:
    - final_frame: The resized and padded/cropped frame with target dimensions.
    """
    original_height, original_width = frame.shape[:2]
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Resize frame to fit within target dimensions
    if original_aspect_ratio > target_aspect_ratio:
        # Original is wider than target
        new_width = target_width
        new_height = int(target_width / original_aspect_ratio)
    else:
        # Original is taller or matches the target
        new_height = target_height
        new_width = int(target_height * original_aspect_ratio)

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Check if padding or cropping is required
    if new_width < target_width or new_height < target_height:
        # Create a blank canvas for padding
        padded_frame = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)

        # Center the resized frame on the canvas
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
        return padded_frame
    else:
        # Crop if dimensions exceed target
        x_start = (new_width - target_width) // 2
        y_start = (new_height - target_height) // 2
        cropped_frame = resized_frame[y_start:y_start + target_height, x_start:x_start + target_width]
        return cropped_frame
        
def find_end_frame_velocity(orientations, criteria, fps, start_frame=1, end_frame=-1, best_score = float("inf")):
    """
    Finds the best frame according to the pose criteria.
    Calculates wrist z changes every 0.3 seconds for every frame and identifies 
    the exact moment when z changes from decreasing to increasing.

    Parameters:
        orientations (list): List of keypoint orientations for each frame.
        criteria (function): A function to evaluate each frame.
        fps (int): Frames per second of the video.
        start_frame (int): Starting frame index.
        end_frame (int): Ending frame index.

    Returns:
        int: The index of the best frame where z changes from decreasing to increasing.
    """
    """
    best_frame = None
    time_interval_frames = int(0.6 * fps)
    
    for i in range(start_frame, end_frame - time_interval_frames, int(fps//15)):
        prev_orientation = orientations[i]
        orientation = orientations[i + time_interval_frames] if i + time_interval_frames < len(orientations) else None

        if orientation and prev_orientation:  # Ensure valid frames
            score = criteria(orientation, prev_orientation, i + time_interval_frames, fps)
            print(i + time_interval_frames,score, best_score)
            if score < float("inf") and score > best_score / 2:
                print("taken")
                return i + time_interval_frames, score
    """
    best_frame = None
    best_score = float("inf")
    time_interval_frames = int(0.6 * fps)
    ending_frame = None
    start_frame = max(1, start_frame)
    end_frame = min(len(orientations) -1 , end_frame)
    

    for i in range(start_frame, end_frame - time_interval_frames):
        prev_orientation = orientations[i]
        orientation = orientations[i + time_interval_frames] if i + time_interval_frames < len(orientations) else None

        if orientation and prev_orientation:  # Ensure valid frames
            score = criteria(orientation, prev_orientation, i + time_interval_frames, fps)
            if score < best_score:
                best_score = score
                best_frame = i + time_interval_frames
           
    for i in range(start_frame, end_frame - time_interval_frames):
        prev_orientation = orientations[i]
        orientation = orientations[i + time_interval_frames] if i + time_interval_frames < len(orientations) else None

        if orientation and prev_orientation:  # Ensure valid frames
            score = criteria(orientation, prev_orientation, i + time_interval_frames, fps)
            if score < float("inf") and  score > best_score/1.5 and i + time_interval_frames > best_frame:
                if orientation["rw_x"] < orientation["rs_x"] and orientation["rw_z"] > orientation["rs_z"]:
                    print(score, best_score)
                    ending_frame = i + time_interval_frames
                    break          
                
    return best_frame , best_score, ending_frame


# Function to calculate body orientation and distances for each frame
# called by find_most_parallel_and_furthest_segment
def calculate_body_orientation(keypoints):
    orientations = []
    for keypoint in keypoints:
        if keypoint is not None:
            try:
                # Key body points (only x and y coordinates are used for rotation along the y-axis)
                nose_x, nose_y, nose_z = keypoint[5][:3]
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
                elbow_angle_left = calculate_angle([ls_x, ls_y], [le_x, le_y], [lw_x, lw_y])
                # Calculate the angle between right shoulder, right elbow, and right wrist (to check for elbow bend)
                elbow_angle_right = calculate_angle([rs_x, rs_y], [re_x, re_y], [rw_x, rw_y])
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
                    "re_z": re_z,
                    "rw_z": rw_z,
                    "le_x": le_x,
                    "lh_x": lh_x,
                    "rh_y": rh_y,
                    "nose_x": nose_x,
                    "nose_z": nose_z,
                    "elbow_angle_left": elbow_angle_left,  # Angle of the left arm at the elbow
                    "elbow_angle_right": elbow_angle_right  # Angle of the right arm at the elbow
                })
            except Exception as e:
                orientations.append(None)
        else:
            orientations.append(None)
    return orientations

"""
Criterias for evaluating pose1 and pose2
Parameters:
    orientation (dict): A dictionary containing keypoints of the pose.
    i: frame
Returns:
    float: A score based on the criteria (lower is better).
"""
# Pose 1 Criteria: Body rotated as far right as possible (body_rotation as negative as possible), left hand extended
# and left arm as straight as possible (least elbow bend)
# called by find_best_frame
def forehand_start_criteria(orientation, i):
    # Body rotation should be negative (right rotated)
    # Left hand extended should be positive
    # Elbow angles (left) should be minimized (as straight as possible)
    
    body_rotation = orientation["body_rotation"] * 100
    body_rotation_score = -orientation["body_rotation"] if orientation["body_rotation"] > 0 else float('inf')
    
    left_hand_position = (orientation["lw_x"] - orientation["rs_x"]) * 1000
    left_hand_position_score = -left_hand_position if orientation["left_hand_position"] > 0 else float("inf")
    
    arm_on_right_side = orientation["rw_x"] > orientation["rs_x"] and orientation["lw_x"] > orientation["rs_x"]
    arm_on_right_side_score = 0 if arm_on_right_side else float("inf")
    
    elbow_angle_left = orientation["elbow_angle_left"]
    
    
    score = (body_rotation_score + left_hand_position_score + arm_on_right_side_score)
    
    if score < float("inf"):
        #print(f"frame: {i} score: {score} body rotation: {body_rotation} l_hand: {left_hand_position} el_angle: {elbow_angle_left}")
        pass
        
    #score = (orientation["body_rotation"] - orientation["left_hand_position"] + orientation["elbow_angle_left"])
    return score
  
# Backhand start Criteria
# called by find_best_frame
def backhand_start_criteria(orientation, i):
    """
    Pose 2 Criteria
    
    Parameters:
    - orientation: A dictionary containing body landmarks' x and y coordinates.
    - i: Current frame index.
    
    Returns:
    - A score where 0 means all conditions are met, and inf means a condition failed.
    """
    
    # Condition 1: both wrists to the left of the left shoulder
    arms_on_left_side = orientation["rw_x"] < orientation["ls_x"] and orientation["lw_x"] < orientation["ls_x"]
    arms_on_left_side_score = 0 if arms_on_left_side else float("inf")
    
    # Condition 2: Left wirst behind the right shoulder
    left_hand_behind_right_shoulder = orientation["lw_z"] < orientation["rs_z"]
    left_hand_behind_right_shoulder_score = orientation["lw_z"] - orientation["rs_z"] if left_hand_behind_right_shoulder else float("inf")
    
    # Condition 3: Body is rotated to the left (smaller z value for the left shoulder)
    body_rotated_left = orientation["ls_z"] < orientation["rs_z"]
    body_rotated_left_score = orientation["ls_z"] - orientation["rs_z"] if body_rotated_left else float("inf")
    
    # Condition 5: Right foot is in front of the left foot (smaller z value for the left foot)
    left_foot_in_front = orientation["la_z"] < orientation["ra_z"]
    foot_position_score = 0 if left_foot_in_front else float("inf")
    
    # Condition 7: Right wrist is below the right shoulder
    right_wrist_below_shoulder = orientation["rw_y"] > orientation["rs_y"]
    wrist_below_shoulder_score = 0 if right_wrist_below_shoulder else float("inf")
    #print(orientation["elbow_angle_left"], orientation["elbow_angle_right"])
    
   
    # Total score (0 means all conditions met, inf means at least one condition failed)
    score = (
        arms_on_left_side_score +
        left_hand_behind_right_shoulder_score +
        body_rotated_left_score +
        foot_position_score + 
        wrist_below_shoulder_score 
    )
    
    if score < float("inf"):
        print(f"frame: {i} score: {score}")
    return score

# backhand end Criteria: minimize body rotation while right arm is on right side of body and left arm is not on right side of body
def backhand_end_criteria(orientation, i):
    # Ensure both hands are on the right side of the body
    both_hands_on_right_side = orientation["rw_x"] > orientation["ls_x"] and orientation["lw_x"] > orientation["ls_x"]
    both_hands_on_right_side_score =  0 if both_hands_on_right_side else float("inf")  # 0 if arm is on the right side, inf if not

    # Condition 2: Left wirst in front of the right shoulder
    left_hand_front_of_right_shoulder = orientation["lw_z"] > orientation["rs_z"]
    left_hand_front_of_right_shoulder_score = 0 if left_hand_front_of_right_shoulder else float("inf")
    
    # Combine all criteria (can be weighted if necessary)
    total_score = both_hands_on_right_side_score + left_hand_front_of_right_shoulder_score
    
    return total_score
    
def find_best_frame_velocity(orientations, criteria, fps, start_frame=1, end_frame=-1):
    """
    Finds the best frame according to the pose criteria.
    Calculates wrist z changes every 0.3 seconds for every frame and identifies 
    the exact moment when z changes from decreasing to increasing.

    Parameters:
        orientations (list): List of keypoint orientations for each frame.
        criteria (function): A function to evaluate each frame.
        fps (int): Frames per second of the video.
        start_frame (int): Starting frame index.
        end_frame (int): Ending frame index.

    Returns:
        int: The index of the best frame where z changes from decreasing to increasing.
    """
    best_frame = None
    best_score = float("inf")
    time_interval_frames = int(0.6 * fps)

    for i in range(start_frame, end_frame - time_interval_frames, int(fps//15)):
        prev_orientation = orientations[i]
        orientation = orientations[i + time_interval_frames] if i + time_interval_frames < len(orientations) else None

        if orientation and prev_orientation:  # Ensure valid frames
            score = criteria(orientation, prev_orientation, i + time_interval_frames, fps)

            if score < best_score:
                best_score = score
                best_frame = i + time_interval_frames

    return best_frame, best_score

def detect_wrist_velocity_change(orientation, prev_orientation, frame_idx, fps, threshold=0):
    """
    Detects a significant change in wrist velocity in the z direction.
    Ensures there is a clear point where the hand moves back (smaller z) and then forward (larger z).

    Parameters:
        orientation (dict): Keypoint data for the current frame.
        prev_orientation (dict): Keypoint data for the previous frame (0.3 seconds earlier).
        frame_idx (int): Index of the current frame.
        fps (int): Frames per second of the video.
        threshold (float): Minimum velocity change to consider significant.

    Returns:
        float: The score for the current frame, lower is better.
               Frames with significant velocity changes will have lower scores.
    """
    # Extract z-coordinates for the wrist
    wrist_z = orientation['re_z']
    prev_wrist_z = prev_orientation['re_z']

    if wrist_z is None or prev_wrist_z is None:
        return float("inf")  # Skip if z data is missing

    # Check for a change in direction from decreasing to increasing
    if prev_wrist_z > wrist_z:
        # Calculate velocity change
        velocity_change = (wrist_z - prev_wrist_z) * 100  # Change in z over 0.3 seconds

        if velocity_change:
            print(f"Frame {frame_idx}: Velocity Change = {velocity_change}")
            return velocity_change  # Lower score for higher velocity changes

    return float("inf")  # Return a high score if no significant change

# Function to find the best matching frames for each pose
def find_most_parallel_and_furthest_segment(keypoints, swing_type, fps):
    # Calculate orientations
    video_orientations = calculate_body_orientation(keypoints)

    # Find the best frame for Pose 1
    if swing_type == 'backhand':
        pose1_frame = find_best_frame(video_orientations, backhand_start_criteria, start_frame = fps // 5, end_frame = int(1.5*fps))
    else:  # Assumes only forehand and backhand for now
        #pose1_frame = find_best_frame(video_orientations, forehand_start_criteria)
        pose1_frame, pose1_score, pose2_frame = find_end_frame_velocity(video_orientations, detect_wrist_velocity_change, fps, start_frame = fps // 5, end_frame = int(2.9*fps))

    # Pose 2 is evaluated only after Pose 1 is found, and it should start after Pose 1 frame
    if pose1_frame is not None:
        if swing_type == "backhand":
            pose2_frame = find_best_frame(video_orientations, backhand_end_criteria, start_frame=pose1_frame+2)
        else:  # Assumes only forehand and backhand for now
            pass
            #pose2_frame = find_best_frame(video_orientations, forehand_end_criteria1, start_frame=pose1_frame + 5)
            #pose2_frame, pose2_Score = find_end_frame_velocity(video_orientations, detect_wrist_velocity_change, fps, start_frame = pose1_frame, end_frame = int(3*fps), best_score = pose1_score)
    else:
        # If Pose 1 is not found, Pose 2 cannot be evaluated
        pose2_frame = None

    # Print results
    print(f"Pose 1: Frame = {pose1_frame}")
    print(f"Pose 2: Frame = {pose2_frame}")

    return (pose1_frame, pose2_frame)

# Pose 2 Criteria: minimize body rotation while right arm is on right side of body and left arm is not on right side of body
def forehand_end_criteria1(orientation, i):

    right_wrist_front_of_both_shoulder = orientation["rw_z"] < orientation["ls_z"] and orientation["rw_z"] < orientation["rs_z"] and orientation["rw_z"] < orientation["nose_z"]
    right_wrist_front_of_both_shoulder_score = 0 if right_wrist_front_of_both_shoulder else float("inf")  # 0 if arm is on the right side, inf if not

    left_wrist_left_of_both_shoulder = orientation["le_x"] < orientation["rs_x"] and orientation["le_x"] < orientation["rs_x"] and orientation["le_x"] < orientation["nose_x"]
    left_wrist_left_of_both_shoulder_score = 0 if left_wrist_left_of_both_shoulder else float("inf")  # 0 if arm is on the right side, inf if not


    right_wrist_left_of_right_shoulder = orientation["rw_x"] < orientation["rs_x"] and orientation["rw_x"] < orientation["nose_x"]
    right_wrist_left_of_right_shoulder_score = 0 if right_wrist_left_of_right_shoulder else float("inf")  # 0 if condition is met, inf otherwise

    # Combine all criteria (can be weighted if necessary)
    total_score = right_wrist_left_of_right_shoulder_score + right_wrist_front_of_both_shoulder_score + left_wrist_left_of_both_shoulder_score
    if total_score < float("inf"):
        pass
        #print(f"frame{i}: {total_score}")  # Debugging output
    return total_score

# Pose 2 Criteria: minimize body rotation while right arm is on right side of body and left arm is not on right side of body
def forehand_end_criteria2(orientation, i):

    right_wrist_left_of_head = orientation["rw_x"] < orientation["nose_x"]
    right_wrist_left_of_head_score = 0 if right_wrist_left_of_head else float("inf")  # 0 if arm is on the right side, inf if not

    right_wrist_front_of_both_shoulder = orientation["rw_z"] < orientation["ls_z"] and orientation["rw_z"] < orientation["rs_z"] and orientation["rw_z"] < orientation["nose_z"]
    right_wrist_front_of_both_shoulder_score = 0 if right_wrist_front_of_both_shoulder else float("inf")  # 0 if arm is on the right side, inf if not
    
    # Combine all criteria (can be weighted if necessary)
    total_score = right_wrist_left_of_head_score + right_wrist_front_of_both_shoulder_score
    if total_score < float("inf"):
        pass
        #print(f"frame{i}: {total_score}")  # Debugging output
    return total_score
'''

'''
# Pose 2 Criteria: minimize body rotation while right arm is on right side of body and left arm is not on right side of body
def forehand_end_criteria1(orientation, i):

    # Ensure the entire right arm is on the left side of the right shoulder
    arm_on_right_side = orientation["rw_x"] < orientation["rs_x"] 
    arm_on_right_side_score = 0 if arm_on_right_side else float("inf")  # 0 if arm is on the right side, inf if not

    # Ensure the left hand is to the left of the right shoulder
    left_hand_right_of_right_shoulder = orientation["lw_x"] < orientation["rs_x"]
    left_hand_position_score = 0 if left_hand_right_of_right_shoulder else float("inf")  # 0 if condition is met, inf otherwise

    # Combine all criteria (can be weighted if necessary)
    total_score = arm_on_right_side_score + left_hand_position_score
    if total_score < float("inf"):
        pass
        #print(f"frame{i}: {total_score}")  # Debugging output
    return total_score

'''