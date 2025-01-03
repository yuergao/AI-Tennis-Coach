import os
import cv2
import mediapipe as mp
import numpy as np
import shutil

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
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


# Function to calculate the angle between three points (p1, p2, p3) using the law of cosines
def calculate_angle(p1, p2, p3):
    # Calculate the vector from p2 to p1 and from p2 to p3
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Calculate the dot product and the magnitudes of the vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Calculate the angle in radians and convert it to degrees
    angle = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    return np.degrees(angle)


# Function to calculate body orientation and distances for each frame
def calculate_body_orientation(keypoints):
    orientations = []
    for keypoint in keypoints:
        if keypoint is not None:
            try:
                # Key body points (only x and y coordinates are used for rotation along the y-axis)
                rs_x, rs_y = keypoint[12][:2]  # Right shoulder
                ls_x, ls_y = keypoint[11][:2]  # Left shoulder
                rh_x, rh_y = keypoint[24][:2]  # Right hip
                lh_x, lh_y = keypoint[23][:2]  # Left hip
                lw_x, lw_y = keypoint[15][:2]  # Left wrist
                rw_x, rw_y = keypoint[16][:2]  # Right wrist
                le_x, le_y = keypoint[13][:2]  # Left elbow
                re_x, re_y = keypoint[14][:2]  # Right elbow
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
                    "rs_x":rs_x,
                    "rh_x": rh_x,
                    "re_x":re_x,
                    "lw_x": lw_x,
                    "elbow_angle_left": elbow_angle_left,  # Angle of the left arm at the elbow
                    "elbow_angle_right": elbow_angle_right  # Angle of the right arm at the elbow
                })
            except Exception as e:
                orientations.append(None)
        else:
            orientations.append(None)
    return orientations

# Pose 1 Criteria: Body rotated as far right as possible (body_rotation as negative as possible), left hand extended
# and left arm as straight as possible (least elbow bend)
def pose1_criteria(orientation, i):
    # Body rotation should be negative (right rotated)
    # Left hand extended should be positive
    # Elbow angles (left) should be minimized (as straight as possible)
    score = (orientation["body_rotation"] - orientation["left_hand_position"] + 
            orientation["elbow_angle_left"])
    #print(f"frame{i}: {score}")
    return score
def pose2_criteria(orientation, i):
    """
    Criteria for evaluating a pose.
    
    Parameters:
        orientation (dict): A dictionary containing keypoints of the pose.
    
    Returns:
        float: A score based on the criteria (lower is better).
    """
    # Minimize the body rotation (i.e., body_rotation should be as close to 0 as possible)
    body_rotation_score = abs(orientation["body_rotation"])

    # Ensure the entire right arm is on the right side of the body
    arm_on_right_side = orientation["rw_x"] > orientation["rs_x"] and orientation["re_x"] > orientation["rs_x"]
    arm_on_right_side_score = 0 if arm_on_right_side else float("inf")  # 0 if arm is on the right side, inf if not

    # Ensure the left hand is to the left of the right shoulder
    left_hand_right_of_right_shoulder = orientation["lw_x"] < orientation["rs_x"]
    left_hand_position_score = 0 if left_hand_right_of_right_shoulder else float("inf")  # 0 if condition is met, inf otherwise

    # Combine all criteria (can be weighted if necessary)
    total_score = body_rotation_score + arm_on_right_side_score + left_hand_position_score
    if total_score < float("inf"):
        pass
        #print(f"frame{i}: {total_score}")  # Debugging output
    return total_score
  
def find_best_frame(orientations, criteria, start_frame= 2, end_frame = -2):
    best_frame = None
    best_score = float("inf")
    for i, orientation in enumerate(orientations[start_frame:end_frame], start=start_frame):
        if orientation:
            score = criteria(orientation, i)
            if score < best_score:
                best_score = score
                best_frame = i
    return best_frame  


# Function to find the best matching frames for each pose
def find_most_parallel_and_furthest_segment(keypoints1, keypoints2):
    
    # Calculate orientations for both videos
    video1_orientations = calculate_body_orientation(keypoints1)
    video2_orientations = calculate_body_orientation(keypoints2)

    # Find best frames for Pose 1 in both videos
    #print("---------vid 1 pose 1---------")
    video1_pose1_frame = find_best_frame(video1_orientations, pose1_criteria)
    #print("---------vid 2 pose 1---------")
    video2_pose1_frame = find_best_frame(video2_orientations, pose1_criteria)

    # Pose 2 is only evaluated after Pose 1 is found, and it should start after Pose 1 frame
    if video1_pose1_frame is not None and video2_pose1_frame is not None:
        #print("---------vid 1 pose 2---------")
        video1_pose2_frame = find_best_frame(video1_orientations, pose2_criteria, start_frame=video1_pose1_frame + 1)
        #print("---------vid 2 pose 2---------")
        video2_pose2_frame = find_best_frame(video2_orientations, pose2_criteria, start_frame=video2_pose1_frame + 1)
    else:
        # If Pose 1 is not found, Pose 2 cannot be evaluated
        video1_pose2_frame = None
        video2_pose2_frame = None

    # Print results
    print(f"Pose 1: (Video 1 frame = {video1_pose1_frame}, Video 2 frame = {video2_pose1_frame})")
    print(f"Pose 2: (Video 1 frame = {video1_pose2_frame}, Video 2 frame = {video2_pose2_frame})")

    return (
        (video1_pose1_frame, video2_pose1_frame),
        (video1_pose2_frame, video2_pose2_frame)
    )

def resize_and_pad(frame, target_width, target_height):
    original_height, original_width = frame.shape[:2]
    original_aspect_ratio = original_width / original_height

    target_aspect_ratio = target_width / target_height

    if original_aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / original_aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * original_aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))

    padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

    return padded_frame

def drop_frames_to_match_fps(video_path, target_fps, output_path):
    """
    Adjusts the frame rate of a video by dropping frames to match the target FPS.
    
    Args:
        video_path (str): Path to the input video.
        target_fps (float): Target frames per second.
        output_path (str): Path to save the output video.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Get input video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original FPS: {original_fps}, Target FPS: {target_fps}, Total Frames: {frame_count}")

    # Validate FPS
    if target_fps > original_fps:
        raise ValueError("Target FPS cannot be higher than the original FPS.")

    # Calculate frame skipping interval
    skip_interval = original_fps / target_fps

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    if not out.isOpened():
        raise IOError(f"Cannot create output video file: {output_path}")

    frame_idx = 0  # Keep track of current frame index
    frames_written = 0  # Track the number of frames written

    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break

        # Write frames that align with the skip interval
        if int(frame_idx / skip_interval) == frames_written:
            out.write(frame)
            frames_written += 1

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Video processed successfully and saved to {output_path}")
    
    
def sync_video_timings(video1_path, video2_path, output_path, top_bottom_original_path):  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(top_bottom_original_path), exist_ok=True)
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open video files.")
        return

    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    
    # sync videos to lower fps
    if fps1 != fps2:
        if fps1 > fps2:
            paths  = video1_path.split('/')
            newpath = f'{paths[0]}/{paths[1]}/{paths[1]}_{fps2}_fps.mp4'
            drop_frames_to_match_fps(video1_path,fps2,newpath)
            video1_path = newpath
            cap1 = cv2.VideoCapture(video1_path)
            fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
        else:
            paths  = video2_path.split('/')
            newpath = f'{paths[0]}/{paths[1]}/{paths[1]}_{fps1}_fps.mp4'
            drop_frames_to_match_fps(video2_path,fps1,newpath)
            video2_path = newpath
            cap2 = cv2.VideoCapture(video2_path)
            fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    
    # find output video ratio
    original_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if original_height > original_width:
        target_width = 720
        target_height = 1280
        output_video_width = target_width * 2
        output_video_height = target_height
    else:
        target_width = 1920
        target_height = 1080
        output_video_width = target_width
        output_video_height = target_height * 2

    keypoints1, scales1 = extract_keypoints_and_scale(video1_path)
    keypoints2, scales2 = extract_keypoints_and_scale(video2_path)


    # find first valid frame between 2 videos 
    first_valid_frame_idx = None
    for idx in range(len(keypoints1)):
        if keypoints1[idx] is not None and keypoints2[idx] is not None:
            first_valid_frame_idx = idx
            break

    if first_valid_frame_idx is None:
        print("Error: Could not find a valid frame with keypoints for both players.")
        return

    # use body metrics to calculate time offset and sync player swings
    frame_sync_stamps = find_most_parallel_and_furthest_segment(keypoints1,keypoints2)
    vid1_start_sync = frame_sync_stamps[0][0]
    vid2_start_sync = frame_sync_stamps[0][1]
    vid1_end_sync = frame_sync_stamps[1][0]
    vid2_end_sync = frame_sync_stamps[1][1]
    time_offset = vid2_start_sync - vid1_start_sync #between first poses - prep
    vid1_swing_len = vid1_end_sync - vid1_start_sync
    vid2_swing_len = vid2_end_sync - vid2_start_sync
    
    print(f"Time offset between {video1_path} and {video2_path} start swing: {time_offset} frames")
    
    min_fps = min(fps1, fps2)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), min_fps, (target_width, target_height))   
    out3 = cv2.VideoWriter(top_bottom_original_path, cv2.VideoWriter_fourcc(*'mp4v'), min_fps, (output_video_width, output_video_height))

    scale_ratio = scales1[first_valid_frame_idx] / scales2[first_valid_frame_idx]

    frame_idx1 = first_valid_frame_idx
    frame_idx2 = first_valid_frame_idx 

    if time_offset > 0:
        for _ in range(time_offset):
            cap2.read()
            frame_idx2 += 1
    elif time_offset < 0:
        for _ in range(-time_offset):
            cap1.read()
            frame_idx1 += 1

    # Frame counters
    frame_counter1 = 0
    frame_counter2 = 0
    
    # ------------------Choose frames to be duplicated, curr assumes vid2 pro swing is shorter------------------
    # TODO: detect which video swing is longer and sync accordingly. currently assume player1 swing is longer
    dup_frame_idx_list = [] #initialize frames to be duplicated based on interval
    if vid1_swing_len > vid2_swing_len: #video 1 has the longer swing, duplicate vid2 pro frames
        frame_diff = vid1_swing_len - vid2_swing_len
        interval = max(1,vid2_swing_len // frame_diff) # so interval at least 1 not 0
        dup_frame_idx = vid2_start_sync
        for _ in range(frame_diff):
            dup_frame_idx_list.append(dup_frame_idx)
            dup_frame_idx+= interval
        print(f"frames to be duplicated for video 2: {dup_frame_idx_list}")
        
    while True:
        ret1, frame1 = cap1.read()
        
        #skip reading frame2 if it should be duplicated
        skip = False
        if frame_idx2 in dup_frame_idx_list:
            dup_frame_idx_list.remove(frame_idx2)
            skip = True
        else:
            ret2, frame2 = cap2.read()    

        if not ret1 or not ret2:
            print(f"exited{frame_idx1, frame_idx2}")
            break
        
        # Stop when one video ends (based on the shorter video)
        if frame_idx1 > len(keypoints1) or frame_idx2 > len(keypoints2):
            break
        print(frame_idx1, frame_idx2)
        # ------------------     
            
        if frame_idx1 < len(keypoints1) and frame_idx2 < len(keypoints2):
            if keypoints1[frame_idx1] and keypoints2[frame_idx2]:

                fixed_scale1 = scales1[frame_idx1]
                adjusted_scale2 = scales2[frame_idx2] * scale_ratio
                aligned_kp1 = align_keypoints(keypoints1[frame_idx1], fixed_scale1, fixed_scale1, target_center=(0.5, 0.5))
                aligned_kp2 = align_keypoints(keypoints2[frame_idx2], scales2[frame_idx2], adjusted_scale2, target_center=(0.5, 0.5))

                blank_frame = np.zeros_like(frame1)
                overlay_frame = draw_overlay(blank_frame, aligned_kp1, aligned_kp2)

                overlay_frame = resize_and_pad(blank_frame, target_width, target_height)
                padded_frame1 = resize_and_pad(frame1, target_width, target_height)
                padded_frame2 = resize_and_pad(frame2, target_width, target_height)

                if original_height > original_width:
                    combined_frame = np.hstack([padded_frame1, padded_frame2])
                else:
                    combined_frame = np.vstack([padded_frame1, padded_frame2])

                out.write(overlay_frame)
                out3.write(combined_frame)

        frame_idx1 += 1
        # if frame2 is duplicate frane don't increrase idx
        if not skip:
            frame_idx2 += 1
        frame_counter1 += 1
        frame_counter2 += 1

    cap1.release()
    cap2.release()
    out.release()
    out3.release()

    print(f"Synchronized overlay video saved to {output_path}")
    print(f"Top-bottom original video saved to {top_bottom_original_path}")
    
# Function to overlay keypoints onto a blank frame
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

# Function to extract keypoints and reference scale from a video
def extract_keypoints_and_scale(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    scales = []

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
            keypoints.append([(lm.x, lm.y) for lm in landmarks])

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
    return keypoints, scales

# Function to align keypoints by scaling and translating
def align_keypoints(keypoints, scale, target_scale, target_center):
    if keypoints is None:
        return None

    # Calculate the centroid of the keypoints
    centroid = np.mean(keypoints, axis=0)

    # Scale and translate keypoints
    aligned_keypoints = [
        ((x - centroid[0]) * target_scale / scale + target_center[0],
         (y - centroid[1]) * target_scale / scale + target_center[1])
        for x, y in keypoints
    ]
    return aligned_keypoints

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


    

#------------------------------Established FUnctions below---------------------------------------------

def detect_swing_boundaries(keypoints):
    """Detect start and end indices of tennis swings based on wrist position."""
    hip_threshold=0.08
    prep_ignore_interval = 75
    wrist_in_front_frames = []  # List to keep track of frames when wrists are in front
    feet_threshold = 0.1

    # Look for key aspects of tennis swing based on wrist position
    for i, frame_keypoints in enumerate(keypoints):
        if frame_keypoints is None:
            continue

        # Get the keypoints for the wrists and hips
        right_wrist = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value][:2])
        left_wrist = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value][:2])
        right_hip = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value][:2])
        left_hip = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_HIP.value][:2])
        right_foot = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value][:2])
        left_foot = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][:2])



        # Calculate the hip center
        hip_center = np.array([(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2])

        # Calculate distances from wrists to hip center
        left_wrist_distance = np.linalg.norm(left_wrist - hip_center)
        right_wrist_distance = np.linalg.norm(right_wrist - hip_center)

        # Check if both wrists are within the distance threshold
        wrists_in_front = left_wrist_distance < hip_threshold and right_wrist_distance < hip_threshold

        # Calculate distance between the feet
        feet_distance = abs(left_foot[0] - right_foot[0])
        
        # Check if the feet are wide enough
        feet_wide = feet_distance > feet_threshold
        
       # Record the frame if wrists are in front and conditions are met
        if wrists_in_front and feet_wide:
            if not wrist_in_front_frames or (i - wrist_in_front_frames[-1] >= prep_ignore_interval):
                wrist_in_front_frames.append(i)
                print(f"----------prep position at {i/30:.1f} seconds---------")
                print(f"{i/30:.1f} seconds & frame{i}: {left_wrist_distance:.4f},{right_wrist_distance:.4f}/{hip_threshold} hip threshold")
                print(f"{i/30:.1f} seconds & frame{i}: {feet_distance} feet distance\n")
            
        # Debugging: Print comparison for wrists together
        #print(f"{i/30:.1f} seconds & frame{i}: {left_wrist_distance:.4f},{right_wrist_distance:.4f}/{hip_threshold} hip threshold")
        #print(f"{i/30:.1f} seconds & frame{i}: {feet_distance} feet distance\n")
    
    '''
     wrists together calculation
        wrist_threshold=0.05
        
        # Check if wrists are together (x-coordinates close to each other)
        wrists_together = abs(right_wrist[0] - left_wrist[0]) < wrist_threshold
       
        
        # Debugging: Print comparison for wrists together
        print(f"Second {i/30:.1f}: {abs(right_wrist[0] - left_wrist[0]):.4f}/{wrist_threshold} wrists together threshold")
        
        wrist_in_front = wrists_together and wrists_in_front
        
        # Detect swing start (when both wrists are in front of the body)
        if wrist_in_front and not in_swing:
            in_swing = True
            start = i
            print(f"Swing start detected at frame {i} (Both wrists in front of body)")

        # Detect swing end (when both wrists return to in front of the body again)
        elif wrist_in_front and in_swing:
            # Swing ends when both wrists are back in front of the body again
            motions.append((start, i))
            print(f"Swing end detected at frame {i} (Both wrists returned to front of body)")
            in_swing = False  # Reset for next swing
    '''

    return wrist_in_front_frames



def segment_swings(swing_frames):
    """
    Segments the keypoints into swings based on detected swing frames.
    
    Args:
        keypoints (list): List of keypoints for each frame.
        swing_frames (list): List of frame indices marking swing boundaries.
        
    Returns:
        list: List of tuples representing start and end frame indices for swings.
    """
    if not swing_frames or len(swing_frames) < 2:
        return []

    swings = [(swing_frames[i], swing_frames[i + 1]) for i in range(len(swing_frames) - 1)]
    return swings


def extract_motion_segment(video_path, start_frame, end_frame, output_path):
    """Extract a segment of the video based on frame indices."""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    if start_frame < 0 or end_frame >= total_frames or start_frame >= end_frame:
        print(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}, total_frames={total_frames}")
        return

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Jump to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    print(f"Extracting video segment: {start_frame} to {end_frame}. {(end_frame-start_frame)//30} seconds long")
    
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_idx} could not be read, ending extraction.")
            break
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Segment saved to {output_path}")
    
def clear_output_folder(output_folder):
    """Clear the contents of the output folder."""
    if os.path.exists(output_folder):
        # Loop through all files in the directory and remove them
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directories if any


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



def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            # If landmarks are detected
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Collect keypoints (only the first two dimensions for position)
                frame_keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
                keypoints.append(frame_keypoints)
    
    cap.release()
    return np.array(keypoints)


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
'''
-----------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------- Begin Driver--------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''
# User inputs
# TODO: make into classes
swing_type = 'forehand'
extension = '.mp4'

player1 = 'audrey'
player1_vid_idx = 3
player1_swing_idx = 1

player2 = 'badosa'
player2_vid_idx = 1
player2_swing_idx = 2


# construct based on user inputs
player1_vid = f"{player1}{player1_vid_idx}"
player1_vid_swing = f"{player1_vid}_{swing_type}{player1_swing_idx}"

player2_vid = f"{player2}{player2_vid_idx}"
player2_vid_swing = f"{player2_vid}_{swing_type}{player2_swing_idx}"

video1_path = f"{player1}/{player1_vid}/{swing_type}/{player1_vid_swing}{extension}"
video2_path = f"{player2}/{player2_vid}/{swing_type}/{player2_vid_swing}{extension}"

pipeline1 = True
pipeline2 = True

'''
-----------------------------------------------------------------------------------------------------------------------------------
---------------------- Pipeline Segment 1: segment original full video into separate swings----------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''

# TODO: detect swing types (forehand, backhand) and output into respective folders, for now all forehand
original_video1_path = f"{player1}/{player1_vid}/{player1_vid}{extension}"
original_video2_path = f"{player2}/{player2_vid}/{player2_vid}{extension}"

''' controls which video to be segmented'''
video_paths = [original_video1_path]

#extract_and_annotate("videos/Federer1.mp4", 'results/Federer1.mp4')

# Extract keypoints for each video 
#my_swing_keypoints = extract_keypoints_from_video(video1_path)
#pro_swing_keypoints = extract_keypoints_from_video(video2_path)

# Extract swing features for comparison and feedback
#my_swing_features = extract_swing_features(my_swing_keypoints)
#pro_swing_features = extract_swing_features(pro_swing_keypoints)

if pipeline1:
    for video_path in video_paths:
        output_levels = video_path.split('/')
        output_folder = f"{output_levels[0]}/{output_levels[1]}"   
        output_path = f"{output_folder}/{swing_type}"
        
        clear_output_folder(output_path)
        
        #Extract keypoints from the video
        keypoints = extract_keypoints_from_video(video_path)

        # Detect swing boundaries
        swing_frames = detect_swing_boundaries(keypoints)

        # Segment swings based on the detected frames
        swing_segments = segment_swings(swing_frames)

        print(f"Detected {len(swing_segments)} swings.")

        # Process each swing and extract video segments
        for idx, (start_frame, end_frame) in enumerate(swing_segments):
            if start_frame >= 0 and end_frame > start_frame:
                #find previous directory
                output_video = f"{output_path}/{output_levels[1]}_{swing_type}{idx+1}{extension}"
                extract_motion_segment(video_path, start_frame, end_frame, output_video)


'''
-----------------------------------------------------------------------------------------------------------------------------------
---------------------- Pipeline Segment 2: output segmented swing comparison videos------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''


# TODO: seperate code into separate files: driver, pipeline1, pipeline2

if pipeline2:
    # output 
    player_compare = f"{player1}_{player2}"
    player_compare_vid = f"{player1_vid}_{player2_vid}"
    player_compare_vid_swing = f"{player_compare_vid}_{swing_type}{player1_swing_idx}_{player2_swing_idx}"
    overlay_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}/{player_compare_vid_swing}_overlay{extension}"
    comparison_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}/{player_compare_vid_swing}{extension}"

    #create_aligned_overlay_video(video1, video2, output_video)

    # Call the function to sync the timing and overlay the videos
    sync_video_timings(video1_path, video2_path, overlay_path, comparison_path)

