import os
import cv2
import numpy as np
from mediapipe import solutions as mp_solutions

mp_pose = mp_solutions.pose
pose = mp_pose.Pose()

# Called by driver. compares 2 video swings. outputs overlay video and comparison video. 
def sync_video_timings(video1_path, video2_path, overlay_path, comaprison_path, swing_type):  
    
    # create directories if don't already exist
    os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
    os.makedirs(os.path.dirname(comaprison_path), exist_ok=True)
    
    #initialize video capture
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open video files.")
        return

    # match frame rates - sync videos to lower fps
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    min_fps = min(fps1,fps2)
    
    if fps1 > fps2:
        paths  = video1_path.split('/')
        newpath = f'{paths[0]}/{paths[1]}/{swing_type}/{paths[1]}_{min_fps}_fps.mp4'
        if os.path.exists(newpath) and os.path.isfile(newpath):
            print(f"dropped frame video exists at {min_fps}")
        else:
            drop_frames_to_match_fps(video1_path,min_fps,newpath)
        video1_path = newpath
        cap1 = cv2.VideoCapture(video1_path)
        fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
        
    elif fps1 < fps2:
        paths  = video2_path.split('/')
        newpath = f'{paths[0]}/{paths[1]}/{swing_type}/{paths[1]}_{min_fps}_fps.mp4'
        if os.path.exists(newpath) and os.path.isfile(newpath):
            print(f"dropped frame video exists at {min_fps}")
        else:
            drop_frames_to_match_fps(video2_path,min_fps,newpath)
        video2_path = newpath
        cap2 = cv2.VideoCapture(video2_path)
        fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    
    # find output video ratio based on usesr video input orientation (vid1)
    vid1_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid1_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid2_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid2_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if vid1_width < vid1_height:
        target_width = vid1_width
        target_height = vid1_width
    else:    
        target_width = max(vid1_width, vid2_width)
        target_height = max(vid1_height, vid2_height)
    output_video_width = target_width 
    output_video_height = target_height + vid2_height

    # extract keypoints with scales in mind
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

    scale_ratio = scales1[first_valid_frame_idx] / scales2[first_valid_frame_idx]
    frame_idx1 = first_valid_frame_idx
    frame_idx2 = first_valid_frame_idx 
    
    # initialize video writer
    out1 = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*'mp4v'), min_fps, (target_width, target_height))   
    out2 = cv2.VideoWriter(comaprison_path, cv2.VideoWriter_fourcc(*'mp4v'), min_fps, (output_video_width, output_video_height))
    
    # use body metrics to calculate time offset and sync player swings
    frame_sync_stamps = find_most_parallel_and_furthest_segment(keypoints1,keypoints2, swing_type)
    vid1_start_sync = frame_sync_stamps[0][0]
    vid2_start_sync = frame_sync_stamps[0][1]
    vid1_end_sync = frame_sync_stamps[1][0]
    vid2_end_sync = frame_sync_stamps[1][1]
    time_offset = vid2_start_sync - vid1_start_sync # time between first poses - prep
    vid1_swing_len = vid1_end_sync - vid1_start_sync
    vid2_swing_len = vid2_end_sync - vid2_start_sync  
    print(f"Time offset: {time_offset} frames")
    print(f"vid1 start sync: {vid1_start_sync} | end sync: {vid1_end_sync} | swing len: {vid1_swing_len}")
    print(f"vid2 start sync: {vid2_start_sync} | end sync: {vid2_end_sync} | swing len: {vid2_swing_len}")
    
    # sync video start based on first pose
    if time_offset > 0:
        for _ in range(time_offset):
            cap2.read()
            frame_idx2 += 1
    elif time_offset < 0:
        for _ in range(-time_offset):
            cap1.read()
            frame_idx1 += 1
            
    # Choose frames to be duplicated, curr assumes vid2 pro swing is shorter
    # TODO: detect which video swing is longer and sync accordingly. currently assume player1 swing is longer
    dup_frame_idx_list = [] #initialize frames to be duplicated based on interval
    if vid1_swing_len > vid2_swing_len: #video 1 has the longer swing, duplicate vid2 pro frames
        frame_diff = vid1_swing_len - vid2_swing_len
        interval = max(1,vid2_swing_len // frame_diff) # so interval is at least 1 not 0
        dup_frame_idx = vid2_start_sync
        for _ in range(frame_diff):
            dup_frame_idx_list.append(dup_frame_idx)
            dup_frame_idx+= interval
        print(f"frames to be duplicated for video 2: {dup_frame_idx_list}")
        
    
    # enter frame processing loop    
    while True:
        
        # reads frame (TODO: change parts for swing length auto detection at dup_frame_idx_list = [])
        ret1, frame1 = cap1.read()
        skip = False # skip reading frame2 if it should be duplicated
        if frame_idx2 in dup_frame_idx_list:
            dup_frame_idx_list.remove(frame_idx2)
            skip = True
        else:
            ret2, frame2 = cap2.read()    

        if not ret1 or not ret2:
            break
        
        # Stop when the shorter video ends
        if frame_idx1 > len(keypoints1) or frame_idx2 > len(keypoints2):
            break
         
        # process valid frames, bulk of video processing    
        if frame_idx1 < len(keypoints1) and frame_idx2 < len(keypoints2):
            if keypoints1[frame_idx1] and keypoints2[frame_idx2]:

                fixed_scale1 = scales1[frame_idx1]
                adjusted_scale2 = scales2[frame_idx2] * scale_ratio
                aligned_kp1 = align_keypoints(keypoints1[frame_idx1], fixed_scale1, fixed_scale1, target_center=(0.5, 0.5))
                aligned_kp2 = align_keypoints(keypoints2[frame_idx2], scales2[frame_idx2], adjusted_scale2, target_center=(0.5, 0.5))

                blank_frame = np.zeros_like(frame1)
                overlay_frame = draw_overlay(blank_frame, aligned_kp1, aligned_kp2)

                overlay_frame = resize_and_pad(blank_frame, target_width, target_height)
                if vid1_width <= vid1_height:
                    padded_frame1 = crop_to_square(frame1)
                    padded_frame2 = adjust_frame_to_target(frame2, target_width, vid2_height)
                else:
                    padded_frame1 = adjust_frame_to_target(frame1, target_width, target_height)
                    padded_frame2 = adjust_frame_to_target(frame2, target_width, target_height)

                # combine videos based on video input orientation
                # TODO: auto use verticle or horizontal vid2 according to vid1 orientation

                combined_frame = np.vstack([padded_frame1, padded_frame2])

                out1.write(overlay_frame)
                out2.write(combined_frame)

        frame_idx1 += 1
        if not skip: # if frame2 is duplicate frame don't increrase idx
            frame_idx2 += 1

    cap1.release()
    cap2.release()
    out1.release()
    out2.release()

    print(f"overlay video saved to {overlay_path}")
    print(f"comparison video saved to {comaprison_path}")
    
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
        if keypoint is not None:
            try:
                # Key body points (only x and y coordinates are used for rotation along the y-axis)
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
                    "rw_z": rw_z,
                    "le_x": le_x,
                    "lh_x": lh_x,
                    "rh_y": rh_y,
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
    score = (orientation["body_rotation"] - orientation["left_hand_position"] + 
            orientation["elbow_angle_left"])
    #print(f"frame{i}: {score}")
    return score

# Pose 2 Criteria: minimize body rotation while right arm is on right side of body and left arm is not on right side of body
def forehand_end_criteria(orientation, i):
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
    arm_on_right_side = orientation["rw_x"] < orientation["ls_x"] and orientation["lw_x"] < orientation["ls_x"]
    arm_on_right_side_score = 0 if arm_on_right_side else float("inf")
    
    # Condition 2: Left wirst behind the right shoulder
    left_hand_right_of_right_shoulder = orientation["lw_z"] < orientation["rs_z"]
    left_hand_position_score = 0 if left_hand_right_of_right_shoulder else float("inf")
    
    # Condition 3: Body is rotated to the left (right shoulder is in front of the left shoulder)
    body_rotated_right = orientation["ls_z"] < orientation["rs_z"]
    body_rotation_score = 0 if body_rotated_right else float("inf")
    
    # Condition 5: Right foot is in front of the left foot (smaller y value for the left foot)
    left_foot_in_front = orientation["la_z"] < orientation["ra_z"]
    foot_position_score = 0 if left_foot_in_front else float("inf")
    
    # Condition 7: Right wrist is below the right shoulder
    right_wrist_below_shoulder = orientation["rw_y"] > orientation["rs_y"]
    wrist_below_shoulder_score = 0 if right_wrist_below_shoulder else float("inf")
    #print(orientation["elbow_angle_left"], orientation["elbow_angle_right"])
    
   
    # Total score (0 means all conditions met, inf means at least one condition failed)
    score = (
        arm_on_right_side_score +
        left_hand_position_score +
        body_rotation_score +
        foot_position_score + 
        wrist_below_shoulder_score 
    )
    
    if score < float("inf"):
        pass
        #print(orientation["elbow_angle_left"], orientation["elbow_angle_right"])
        #print(f"frame: {i}, wrist together: {wrist_together}")
    return score

# backhand end Criteria: minimize body rotation while right arm is on right side of body and left arm is not on right side of body
def backhand_end_criteria(orientation, i):
    # Ensure both hands are on the right side of the body
    both_hands_on_right_side = orientation["rw_x"] > orientation["rs_x"] and orientation["re_x"] > orientation["rs_x"]
    both_hands_on_right_side_score = 0 if both_hands_on_right_side else float("inf")  # 0 if arm is on the right side, inf if not

    # Combine all criteria (can be weighted if necessary)
    total_score = both_hands_on_right_side_score
    
    return total_score

# finds the best frame according to the pose criteria. 
# called by find_most_parallel_and_furthest_segment.   
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
def find_most_parallel_and_furthest_segment(keypoints1, keypoints2, swing_type):
    
    # Calculate orientations for both videos
    video1_orientations = calculate_body_orientation(keypoints1)
    video2_orientations = calculate_body_orientation(keypoints2)

    # Find best frames for Pose 1 in both videos
    #print("---------vid 1 pose 1---------")
    if swing_type == 'backhand':
        video1_pose1_frame = find_best_frame(video1_orientations, backhand_start_criteria)
        #print("---------vid 2 pose 1---------")
        video2_pose1_frame = find_best_frame(video2_orientations, backhand_start_criteria)
    else: # assumes only forehand and backhand for now 
        video1_pose1_frame = find_best_frame(video1_orientations, forehand_start_criteria)
        #print("---------vid 2 pose 1---------")
        video2_pose1_frame = find_best_frame(video2_orientations, forehand_start_criteria)
        
    # Pose 2 is only evaluated after Pose 1 is found, and it should start after Pose 1 frame
    if video1_pose1_frame is not None and video2_pose1_frame is not None:
        if swing_type == "backhand":
            #print("---------vid 1 pose 2---------")
            video1_pose2_frame = find_best_frame(video1_orientations, backhand_end_criteria, start_frame=video1_pose1_frame + 1)
            #print("---------vid 2 pose 2---------")
            video2_pose2_frame = find_best_frame(video2_orientations, backhand_end_criteria, start_frame=video2_pose1_frame + 1)
        else: # assumes only forehand and backhand for now 
            #print("---------vid 1 pose 2---------")
            video1_pose2_frame = find_best_frame(video1_orientations, forehand_end_criteria, start_frame=video1_pose1_frame + 1)
            #print("---------vid 2 pose 2---------")
            video2_pose2_frame = find_best_frame(video2_orientations, forehand_end_criteria, start_frame=video2_pose1_frame + 1)
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

# called by sync_video_timings
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

# called by sync_video_timings
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
    
    
 
def adjust_frame_to_target(frame, target_width, target_height):
    """
    Adjust a frame to match a target width and height by cropping if larger or padding if smaller.
    
    Parameters:
        frame (np.ndarray): The input frame (image).
        target_width (int): The target width.
        target_height (int): The target height.
    
    Returns:
        np.ndarray: The adjusted frame.
    """
    original_height, original_width = frame.shape[:2]

    # Crop if the frame is larger than the target
    if original_width > target_width:
        crop_x_start = (original_width - target_width) // 2
        crop_x_end = crop_x_start + target_width
        frame = frame[:, crop_x_start:crop_x_end]
    elif original_width < target_width:  # Pad if the frame is smaller than the target
        padding_left = (target_width - original_width) // 2
        padding_right = target_width - original_width - padding_left
        frame = cv2.copyMakeBorder(frame, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    if original_height > target_height:
        crop_y_start = (original_height - target_height) // 2
        crop_y_end = crop_y_start + target_height
        frame = frame[crop_y_start:crop_y_end, :]
    elif original_height < target_height:  # Pad if the frame is smaller than the target
        padding_top = (target_height - original_height) // 2
        padding_bottom = target_height - original_height - padding_top
        frame = cv2.copyMakeBorder(frame, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return frame


 # Function to extract keypoints and reference scale from a video
 # called by sync_video_timings
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
    return keypoints, scales   
    
    


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

