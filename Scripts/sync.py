import os
import cv2
import numpy as np
from mediapipe import solutions as mp_solutions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

mp_pose = mp_solutions.pose
pose = mp_pose.Pose()
    
# Called by driver. compares 2 video swings. outputs overlay video and comparison video. 
def sync_video_timings(video1_path, video2_path, overlay_path, comaprison_path, swing_type, keypoints1, scales1, keypoints2, scales2, frame_sync_stamps1, frame_sync_stamps2, fps1, fps2):  
    # initialize swing stamps
    vid1_start_sync = frame_sync_stamps1[0]
    vid1_end_sync = frame_sync_stamps1[1]
    vid2_start_sync =  frame_sync_stamps2[0]
    vid2_end_sync =  frame_sync_stamps2[1]
    vid1_swing_len = vid1_end_sync - vid1_start_sync
    vid2_swing_len = vid2_end_sync - vid2_start_sync 
    print(frame_sync_stamps1, fps1, frame_sync_stamps2,fps2)
    #initialize video capture
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # match frame rates - sync videos to lower fps
    min_fps = min(fps1,fps2)
    if fps1 > fps2:
        paths  = video1_path.split('/')
        newpath = f'{paths[0]}/{paths[1]}/{swing_type}/{paths[1]}_{min_fps}_fps.mp4'
        #if os.path.exists(newpath) and os.path.isfile(newpath):
         #   print(f"dropped frame video exists at {min_fps}")
        #else:
        vid1_start_sync, vid1_end_sync = drop_frames_to_match_fps(video1_path,min_fps,newpath, vid1_start_sync, vid1_end_sync)
        vid1_swing_len = vid1_end_sync - vid1_start_sync
        video1_path = newpath
        cap1 = cv2.VideoCapture(video1_path)
        fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
        
    elif fps1 < fps2:
        paths  = video2_path.split('/')
        newpath = f'{paths[0]}/{paths[1]}/{swing_type}/{paths[1]}_{min_fps}_fps.mp4'
        #if os.path.exists(newpath) and os.path.isfile(newpath):
           # print(f"dropped frame video exists at {min_fps}")
        #else:
        vid2_start_sync, vid2_end_sync = drop_frames_to_match_fps(video2_path,min_fps,newpath, vid2_start_sync, vid2_end_sync)
        vid2_swing_len = vid2_end_sync - vid2_start_sync
        video2_path = newpath
        cap2 = cv2.VideoCapture(video2_path)
        fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    
    print(f"vid1 start sync: {vid1_start_sync} | end sync: {vid1_end_sync} | swing len: {vid1_swing_len}")
    print(f"vid2 start sync: {vid2_start_sync} | end sync: {vid2_end_sync} | swing len: {vid2_swing_len}")
    
    # switch videos around if vid1 has a shorter swing len than vid2
    if vid1_swing_len < vid2_swing_len:
        print("videos switched!!!!!")
        keypoints1, keypoints2 = keypoints2, keypoints1   
        scales1, scales2 = scales2, scales1
        video1_path, video2_path = video2_path, video1_path
        vid1_start_sync, vid2_start_sync = vid2_start_sync, vid1_start_sync
        vid1_end_sync, vid2_end_sync = vid2_end_sync, vid1_end_sync
        vid1_swing_len, vid2_swing_len = vid2_swing_len, vid1_swing_len
        fps1, fps2 = fps2, fps1
        cap1, cap2 = cap2, cap1
        print(f"vid1 start sync: {vid1_start_sync} | end sync: {vid1_end_sync} | swing len: {vid1_swing_len}")
        print(f"vid2 start sync: {vid2_start_sync} | end sync: {vid2_end_sync} | swing len: {vid2_swing_len}")
    
    time_offset = vid2_start_sync - vid1_start_sync 
    # create directories if don't already exist
    #os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
    os.makedirs(os.path.dirname(comaprison_path), exist_ok=True)
    

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open video files.")
        return

    # find output video ratio based on usesr video input orientation (vid1)
    vid1_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid1_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid2_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid2_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
    target_width = max(vid1_width, vid2_width)
    target_height = max(vid1_height, vid2_height)
    output_video_height = target_height * 2
    output_video_width = target_width 
    
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
    #out1 = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*'mp4v'), min_fps, (target_width, target_height))   
    out2 = cv2.VideoWriter(comaprison_path, cv2.VideoWriter_fourcc(*'mp4v'), min_fps, (output_video_width, output_video_height))
    
    # use body metrics to calculate time offset and sync player swings
    print(f"Time offset: vid2 - vid1 = {time_offset} frames")
    
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
    #if vid1_swing_len > vid2_swing_len: #video 1 has the longer swing, duplicate vid2 pro frames
    frame_diff = vid1_swing_len - vid2_swing_len
    if frame_diff != 0:
        interval = 1 # so interval is at least 1 not 0
        dup_frame_idx = max(frame_idx2, vid2_start_sync - min(fps1, fps2) // 6)
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

                #fixed_scale1 = scales1[frame_idx1]
                #adjusted_scale2 = scales2[frame_idx2] * scale_ratio
                #aligned_kp1 = align_keypoints(keypoints1[frame_idx1], fixed_scale1, fixed_scale1, target_center=(0.5, 0.5))
                #aligned_kp2 = align_keypoints(keypoints2[frame_idx2], scales2[frame_idx2], adjusted_scale2, target_center=(0.5, 0.5))

                blank_frame = np.zeros_like(frame1)
                #overlay_frame = draw_overlay(blank_frame, aligned_kp1, aligned_kp2)

                #overlay_frame = resize_and_pad(blank_frame, target_width, target_height)
                

                padded_frame1 = adjust_frame_to_target(frame1, target_width, target_height)
                padded_frame2 = adjust_frame_to_target(frame2, target_width, target_height)

                # combine videos based on video input orientation
                combined_frame = np.vstack([padded_frame1, padded_frame2])

                #out1.write(overlay_frame)
                out2.write(combined_frame)

        frame_idx1 += 1
        if not skip: # if frame2 is duplicate frame don't increrase idx
            frame_idx2 += 1

    cap1.release()
    cap2.release()
    #out1.release()
    out2.release()

    #print(f"overlay video saved to {overlay_path}")
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

# finds the best frame according to the pose criteria. 
# called by find_most_parallel_and_furthest_segment.   
def find_best_frame(orientations, criteria, start_frame=1, end_frame=-1):
    best_frame = None
    best_score = float("inf")

    for i, orientation in enumerate(orientations[start_frame:end_frame], start=start_frame):  # Loop to check pairs of consecutive frames
        orientation1 = orientations[i]
        orientation2 = orientations[i + 1]

        if orientation1:
            score1 = criteria(orientation1, i)
            score2 = criteria(orientation2, i + 1)
                
            # Check if both consecutive scores are less than the current best_score
            if score1 < best_score:
                best_score = score1
                best_frame = i 
    return best_frame  


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
    '''
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
    '''
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

def drop_frames_to_match_fps(video_path, target_fps, output_path, vid_start_sync, vid_end_sync):
    """
    Adjusts the frame rate of a video by dropping frames to match the target FPS, 
    and adjusts vid_start_sync and vid_end_sync to match the new FPS.

    Args:
        video_path (str): Path to the input video.
        target_fps (float): Target frames per second.
        output_path (str): Path to save the output video.
        vid_start_sync (int): Start frame index based on original FPS.
        vid_end_sync (int): End frame index based on original FPS.

    Returns:
        tuple: Adjusted vid_start_sync and vid_end_sync for the new FPS.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Get input video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original FPS: {original_fps}, Target FPS: {target_fps}, Total Frames: {total_frames}")

    # Validate FPS
    if target_fps <= 0:
        raise ValueError("Target FPS must be greater than 0.")
    if target_fps > original_fps:
        raise ValueError("Target FPS cannot be higher than the original FPS.")

    # Calculate frame skipping interval
    skip_interval = original_fps / target_fps

    # Adjust sync frame indices to new FPS
    adjusted_vid_start_sync = int(vid_start_sync / skip_interval)
    adjusted_vid_end_sync = int(vid_end_sync / skip_interval)
    print(f"Adjusted Sync Frames: Start = {adjusted_vid_start_sync}, End = {adjusted_vid_end_sync}")

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    if not out.isOpened():
        raise IOError(f"Cannot create output video file: {output_path}")

    frame_idx = 0
    skip_counter = 0  # Counter to control frame skipping

    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break

        # Write frames at appropriate intervals
        if skip_counter % int(skip_interval) == 0:
            out.write(frame)

        frame_idx += 1
        skip_counter += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Video processed successfully and saved to {output_path}")

    return adjusted_vid_start_sync, adjusted_vid_end_sync
    
 
def adjust_frame_to_target(frame, target_width, target_height):
    """
    Adjust a frame to match a target width and height by stretching it to fit while maintaining aspect ratio.
    Pad only if the aspect ratios differ.

    Parameters:
        frame (np.ndarray): The input frame (image).
        target_width (int): The target width.
        target_height (int): The target height.

    Returns:
        np.ndarray: The adjusted frame.
    """
    original_height, original_width = frame.shape[:2]
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Stretch the frame to match the target dimensions
    if original_aspect_ratio > target_aspect_ratio:
        # Original frame is wider: Match the height and stretch width
        new_width = target_width
        new_height = int(target_width / original_aspect_ratio)
    else:
        # Original frame is taller or matches aspect ratio: Match the width and stretch height
        new_height = target_height
        new_width = int(target_height * original_aspect_ratio)

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Check if padding is needed
    if new_width != target_width or new_height != target_height:
        padding_top = (target_height - new_height) // 2
        padding_bottom = target_height - new_height - padding_top
        padding_left = (target_width - new_width) // 2
        padding_right = target_width - new_width - padding_left

        # Add padding to center the frame
        adjusted_frame = cv2.copyMakeBorder(
            resized_frame,
            padding_top, padding_bottom, padding_left, padding_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return adjusted_frame

    return resized_frame


 # Function to extract keypoints and reference scale from a video
 # called by sync_video_timings
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

