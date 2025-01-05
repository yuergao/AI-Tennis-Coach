import os
import numpy as np
from mediapipe import solutions as mp_solutions
import cv2
import shutil
from sync import(calculate_body_orientation)

mp_pose = mp_solutions.pose
pose = mp_pose.Pose()

def find_matching_frames(orientations, criteria, fps, threshold=float("inf")):
    """
        Find all frames that meet the criteria within a specified range, ensuring no
        frames are selected within a cooldown period after a match.

        Parameters:
        - orientations: List of pose orientations for each frame.
        - criteria: Function to compute the score for a given orientation.
        - threshold: Maximum acceptable score to include the frame (default: infinity).
        - start_frame: Index to start evaluating frames.
        - end_frame: Index to stop evaluating frames.
        - fps: Frames per second of the video.
        - cooldown_seconds: Cooldown time in seconds after a match.

        Returns:
        - matching_frames: List of indices of frames that meet the criteria.
        - scores: List of scores for each matching frame.
        """
    matching_frames = []
    cooldown_seconds = 2
    cooldown_frames = int(cooldown_seconds * fps)  # Number of frames to skip during cooldown
    last_selected_frame = -cooldown_frames  # Initialize to a value before the start

    for i, orientation in enumerate(orientations):
        # Skip if within cooldown
        if i < last_selected_frame + cooldown_frames:
            continue

        if orientation:
            score = criteria(orientation, i)
            if score < threshold:  
                matching_frames.append((i, i+(3 *fps)))
                last_selected_frame = i  # Update the last selected frame
                print(i, score)
    return matching_frames

def pose1_criteria(orientation, i):
    """
    Pose 1 Criteria:
    - Both hands are on the right side of the body.
    - Body is rotated to the right side (torso facing right).
    - Left hand is to the right of the right shoulder.
    - Left foot is in front of the right foot.
    
    Parameters:
    - orientation: A dictionary containing body landmarks' x and y coordinates.
    - i: Current frame index.
    
    Returns:
    - A score where 0 means all conditions are met, and inf means a condition failed.
    """
    
    # Condition 1: Both hands on the right side of the body
    arm_on_right_side = orientation["rw_x"] > orientation["rs_x"] and orientation["re_x"] > orientation["rs_x"]
    arm_on_right_side_score = 0 if arm_on_right_side else float("inf")
    
    # Condition 2: Left hand is to the right of the right shoulder
    left_hand_right_of_right_shoulder = orientation["lw_x"] > orientation["rs_x"]
    left_hand_position_score = 0 if left_hand_right_of_right_shoulder else float("inf")
    
    
    # Condition 2: Left hand is to the right of the right shoulder
    left_hand_right_of_right_shoulder = orientation["lw_y"] > orientation["rs_y"]
    left_hand_position_score1 = 0 if left_hand_right_of_right_shoulder else float("inf")
    
    # Condition 3: Body is rotated to the right (left shoulder is farther left than the right shoulder)
    body_rotated_right = orientation["ls_y"] < orientation["rs_y"]
    body_rotation_score = 0 if body_rotated_right else float("inf")
    
    # Condition 5: Left foot is in front of the right foot (smaller y value for the left foot)
    left_foot_in_front = orientation["la_y"] < orientation["ra_y"]
    foot_position_score = 0 if left_foot_in_front else float("inf")
    
    # Condition 7: Right wrist is below the right shoulder
    right_wrist_below_shoulder = orientation["rw_y"] > orientation["rs_y"]
    wrist_below_shoulder_score = 0 if right_wrist_below_shoulder else float("inf")
    

    # Total score (0 means all conditions met, inf means at least one condition failed)
    score = (
        arm_on_right_side_score +
        left_hand_position_score +
        body_rotation_score +
        foot_position_score +
        wrist_below_shoulder_score +
        left_hand_position_score1
    )
    return score


# called by driver
# function to detect start and end indices of tennis swings based on wrist position.
def detect_swing_boundaries(keypoints, fps):
    hip_threshold=0.08
    prep_ignore_interval = 75
    wrist_in_front_frames = []  # List to keep track of frames when wrists are in front
    feet_threshold = 0.08

    orientations = calculate_body_orientation(keypoints)
    video1_pose1_frame = find_matching_frames(orientations, pose1_criteria, fps)
    # Look for key aspects of tennis swing based on wrist position
    '''
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
                print(f"{i/fps:.1f} seconds & frame{i}: {left_wrist_distance:.4f},{right_wrist_distance:.4f}/{hip_threshold} hip threshold")
                print(f"{i/fps:.1f} seconds & frame{i}: {feet_distance} feet distance\n")
            
        # Debugging: Print comparison for wrists together
        #print(f"{i/30:.1f} seconds & frame{i}: {left_wrist_distance:.4f},{right_wrist_distance:.4f}/{hip_threshold} hip threshold")
        #print(f"{i/30:.1f} seconds & frame{i}: {feet_distance} feet distance\n")
    '''
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

    return video1_pose1_frame


"""
Segments the keypoints into swings based on detected swing frames.

Args:
    keypoints (list): List of keypoints for each frame.
    swing_frames (list): List of frame indices marking swing boundaries.
    
Returns:
    list: List of tuples representing start and end frame indices for swings.
"""
# called by driver in pipeline 1
# Segments the keypoints into swings based on detected swing frames.
def segment_swings(swing_frames):
    if not swing_frames or len(swing_frames) == 0:
        return []

    swings = [(swing_frames[i][0], swing_frames[i][1]) for i in range(len(swing_frames))]
    return swings


# called by driver in pipeline 1
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
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if start_frame < 0 or end_frame >= total_frames or start_frame >= end_frame:
        if end_frame >= total_frames:
            end_frame = total_frames - 1
        else:    
            print(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}, total_frames={total_frames}")
            return

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Jump to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    print(f"Extracting video segment: {start_frame} to {end_frame}. {round((end_frame-start_frame)/fps, 1)} seconds long")
    
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


# function to clear folder 
# called by driver
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



# called by driver
def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
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
    return np.array(keypoints), fps



