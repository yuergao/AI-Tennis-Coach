import os
import numpy as np
from mediapipe import solutions as mp_solutions
import cv2
import shutil
from sync import (calculate_angle)
import mediapipe as mp

mp_pose = mp_solutions.pose
pose = mp_pose.Pose()
# Function to calculate body orientation and distances for each frame
# called by find_most_parallel_and_furthest_segment
def calculate_body_orientation(keypoints):
    orientations = []
    for keypoint in keypoints:
        
        
        if keypoint[0][0] is not None or  keypoint[10][2] is not None or keypoint[23][0] is not None:
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
                    "wrist_distance": wrist_distance,
                    "elbow_angle_left": elbow_angle_left,  # Angle of the left arm at the elbow
                    "elbow_angle_right": elbow_angle_right  # Angle of the right arm at the elbow
                })
            except Exception as e:
                orientations.append(None)
        else:
            orientations.append(None)
    return orientations


def find_matching_frames_forehand(orientations, fps, threshold=float("inf")):
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
            score = pose1_criteria(orientation, i)
            if score < threshold:  
                matching_frames.append((max(0,i - fps), i+(2 *fps)))
                last_selected_frame = i  # Update the last selected frame
                #print(f"Frame: {i}, Score: {score}")
    return matching_frames

def find_matching_frames_backhand(orientations,fps, threshold=float("inf")):
    """
        Find all frames that meet the criteria within a specified range, ensuring no
        frames are selected within a cooldown period after a match.

        Parameters:
        - orientations: List of pose orientations for each frame.
        - criteria: Function to compute the score for a given orientation.
        - threshold: Maximum acceptable score to include the frame (default: infinity).
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
            if pose2_criteria(orientation, i) < threshold:  
                # Calculate scores for the current frame and the next fps/2 frames
                scores_window = []
                for j in range(max(0, i - int(fps / 2)), i): 
                    if orientations[j]:  # Ensure the frame has valid orientation data
                        shoulder_hip_distance = abs(orientations[j]["rs_y"] - orientations[j]["rh_y"])
                        wrist_distance = abs(orientations[j]["lw_y"] - orientations[j]["rw_y"]) + abs(orientations[j]["lw_x"] - orientations[j]["rw_x"])
                        wrist_distance = wrist_distance / shoulder_hip_distance
                        score = wrist_distance
                        scores_window.append(score)

                # Calculate the average score for this window of frames
                if scores_window:
                    avg_score = np.mean(scores_window)
                    if avg_score < 0.3:  
                        print(f"frame: {avg_score}")
                        matching_frames.append((max(0,i - fps), i+(2 *fps)))
                        last_selected_frame = i  # Update the last selected frame
                        #print(f"Frame: {i}, Average Score: {avg_score}")
    
    return matching_frames

def pose1_criteria(orientation, i):
    """
    Pose 1 Criteria
    
    Parameters:
    - orientation: A dictionary containing body landmarks' x and y coordinates.
    - i: Current frame index.
    
    Returns:
    - A score where 0 means all conditions are met, and inf means a condition failed.
    """
    
    # Condition 1: Right hand is to the right of the right shoulder
    arm_on_right_side = orientation["rw_x"] > orientation["rs_x"] and orientation["le_x"] > orientation["rs_x"] and orientation["lw_x"] > orientation["rs_x"]
    arm_on_right_side_score = 0 if arm_on_right_side else float("inf")
    
    # Condition 2: Left wirst is in front of the right shoulder
    left_hand_right_of_right_shoulder = orientation["lw_z"] > orientation["rs_z"]
    left_hand_position_score = 0 if left_hand_right_of_right_shoulder else float("inf")
    
    # Condition 3: Body is rotated to the right (left shoulder is in front of the right shoulder)
    body_rotated_right = orientation["ls_z"] > orientation["rs_z"]
    body_rotation_score = 0 if body_rotated_right else float("inf")
    
    # Condition 5: Left foot is in front of the right foot (smaller y value for the left foot)
    left_foot_in_front = orientation["la_z"] > orientation["ra_z"]
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
        wrist_below_shoulder_score 
    )
    
    return score

def pose2_criteria(orientation, i):
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
    
    left_elbow_angle = orientation["elbow_angle_left"] > 130 and orientation["elbow_angle_right"] > 130
    left_elbow_angle_score = orientation["elbow_angle_left"] if left_elbow_angle else float("inf")
    #print(orientation["elbow_angle_left"], orientation["elbow_angle_right"])
    
    wrist_together = orientation["wrist_distance"]
    wrist_together_score = orientation["wrist_distance"] if orientation["wrist_distance"] < 0.4 else float("inf")
    # Total score (0 means all conditions met, inf means at least one condition failed)
    score = (
        arm_on_right_side_score +
        left_hand_position_score +
        body_rotation_score +
        foot_position_score + 
        wrist_below_shoulder_score + 
        left_elbow_angle_score + 
        wrist_together_score
    )
    
    if score < float("inf"):
        pass
        #print(orientation["elbow_angle_left"], orientation["elbow_angle_right"])
        #print(f"frame: {i}, wrist together: {wrist_together}")
    return score

# called by driver
# function to detect start and end indices of tennis swings based on wrist position.
def detect_swing_boundaries(keypoints, fps):
    hip_threshold=0.08
    prep_ignore_interval = 75
    wrist_in_front_frames = []  # List to keep track of frames when wrists are in front
    feet_threshold = 0.08

    orientations = calculate_body_orientation(keypoints)
    forehand_frames = find_matching_frames_forehand(orientations,fps)
    backhand_frames = find_matching_frames_backhand(orientations,fps)
    
    # Look for key aspects of tennis swing based on wrist position
    return [forehand_frames, backhand_frames]


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
def extract_motion_segment(video_path, start_frame, end_frame, output_path, draw_pose=True):
    """Extract a segment of the video based on frame indices with optional pose detection lines."""
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
    print(f"Extracting video segment: {start_frame} to {end_frame}. {round((end_frame - start_frame) / fps, 1)} seconds long")

    # Initialize MediaPipe Pose if pose drawing is enabled
    if draw_pose:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        mp_drawing = mp.solutions.drawing_utils

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_idx} could not be read, ending extraction.")
            break

        if draw_pose:
            # Convert the frame to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Draw pose landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=3),  # Blue landmarks
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=3),  # Red lines
                )

        # Write the frame (annotated or original) to the output video
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    if draw_pose:
        pose.close()
    print(f"Segment saved to {output_path}{' with pose annotations' if draw_pose else ''}")


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
            else:
                keypoints.append([(None, None, None)] * 33)
                
    
    cap.release()
    return np.array(keypoints), fps



