import os
import numpy as np
from mediapipe import solutions as mp_solutions
import cv2
import shutil

mp_pose = mp_solutions.pose
pose = mp_pose.Pose()

# called by driver
# function to detect start and end indices of tennis swings based on wrist position.
def detect_swing_boundaries(keypoints):
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
    if not swing_frames or len(swing_frames) < 2:
        return []

    swings = [(swing_frames[i], swing_frames[i + 1]) for i in range(len(swing_frames) - 1)]
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



