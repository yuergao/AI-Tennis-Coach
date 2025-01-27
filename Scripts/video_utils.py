import os
import numpy as np
import cv2
import shutil
from mediapipe import solutions as mp_solutions
import mediapipe as mp

mp_pose = mp_solutions.pose
pose = mp_pose.Pose(
    model_complexity=2, 
    min_detection_confidence=0.9, 
    min_tracking_confidence=0.8)

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
        pose = mp_pose.Pose(
                model_complexity=1, 
                min_detection_confidence=0.9, 
                min_tracking_confidence=0.8)
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
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),  # Blue landmarks
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1),  # Red lines
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

# Called by driver
def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
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
            # Collect keypoints (x, y, z coordinates)
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            keypoints.append(frame_keypoints)
        else:
            keypoints.append([(None, None, None)] * 33)  # Fill with None for missing landmarks
                
    cap.release()
    return np.array(keypoints), fps
