import os
import numpy as np
import cv2
import shutil
# from mediapipe import solutions as mp_solutions
# import mediapipe as mp
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
VisionRunningMode = mp.tasks.vision.RunningMode
# when local do `export CPU_ONLY=True``
use_torch = not (os.getenv("CPU_ONLY", "False").lower() == "true")

if use_torch:
    import torch
    gpu_available_torch = torch.cuda.is_available()
else:
    gpu_available_torch = False
print("torch gpu available: ", gpu_available_torch)
if gpu_available_torch:
    
    base_options = python.BaseOptions(model_asset_path='../models/pose_landmarker_full.task',  
                                  delegate=python.BaseOptions.Delegate.GPU
    )
else:
    base_options = python.BaseOptions(model_asset_path='../models/pose_landmarker_full.task')

PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

options = PoseLandmarkerOptions(
    base_options=base_options,
    # test the two options below again, but dropped performance on small dataset
    # min_pose_detection_confidence=0.9, 
    # min_tracking_confidence=0.8,
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False # also add detection and tracking confidence threshold
)

detector = vision.PoseLandmarker.create_from_options(options)

# mp_pose = mp_solutions.pose
# pose = mp_pose.Pose(
#     model_complexity=2, 
#     min_detection_confidence=0.9, 
#     min_tracking_confidence=0.8)

def extract_frames_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    i = 0
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break
        # i+=1
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
                
    cap.release()
    return np.array(frames), timestamps, fps

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
            landmarks = results.pose_landmarks.landmark # todo remove the last landmark
            # Collect keypoints (x, y, z coordinates)
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            keypoints.append(frame_keypoints)
        else:
            keypoints.append([(None, None, None)] * 33)  # Fill with None for missing landmarks
                
    cap.release()
    return np.array(keypoints), fps

def extract_keypointsv2_asimg(video_path):
    frames, timestamps, fps = extract_frames_video(video_path)
    keypoints = []
    for frame in frames:
        img=mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = detector.detect(img).pose_landmarks #.pose_world_landmark; needs further testing s # pose_landmarks are not in world coordinates
        # copied from extract_keypoints_from_video
        if result:
            landmarks = result[0]
            # Collect keypoints (x, y, z coordinates)
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            keypoints.append(frame_keypoints)
        else:
            keypoints.append([(None, None, None)] * 33)  # Fill with None for missing landmarks

    return np.array(keypoints), fps

def extract_keypointsv3_asimg(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img=mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(img).pose_landmarks #.pose_world_landmark; needs further testing s # pose_landmarks are not in world coordinates
        # copied from extract_keypoints_from_video
        if result:
            landmarks = result[0]
            # Collect keypoints (x, y, z coordinates)
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            keypoints.append(frame_keypoints)
        else:
            keypoints.append([(None, None, None)] * 33)  # Fill with None for missing landmarks

    return np.array(keypoints), fps


def extract_keypointsv4_asimg(video_path):
    frames, timestamps, fps = extract_frames_video(video_path)
    keypoints = []
    
    for frame, timestamp in zip(frames, timestamps):
        img=mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = detector.detect_for_video(img, int(timestamp)).pose_landmarks #.pose_world_landmark; needs further testing s # pose_landmarks are not in world coordinates
        # copied from extract_keypoints_from_video
        if result:
            landmarks = result[0]
            # Collect keypoints (x, y, z coordinates)
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            keypoints.append(frame_keypoints)
        else:
            keypoints.append([(None, None, None)] * 33)  # Fill with None for missing landmarks

    return np.array(keypoints), fps


def extract_keypointsv5_asimg(video_path):
    # fails
    frames, timestamps, fps = extract_frames_video(video_path)
    keypoints = []
    img=mp.Image(image_format=mp.ImageFormat.SRGB, data=frames)

    for _, timestamp in zip(frames, timestamps):
        result = detector.detect_for_video(img, int(timestamp)).pose_landmarks #.pose_world_landmark; needs further testing s # pose_landmarks are not in world coordinates
        breakpoint()
        # copied from extract_keypoints_from_video
        if result:
            landmarks = result[0]
            # Collect keypoints (x, y, z coordinates)
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            keypoints.append(frame_keypoints)
        else:
            keypoints.append([(None, None, None)] * 33)  # Fill with None for missing landmarks

    return np.array(keypoints), fps

def extract_keypointsv6_asimg(video_path):
    detector = vision.PoseLandmarker.create_from_options(options)  # have to reinitialize the detector due to detect_from_video being video specific and no ability to flush the buffer

    cap = cv2.VideoCapture(video_path)
    keypoints = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img=mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms < 0:
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            fps = cap.get(cv2.CAP_PROP_FPS)
            timestamp_ms = int((frame_number / fps) * 1000)

        result = None
        try:
            result = detector.detect_for_video(img, timestamp_ms).pose_world_landmarks # pose_landark; needs further testing s # pose_landmarks are not in world coordinates
        except:
            print(f"skipping frame for {video_path} and timestamp {timestamp_ms}")

        # copied from extract_keypoints_from_video
        if result:
            landmarks = result[0]
            # Collect keypoints (x, y, z coordinates)
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            keypoints.append(frame_keypoints)
        else:
            keypoints.append([(None, None, None)] * 33)  # Fill with None for missing landmarks

    return np.array(keypoints), fps