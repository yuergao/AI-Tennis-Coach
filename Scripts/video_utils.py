import os
import numpy as np
import cv2
import shutil
import mediapipe as mp
import subprocess
import struct
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.framework.formats import image_format_pb2
from mediapipe.python.solution_base import SolutionBase

VisionRunningMode = mp.tasks.vision.RunningMode
# when local do `export CPU_ONLY=True``
use_torch = not (os.getenv("CPU_ONLY", "False").lower() == "False")

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
    base_options = python.BaseOptions(model_asset_path='./models/pose_landmarker_full.task')

PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

options = PoseLandmarkerOptions(
    base_options=base_options,
    # test the two options below again, but dropped performance on small dataset
    min_pose_detection_confidence=0.8, 
    min_tracking_confidence=0.7,
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False # also add detection and tracking confidence threshold
)

landmarker = PoseLandmarker.create_from_options(options)


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

def add_logo(frame, logo, logo_scale=0.2):
    """
    Adds a logo to the bottom-right corner of a frame, resizing it based on a percentage of the frame width 
    while maintaining the original aspect ratio.

    Parameters:
    - frame (numpy array): The frame where the logo will be added.
    - logo (numpy array): The loaded logo image (should include alpha channel if transparency is needed).
    - logo_scale (float): The percentage of the frame width the logo should take up.

    Returns:
    - frame (numpy array): The frame with the logo added.
    """

    if logo is None or frame is None:
        print("Error: Invalid frame or logo.")
        return frame

    # Get frame and logo dimensions
    frame_height, frame_width, _ = frame.shape
    logo_original_height, logo_original_width = logo.shape[:2]

    # Scale logo width based on frame width
    logo_width = int(frame_width * logo_scale)
    aspect_ratio = logo_original_width / logo_original_height  # Maintain aspect ratio
    logo_height = int(logo_width / aspect_ratio)

    # Resize logo while keeping aspect ratio
    logo = cv2.resize(logo, (logo_width, logo_height))

    # Define logo placement at the bottom-right corner (no gap)
    x_offset = frame_width - logo_width
    y_offset = frame_height - logo_height

    # Check if the logo has an alpha channel (for transparency)
    if logo.shape[2] == 4:
        alpha = logo[:, :, 3] / 255.0  # Normalize alpha values
        for c in range(3):  # Blend the RGB channels with the frame
            frame[y_offset:y_offset+logo_height, x_offset:x_offset+logo_width, c] = (
                (1 - alpha) * frame[y_offset:y_offset+logo_height, x_offset:x_offset+logo_width, c] +
                alpha * logo[:, :, c]
            )
    else:  # No transparency, just overlay the logo
        frame[y_offset:y_offset+logo_height, x_offset:x_offset+logo_width] = logo

    return frame

def extract_all_motion_segment(video_path, output_path, output_stick_path, draw_pose=True, stickfigure=True):
    """Extract the entire video with optional pose detection lines."""
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
    
    start_frame = 0
    end_frame = total_frames - 1

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    out2 = cv2.VideoWriter(output_stick_path, fourcc, fps, (width, height*2))

    # Start from the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    print(f"Extracting entire video: {start_frame} to {end_frame}. {round((end_frame - start_frame) / fps, 1)} seconds long")

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
        blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not ret:
            print(f"Frame {frame_idx} could not be read, ending extraction.")
            break
        
        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)    
            
        if stickfigure:
            mp_drawing.draw_landmarks(
                    blank_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),  # Blue landmarks
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1),  # Red lines
                )
            
            stacked_frame = np.vstack([frame, blank_frame])
            out2.write(stacked_frame)
            
        if draw_pose:
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
    out2.release()
    if draw_pose:
        pose.close()
    print(f"Video saved to {output_path}{' with pose annotations' if draw_pose else ''}")
    
    
# called by driver in pipeline 1  
def extract_motion_segment(video_path, swing, output_path, output_stick_path, draw_pose=True, stickfigure = True):
    """Extract a segment of the video based on frame indices with optional pose detection lines."""
    # Ensure the output directory exists
    start_frame = swing.padded_start_frame # <<<<<<<
    end_frame = swing.padded_end_frame # <<<<<<<
    
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
    out2 = cv2.VideoWriter(output_stick_path, fourcc, fps, (width, height*2))
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

    # Load the logo image
    logo = cv2.imread("logo/logo.png", cv2.IMREAD_UNCHANGED)
    
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not ret:
            print(f"Frame {frame_idx} could not be read, ending extraction.")
            break
        
        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)    
            
        if stickfigure:
            
            mp_drawing.draw_landmarks(
                    blank_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),  # Blue landmarks
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1),  # Red lines
                )
            
            
            # Define multiple lines of text
            lines = [
                f"{swing.type}",
                f'{swing.stance} stance',
                #f'lateral {swing.x_movement:.1f}',
                #f'depth {swing.z_movement:.1f}',
                #f'swing length {swing.get_swing_len():.1f} seconds'
                #f"{'right' if swing.x_movement < 0 else 'left'} ({abs(swing.x_movement):.1f})",
                #f"{'backward' if swing.x_movement < 0 else 'forward'} ({abs(swing.z_movement):.1f})"
            ]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = height / 720  # Scale text size based on 720p reference
            font_thickness = int(font_scale * 2)  # Adjust thickness based on scale
            text_color = (255, 255, 255)  # White text

            # Get text height to determine spacing
            _, text_height = cv2.getTextSize("A", font, font_scale, font_thickness)[0]

            # Define starting position for the first line (top-right)
            x_position = width - 10  # Right-aligned, adjust as needed
            y_position = 30  # Starting height from the top

            # Padding values (adjust as needed)
            padding_x = 20  # Right padding
            padding_y = int(height * 0.05)  # Top padding (5% of frame height)

            # Adjust starting position for text with top padding
            y_position = padding_y  

            # Add each line to the frame
            for line in lines:
                (text_width, _) = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                x = x_position - text_width - padding_x  # Apply right padding
                cv2.putText(blank_frame, line, (x, y_position), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                y_position += text_height + int(10 * (height / 1080))  # Adjust spacing dynamically

            stacked_frame = np.vstack([frame, blank_frame])
            stacked_frame = add_logo(stacked_frame, logo)
            out2.write(stacked_frame)
            
        if draw_pose:
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
    out2.release
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

def new_extract_motion_segment(video_path, start_frame, end_frame, output_path, draw_pose=False):
    """Extract a segment of the video based on frame indices with optional pose detection lines using the new API."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists

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

    # Set up video writer with the same dimensions and FPS as input
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Jump to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    last_timestamp = 0  # Keep track of last timestamp to ensure increasing order

    print(f"Extracting video segment: {start_frame} to {end_frame}. {round((end_frame - start_frame) / fps, 1)} seconds long")

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_idx} could not be read, ending extraction.")
            break

        if draw_pose:
            # Convert frame to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Generate a strictly increasing timestamp
            timestamp_us = max(last_timestamp + 1, int((frame_idx / fps) * 1_000_000))  # Ensure monotonic increase
            last_timestamp = timestamp_us  # Update last timestamp

            # Process frame using the new PoseLandmarker API
            results = landmarker.detect_for_video(mp_frame, timestamp_us)

            # Draw pose landmarks if detected
            if results.pose_landmarks:
                for person_landmarks in results.pose_landmarks:
                    for landmark in person_landmarks:
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green keypoints

        # Write frame (annotated or original) to output video
        out.write(frame)

        # Ensure frame index always increases
        frame_idx += 1  

    cap.release()
    out.release()
    print(f"Segment saved to {output_path}{' with pose annotations' if draw_pose else ''}")


def extract_keypointsv6_asimg(video_path, right_handed):
    detector = vision.PoseLandmarker.create_from_options(options)
    convert_video_to_mp4v(video_path)
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip video horizontally if left-handed
        if not right_handed:
            frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img=mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect_for_video(img, int(cap.get(cv2.CAP_PROP_POS_MSEC))).pose_landmarks #.pose_world_landmark; needs further testing s # pose_landmarks are not in world coordinates
        # copied from extract_keypoints_from_video
        if result:
            landmarks = result[0]
            # Collect keypoints (x, y, z coordinates)
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            keypoints.append(frame_keypoints)
        else:
            keypoints.append([(None, None, None)] * 33)  # Fill with None for missing landmarks
    cap.release()
    return np.array(keypoints), fps



def convert_video_to_mp4v(input_path):
    """Convert a video to MP4 (MPEG-4 `mp4v`) format using FFmpeg, ensuring overwrite works."""
    codec = get_video_codec(input_path)
    if not codec:
        print('Incorrect video Name')
        return 
    
    if 'mp' in codec:
        return
    
    temp_output_path = input_path.replace(".mp4", "_converted.mp4")

    ffmpeg_cmd = [
        "ffmpeg", "-y",  
        "-i", input_path,  
        "-c:v", "mpeg4", "-q:v", "5",  
        "-c:a", "aac", "-b:a", "128k",  
        temp_output_path  
    ]

    print(f"Converting '{input_path}' to MP4 format...")
    
    try:
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(f"Conversion complete: {temp_output_path}")

        # Ensure FFmpeg has finished writing before replacing the file
        if wait_for_file_ready(temp_output_path) and wait_until_opencv_can_open(temp_output_path):
            os.remove(input_path)  
            os.rename(temp_output_path, input_path)  
            print(f"Replaced original file with converted version: {input_path}")
            return True
        else:
            print(f"Error: Converted file '{temp_output_path}' may be incomplete.")
            return False

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr}")
        return False
    
    
def wait_for_file_ready(filepath, timeout=3, check_interval=0.5):
    """Wait until the file is fully written, flushed, and can be accessed."""
    prev_size = -1
    elapsed_time = 0

    while elapsed_time < timeout:
        if os.path.exists(filepath):
            current_size = os.path.getsize(filepath)

            # If size is stable, try opening it in read mode
            if current_size == prev_size and current_size > 0:
                try:
                    with open(filepath, "rb") as f:
                        os.fsync(f.fileno())  # Force OS to write buffered data to disk
                    print(f"File '{filepath}' is fully written and flushed.")
                    return True  # File is fully written and ready
                except IOError:
                    print(f"File '{filepath}' exists but is not accessible yet.")
            
            prev_size = current_size
        
        time.sleep(check_interval)
        elapsed_time += check_interval

    print(f"Warning: File '{filepath}' did not become ready within {timeout} seconds.")
    return False


def wait_until_opencv_can_open(video_path, timeout=3, check_interval=0.5):
    """Wait until OpenCV can successfully open the video."""
    elapsed_time = 0

    while elapsed_time < timeout:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            cap.release()
            print(f"OpenCV can now read '{video_path}'. Ready for processing.")
            return True  # Video is fully written and accessible
        cap.release()

        time.sleep(check_interval)
        elapsed_time += check_interval

    print(f"Warning: OpenCV could not open '{video_path}' within {timeout} seconds.")
    return False

def get_video_codec(video_path):
    """Detects the video codec using FFmpeg."""
    ffmpeg_cmd = [
        "ffmpeg", "-i", video_path
    ]

    try:
        # Capture FFmpeg stderr output (where codec info is printed)
        result = subprocess.run(ffmpeg_cmd, stderr=subprocess.PIPE, text=True, check=False)
        
        # Find codec info in the output
        for line in result.stderr.split("\n"):
            if "Video:" in line:
                codec_info = line.split("Video:")[1].split(",")[0].strip()
                print(f"Detected Video Codec: {codec_info}")
                return codec_info
        
        print("Could not detect codec.")
        return None
    except Exception as e:
        print(f"Error detecting codec: {e}")
        return None