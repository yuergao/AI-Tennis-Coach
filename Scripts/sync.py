import os
import cv2
import numpy as np
from mediapipe import solutions as mp_solutions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

mp_pose = mp_solutions.pose
pose = mp_pose.Pose()
    
# Called by driver. compares 2 video swings. outputs overlay video and comparison video. 
def sync_video_timings(video1_path, video2_path, overlay_path, comaprison_path, swing_type):  
    #initialize video capture
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames1, total_frames2)
    
    # initialize swing stamps
    vid1_start_sync = fps1
    vid1_end_sync = total_frames1 - fps1
    vid2_start_sync =  fps2
    vid2_end_sync =  total_frames2 - fps2
    vid1_swing_len = vid1_end_sync - vid1_start_sync
    vid2_swing_len = vid2_end_sync - vid2_start_sync 
    
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
        video1_path, video2_path = video2_path, video1_path
        vid1_start_sync, vid2_start_sync = vid2_start_sync, vid1_start_sync
        vid1_end_sync, vid2_end_sync = vid2_end_sync, vid1_end_sync
        vid1_swing_len, vid2_swing_len = vid2_swing_len, vid1_swing_len
        fps1, fps2 = fps2, fps1
        cap1, cap2 = cap2, cap1
        print(f"vid1 start sync: {vid1_start_sync} | end sync: {vid1_end_sync} | swing len: {vid1_swing_len}")
        print(f"vid2 start sync: {vid2_start_sync} | end sync: {vid2_end_sync} | swing len: {vid2_swing_len}")
    
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
    first_valid_frame_idx = 1

    #scale_ratio = scales1[first_valid_frame_idx] / scales2[first_valid_frame_idx]
    frame_idx1 = first_valid_frame_idx
    frame_idx2 = first_valid_frame_idx 
    
    # initialize video writer
    #out1 = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*'mp4v'), min_fps, (target_width, target_height))   
    out2 = cv2.VideoWriter(comaprison_path, cv2.VideoWriter_fourcc(*'mp4v'), min_fps, (output_video_width, output_video_height))
         
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
        if frame_idx1 > total_frames1 or frame_idx2 > total_frames2:
            break
         
        # process valid frames, bulk of video processing    
        if frame_idx1 < total_frames1 and frame_idx2 < total_frames2:
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
    import cv2
    
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
    print(f"Frame Skip Interval: {skip_interval}")

    # Adjust sync frame indices to new FPS
    adjusted_vid_start_sync = int(vid_start_sync / skip_interval)
    adjusted_vid_end_sync = int(vid_end_sync / skip_interval)
    print(f"Adjusted Sync Frames: original ({vid_start_sync/original_fps, vid_end_sync/original_fps}) Adjusted ({adjusted_vid_start_sync/target_fps, adjusted_vid_end_sync/target_fps})")

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    if not out.isOpened():
        raise IOError(f"Cannot create output video file: {output_path}")

    # Frame processing logic
    frame_idx = 0
    next_frame_to_write = 0  # Accumulator for the next frame to write

    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break

        # Only write frames that align with the target FPS
        if frame_idx >= next_frame_to_write:
            out.write(frame)
            next_frame_to_write += skip_interval

        frame_idx += 1

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
    