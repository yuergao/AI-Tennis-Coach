from segment import *
from sync import *

'''
-----------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------- Driver --------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''
# User inputs
# TODO: make into classes
swing_type = 'forehand'
extension = '.mp4'

player1 = 'audrey'
player1_vid_idx = 4
player1_swing_idx = 1

player2 = 'badosa'
player2_vid_idx = 1
player2_swing_idx = 1


# construct based on user inputs
player1_vid = f"{player1}{player1_vid_idx}"
player1_vid_swing = f"{player1_vid}_{swing_type}{player1_swing_idx}"

player2_vid = f"{player2}{player2_vid_idx}"
player2_vid_swing = f"{player2_vid}_{swing_type}{player2_swing_idx}"

video1_path = f"{player1}/{player1_vid}/{swing_type}/{player1_vid_swing}{extension}"
video2_path = f"{player2}/{player2_vid}/{swing_type}/{player2_vid_swing}{extension}"

pipeline1 = True
pipeline2 = False

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

if pipeline1:
    for video_path in video_paths:
        output_levels = video_path.split('/')
        output_folder = f"{output_levels[0]}/{output_levels[1]}"   
        output_path = f"{output_folder}/{swing_type}"
        
        clear_output_folder(output_path)
        
        #Extract keypoints from the video
        keypoints, fps = extract_keypoints_from_video(video_path)

        # Detect swing boundaries
        swing_frames = detect_swing_boundaries(keypoints, fps)

        # Segment swings based on the detected frames
        swing_segments = segment_swings(swing_frames)

        print(f"Detected {len(swing_segments)} swings.")

        # Process each swing and extract video segments
        for idx, (start_frame, end_frame) in enumerate(swing_frames):
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

    # Call the function to sync the timing and overlay the videos
    sync_video_timings(video1_path, video2_path, overlay_path, comparison_path)

