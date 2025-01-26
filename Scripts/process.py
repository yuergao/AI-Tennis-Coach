from segment import *
from sync import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
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

player1_vid = f"{player1}{player1_vid_idx}"
player1_vid_swing = f"{player1_vid}_{swing_type}{player1_swing_idx}"
video1_path = f"{player1}/{player1_vid}/{swing_type}/{player1_vid_swing}{extension}"

player2 = 'sinner'
player2_vid_idx = 1
player2_swing_idx = 2

player2_vid = f"{player2}{player2_vid_idx}"
player2_vid_swing = f"{player2_vid}_{swing_type}{player2_swing_idx}"
video2_path = f"{player2}/{player2_vid}/{swing_type}/{player2_vid_swing}{extension}"

pipeline1 = False
pipeline2 = True
'''
-----------------------------------------------------------------------------------------------------------------------------------
---------------------- Pipeline Segment 1: segment original full video into separate swings----------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''

original_video1_path = f"{player1}/{player1_vid}/{player1_vid}{extension}"
original_video2_path = f"{player2}/{player2_vid}/{player2_vid}{extension}"

audrey_paths = [('sinner',1)]
test_paths = [('zverev',1),('alcaraz',1),('medvedev',1),('player',1),('player',2),('swiatek',1),('gauff',1),('badosa',1),('player',8),('insta',1),('player',6),('player',7)]

''' controls which video to be segmented'''
if pipeline1:
    for player, player_vid_idx in audrey_paths:
        player_vid = f"{player}{player_vid_idx}"
        original_video1_path = f"{player}/{player_vid}/{player_vid}{extension}"
        video_paths = [original_video1_path]

        for video_path in video_paths:
            output_levels = video_path.split('/')
            output_folder = f"{output_levels[0]}/{output_levels[1]}"   
            forehand_path = f"{output_folder}/forehand"
            backhand_path = f"{output_folder}/backhand"
            
            clear_output_folder(forehand_path)
            #clear_output_folder(backhand_path)
            
            #Extract keypoints from the video
            keypoints, fps = extract_keypoints_from_video(video_path)
            print(f"Finished keypoint extraction for {player}.")

            # Detect swing boundaries
            forehand_frames, backhand_frames = detect_swing_boundaries(keypoints, fps)

            # Segment swings based on the detected frames
            '''
            ----------forehand_segments and backhand_segments contains tupled frame index array------------
            '''
            forehand_segments = segment_swings(forehand_frames)
            backhand_segments = segment_swings(backhand_frames)

            print(f"Detected {len(forehand_segments)} forehands and {len(backhand_frames)} backhands.")

            # Process each swing and extract video segments
            for idx, (start_frame, end_frame) in enumerate(forehand_segments):
                if start_frame >= 0 and end_frame > start_frame:
                    #find previous directory
                    output_video = f"{forehand_path}/{output_levels[1]}_forehand{idx+1}{extension}"
                    extract_motion_segment(video_path, start_frame, end_frame, output_video)

            # Process each swing and extract video segments
            for idx, (start_frame, end_frame) in enumerate(backhand_segments):
                if start_frame >= 0 and end_frame > start_frame:
                    #find previous directory
                    output_video = f"{backhand_path}/{output_levels[1]}_backhand{idx+1}{extension}"
                    extract_motion_segment(video_path, start_frame, end_frame, output_video)
                    
'''
-----------------------------------------------------------------------------------------------------------------------------------
---------------------- Pipeline Segment 2: output segmented swing comparison videos------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''

if pipeline2:
    # output 
    player_compare = f"{player1}_{player2}"
    player_compare_vid = f"{player1_vid}_{player2_vid}"
    folder1 = f"{player1}/{player1_vid}/{swing_type}"
    folder2 = f"{player2}/{player2_vid}/{swing_type}"
    
    videos1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.mp4'))]
    videos2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.mp4'))]
    folder_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}"
    vid2_swing_lens = []
    clear_output_folder(folder_path)
    
    for idx2, video2 in enumerate(videos2):
        if "_fps." in video2:
            continue
        keypoints2, scales2, fps2 = extract_keypoints_and_scale(video2)
        frame_sync_stamps2 = find_most_parallel_and_furthest_segment(keypoints2, swing_type, fps2)
        
        if frame_sync_stamps2[0] is None or frame_sync_stamps2[1] is None:
            continue
        vid2_swing_len = frame_sync_stamps2[1] - frame_sync_stamps2[0]
        vid2_swing_lens.append((video2, vid2_swing_len, idx2, frame_sync_stamps2, keypoints2, scales2))
        
    for idx1, video1 in enumerate(videos1):
        if "_fps." in video1:
            continue
        keypoints1, scales1, fps1 = extract_keypoints_and_scale(video1)
        frame_sync_stamps1 = find_most_parallel_and_furthest_segment(keypoints1, swing_type, fps1)
        if frame_sync_stamps1[0] is None or frame_sync_stamps1[1] is None:
            continue
        vid1_swing_len = frame_sync_stamps1[1]- frame_sync_stamps1[0]
        min_swing_len_diff = float("inf")
        min_vid2 = videos2[0][:-4]
        min_vid2_idx = 0
        output_frame_sync_stamps2 = (0,0)
        output_keypoints2 = vid2_swing_lens[0][4]
        output_scales2 = vid2_swing_lens[0][-1]
        
        for video2,vid2_swing_len,idx2, frame_sync_stamps2, keypoints2, scales2 in vid2_swing_lens:
            swing_len_diff = abs(vid2_swing_len - vid1_swing_len)
            if swing_len_diff < min_swing_len_diff:
                min_swing_len_diff = swing_len_diff
                min_vid2 = video2
                min_vid2_idx = idx2
                output_frame_sync_stamps2 = frame_sync_stamps2
                output_keypoints2 = keypoints2
                output_scales2 = scales2
        player_compare_vid_swing = f"{player_compare_vid}_{swing_type}{idx1+1}_{min_vid2_idx+1}"
        overlay_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}/{player_compare_vid_swing}_overlay{extension}"
        comparison_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}/{player_compare_vid_swing}{extension}"
        sync_video_timings(video1, min_vid2, overlay_path, comparison_path, swing_type, keypoints1, scales1, output_keypoints2, output_scales2, frame_sync_stamps1, output_frame_sync_stamps2, fps1, fps2)
    
    # Call the function to sync the timing and overlay the videos
    #sync_video_timings(video1_path, video2_path, overlay_path, comparison_path, swing_type)

