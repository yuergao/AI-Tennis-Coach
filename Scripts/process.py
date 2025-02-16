from segment import *
from sync import *
from video_utils import *
from calc_utils import *
import os
import av

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


# User inputs
pipeline1 = False # <<<<<<<<<<<<<<<<<<<<<<<
pipeline2 = True  # <<<<<<<<<<<<<<<<<<<<<<<
extension = '.mp4'

'''
-----------------------------------------------------------------------------------------------------------------------------------
---------------------- Pipeline Segment 1: segment original full video into separate swings----------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''
audrey_paths = [('audrey',12),('audrey',13),('audrey',14),('audrey',4),('audrey',5),('audrey',6),('audrey',8),('audrey',9),('audrey',10),('audrey',1)]
amature_paths = [('claire',1),('player',1),('player',2),('player',4),('player',6),('player',7),('player',8),('insta',1)]
pro_male_paths = [('sinner',1),('zverev',1),('alcaraz',1),('fritz',1),('medvedev',1),('ruud',1),('Djokovic',1),('Minaur',1)]
pro_female_paths = [('sabalenka',1),('swiatek',1),('gauff',1),('paolini',1),('zheng',1),('rybakina',1),('pegula',1),('badosa',1)]
more_amature_paths = [('arushi',2), ('qi',1), ('mark',1)]


test_paths = [('federer',1)] # <<<<<<<<<<<<<<<<<<<<<<<

if pipeline1:
    for player, player_vid_idx in test_paths:
        player_vid = f"{player}{player_vid_idx}"
        video_path = f"{player}/{player_vid}/{player_vid}{extension}"
        process_video_path = f"{player}/{player_vid}/{player_vid}_processed{extension}"
        output_levels = video_path.split('/')
        output_folder = f"{output_levels[0]}/{output_levels[1]}"   
        forehand_path = f"{output_folder}/forehand"
        backhand_path = f"{output_folder}/backhand"
        
        clear_output_folder(forehand_path)
        clear_output_folder(backhand_path)
        #extract_motion_segment(video_path, start_frame, end_frame, output_video)
        
        #Extract keypoints from the video
        keypoints, fps = extract_keypointsv6_asimg(video_path)
        print(f"Finished keypoint extraction for {player}{player_vid_idx}.")

        # Detect swing boundaries
        forehand_segments, backhand_segments = detect_swings(keypoints, fps)
        save_swings_to_json(forehand_segments + backhand_segments, f"{player}/{player_vid}/{player_vid}.json")
        print(f"Detected {len(forehand_segments)} forehands and {len(backhand_segments)} backhands.")

        # Process each swing and extract video segments
        for idx, forehand in enumerate(forehand_segments):
            start_frame = forehand.padded_start_frame
            end_frame = forehand.padded_end_frame
            if start_frame >= 0 and end_frame > start_frame:
                #find previous directory
                output_video = f"{forehand_path}/{output_levels[1]}_forehand{idx+1}{extension}"
                extract_motion_segment(video_path, start_frame, end_frame, output_video)

        # Process each swing and extract video segments
        for idx, backhand in enumerate(backhand_segments):
            start_frame = backhand.padded_start_frame
            end_frame = backhand.padded_end_frame
            if start_frame >= 0 and end_frame > start_frame:
                #find previous directory
                output_video = f"{backhand_path}/{output_levels[1]}_backhand{idx+1}{extension}"
                extract_motion_segment(video_path, start_frame, end_frame, output_video)
                
'''
-----------------------------------------------------------------------------------------------------------------------------------
---------------------- Pipeline Segment 2: output segmented swing comparison videos------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''
is_manual = False # <<<<<<<<<<<<<<<<<<<<<<<
swing_type = 'forehand' # <<<<<<<<<<<<<<<<<<<<<<<

player1 = 'audrey'  # <<<<<<<<<<<<<<<<<<<<<<<
player1_vid_idx = 9 # <<<<<<<<<<<<<<<<<<<<<<<
player1_swing_idx = 4

player2 = 'Badosa' # <<<<<<<<<<<<<<<<<<<<<<<
player2_vid_idx = 1 # <<<<<<<<<<<<<<<<<<<<<<<
player2_swing_idx = 7

player1_vid = f"{player1}{player1_vid_idx}"
player2_vid = f"{player2}{player2_vid_idx}"
   
if pipeline2:
    # output 
    loaded_swings1 = load_swings_from_json(f"{player1}/{player1_vid}/{player1_vid}.json")
    loaded_swings2 = load_swings_from_json(f"{player2}/{player2_vid}/{player2_vid}.json")
    
        
    if is_manual:
        player_compare = f"{player1}_{player2}"
        player_compare_vid = f"{player1_vid}_{player2_vid}"
        player_compare_vid_swing = f"{player_compare_vid}_{swing_type}{player1_swing_idx}_{player2_swing_idx}"
        overlay_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}/{player_compare_vid_swing}_overlay{extension}"
        comparison_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}/{player_compare_vid_swing}{extension}"
        player1_vid_swing = f"{player1_vid}_{swing_type}{player1_swing_idx}"
        player2_vid_swing = f"{player2_vid}_{swing_type}{player2_swing_idx}"
        video1_path = f"{player1}/{player1_vid}/{swing_type}/{player1_vid_swing}{extension}"
        video2_path = f"{player2}/{player2_vid}/{swing_type}/{player2_vid_swing}{extension}"
        # Call the function to sync the timing and overlay the videos
        sync_video_timings(video1_path, video2_path, overlay_path, comparison_path, swing_type)
    else:
        for swing_type in ("forehand", "backhand"):
            player_compare = f"{player1}_{player2}"
            player_compare_vid = f"{player1_vid}_{player2_vid}"
            folder1 = f"{player1}/{player1_vid}/{swing_type}"
            folder2 = f"{player2}/{player2_vid}/{swing_type}"
            
            videos1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.mp4'))]
            videos2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.mp4'))]
            folder_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}"
            vid2_swing_lens = []
            clear_output_folder(folder_path)
            
            for idx1, video1 in enumerate(videos1):
                if "_fps." in video1:
                    continue  

                container1 = av.open(video1)
                vid1_duration = container1.duration / av.time_base

                min_diff = float("inf")
                min_vid2 = None
                min_vid2_idx = -1
                best_swing2 = None
                swing1 = loaded_swings1[idx1]
                
                for idx2, video2 in enumerate(videos2):
                    if "_fps." in video2:
                        continue

                    container2 = av.open(video2)
                    vid2_duration = container2.duration / av.time_base
    
                    swing2 = loaded_swings2[idx2]
                   
                    swing_len_diff = abs(swing2.get_swing_len() - swing1.get_swing_len())
                    movement_diff = abs(swing2.movement - swing1.movement)

                    # Define a weighted criteria for comparison
                    total_diff = movement_diff  # Scale movement difference
                    if total_diff < min_diff:
                        min_diff = total_diff
                        min_vid2 = video2
                        min_vid2_idx = idx2
                        best_swing2 = swing2
                        
                loaded_swings1[idx1].print_info()
                best_swing2.print_info()
                if min_vid2:   
                    player_compare_vid_swing = f"{player_compare_vid}_{swing_type}{idx1+1}_{min_vid2_idx+1}"
                    overlay_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}/{player_compare_vid_swing}_overlay{extension}"
                    comparison_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}/{player_compare_vid_swing}{extension}"
                    sync_video_timings(video1, min_vid2, overlay_path, comparison_path, swing_type)