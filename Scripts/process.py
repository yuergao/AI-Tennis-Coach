from segment import *
from sync import *
from video_utils import *
from calc_utils import *
import os
import av

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


# User inputs
pipeline1 = False # <<<<<<<<<<<<<<<<<<<<<<<
pipeline2 = True # <<<<<<<<<<<<<<<<<<<<<<<

test_paths = [('mark',1, False)] # <<<<<<<<<<<<<<<<<<<<<<<
player2_paths = [('alcaraz',2, True),('federer',2, False)] # <<<<<<<<<<< automate <<<<<<<<<<<<

service = ["forehand", "backhand"] # <<<<<<<<< service
extension = '.mp4'

'''
-----------------------------------------------------------------------------------------------------------------------------------
---------------------- Pipeline Segment 1: segment original full video into separate swings----------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''
audrey_paths = [('audrey',12),('audrey',13),('audrey',14),('audrey',4),('audrey',5),('audrey',6),('audrey',7),('audrey',9),('audrey',10),('audrey',1)]
player_paths = [('player',5),('player',6),('player',4),('player',3),('player',9),('player',11),('player',12),('player',2),('player',1)]
pro_male_paths = [('sinner',1),('zverev',1),('alcaraz',1),('fritz',1),('medvedev',1),('ruud',1),('Djokovic',1),('Minaur',1)]
pro_female_paths = [('sabalenka',1),('swiatek',1),('gauff',1),('paolini',1),('zheng',1),('rybakina',1),('pegula',1),('badosa',1)]
amature_paths = [('arushi',2), ('qi',1), ('mark',1)]

if pipeline1:
    for player, player_vid_idx, double_hand_backhand in test_paths:
        player_vid = f"{player}{player_vid_idx}"
        video_path = f"{player}/{player_vid}/{player_vid}{extension}"
        process_video_path = f"{player}/{player_vid}/{player_vid}_processed{extension}"
        output_levels = video_path.split('/')
        output_folder = f"{output_levels[0]}/{output_levels[1]}"   
        forehand_path = f"{output_folder}/forehand"
        backhand_path = f"{output_folder}/backhand"
        
        #extract_motion_segment(video_path, start_frame, end_frame, output_video)
        
        #Extract keypoints from the video
        keypoints, fps = extract_keypointsv6_asimg(video_path)
        print(f"Finished keypoint extraction for {player}{player_vid_idx}.")

        # Detect swing boundaries
        forehand_segments, backhand_segments = detect_swings(keypoints, fps, double_hand_backhand)
        print(f"Detected {len(forehand_segments)} forehands and {len(backhand_segments)} backhands.")
        
        clear_output_folder(forehand_path)
        clear_output_folder(backhand_path)
        
        # Process each swing and extract video segments
        for idx, forehand in enumerate(forehand_segments):
            #find previous directory
            output_video = f"{forehand_path}/{output_levels[1]}_forehand{idx+1}{extension}"
            output_stick_video = f"{forehand_path}/{output_levels[1]}_forehand{idx+1}stacked{extension}"
            forehand_segments[idx].name = f"{output_levels[1]}_forehand{idx+1}{extension}"
            extract_motion_segment(video_path, forehand, output_video, output_stick_video)
                

        # Process each swing and extract video segments
        for idx, backhand in enumerate(backhand_segments):
            #find previous directory
            output_video = f"{backhand_path}/{output_levels[1]}_backhand{idx+1}{extension}"
            output_stick_video = f"{backhand_path}/{output_levels[1]}_backhand{idx+1}stacked{extension}"
            backhand_segments[idx].name = f"{output_levels[1]}_backhand{idx+1}{extension}"
            extract_motion_segment(video_path, backhand, output_video, output_stick_video)
                
        save_swings_to_json(forehand_segments, f"{player}/{player_vid}/{player_vid}_forehand.json")
        save_swings_to_json(backhand_segments, f"{player}/{player_vid}/{player_vid}_backhand.json")
                
'''
-----------------------------------------------------------------------------------------------------------------------------------
---------------------- Pipeline Segment 2: output segmented swing comparison videos------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''
is_manual = False # <<< manual 
swing_type = '' # <<< manual 

N = 2 # <<<<<<< number of similar videos
'''
player1 = 'mark'  # <<< one 
player1_vid_idx = 1 # <<< one 
player1_swing_idx = 4

player2 = 'federer' # <<<< one 
player2_vid_idx = 2 # <<<< one 
player2_swing_idx = 7
'''


if pipeline2:
    # output 
    if is_manual:
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
        for player1, player1_vid_idx, player1_double_hand_backhand in test_paths: 
            for player2, player2_vid_idx, player2_double_hand_backhand in player2_paths: 
                player1_vid = f"{player1}{player1_vid_idx}"
                player2_vid = f"{player2}{player2_vid_idx}"
                player_compare_detail = f"{player1}{player1_vid_idx}_{player2}{player2_vid_idx}"
                player_compare = f"{player1}_{player2}"
                for swing_type in service:
                    if swing_type == "backhand" and player1_double_hand_backhand != player2_double_hand_backhand:
                        break
                    loaded_swings1 = load_swings_from_json(f"{player1}/{player1_vid}/{player1_vid}_{swing_type}.json")
                    loaded_swings2 = load_swings_from_json(f"{player2}/{player2_vid}/{player2_vid}_{swing_type}.json")
                    player_compare_vid = f"{player1_vid}_{player2_vid}"
                    folder1 = f"{player1}/{player1_vid}/{swing_type}"
                    folder2 = f"{player2}/{player2_vid}/{swing_type}"
                    
                    videos1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.mp4'))]
                    videos2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.mp4'))]
                    folder_path = f"{player1}/{player1_vid}/{player_compare_detail}/{swing_type}"
                    vid2_swing_lens = []
                    clear_output_folder(folder_path)
                    
                    for idx1, video1 in enumerate(videos1):
                        if "_fps." in video1 or "stacked." in video1:
                            continue  

                        container1 = av.open(video1)
                        vid1_duration = container1.duration / av.time_base

                        diffs = []
                        min_vid2 = None
                        min_vid2_idx = -1
                        best_swing2 = None
                        video1_names = video1.split('/')
                        video1_name = video1_names[-1].split('.')
                        swing1 = next((entry for entry in loaded_swings1 if entry.name == video1_names[-1]), None)
                        
                        for idx2, video2 in enumerate(videos2):
                            if "_fps." in video2 or "stacked." in video2:
                                continue

                            container2 = av.open(video2)
                            vid2_duration = container2.duration / av.time_base
                            video2_names = video2.split('/')
                            video2_name = video2_names[-1].split('.')
                            swing2 = next((entry for entry in loaded_swings2 if entry.name == video2_names[-1]), None)
                            swing_len_diff = abs(swing2.get_swing_len() - swing1.get_swing_len())
                            
                            
                            x_movement_diff = abs(abs(swing2.x_movement) - abs(swing1.x_movement))
                            x_same_dir = swing2.x_movement * swing1.x_movement >= 0 and x_movement_diff < min(abs(swing1.x_movement), abs(swing2.x_movement))
                            
                            z_movement_diff = abs(abs(swing2.z_movement) - abs(swing1.z_movement))
                            z_same_dir = swing2.z_movement * swing1.z_movement >= 0 and z_movement_diff < min(abs(swing1.z_movement), abs(swing2.z_movement))
                            
                            similar_swing_len = abs(swing1.get_total_len() - swing2.get_total_len()) < 0.4
                            same_stance = swing1.stance == swing2.stance
                            
                            # Define a weighted criteria for comparison
                            total_diff = x_movement_diff + z_movement_diff # Scale movement difference
                            if x_same_dir and z_same_dir and same_stance and similar_swing_len:
                                diffs.append((total_diff, video2, swing2))
                        
                        
                        if len(diffs) != 0: 
                            sorted_diffs = sorted(diffs, key = lambda x:x[0])
                            for i in range(min(len(sorted_diffs), N)):
                                best_swing2 = diffs[i][2]
                                min_video2 = diffs[i][1]
                                min_video2_names = min_video2.split('/')
                                min_video2_name = min_video2_names[-1].split('.')
                                
                                swing1.print_info()
                                best_swing2.print_info()
                                player_compare_vid_swing = f"{video1_name[0]}_{min_video2_name[0]}"
                                overlay_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}/{player_compare_vid_swing}_overlay{extension}"
                                comparison_path = f"{player1}/{player1_vid}/{player_compare_detail}/{swing_type}/{player_compare_vid_swing}{extension}"
                                sync_video_timings(video1, min_video2, overlay_path, comparison_path, swing_type, swing1, best_swing2)
                