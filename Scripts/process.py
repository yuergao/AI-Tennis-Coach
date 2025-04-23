from segment import *
from sync import *
from video_utils import *
from calc_utils import *
import os
import av

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# pro local paths: name, idx, double_backhand, right_handed
sabalenka = ('sabalenka', 2, True, True, 1.82)
swiatek = ('swiatek', 2, True, True, 1.79)
rybakina = ('rybakina', 2, True, True, 1.82)

djokovic = ('djokovic', 1, True, True, 1.88)
fonseca = ('fonseca', 1, True, True, 1.83)  
zverev = ('zverev', 1, True, True, 1.98)
alcaraz = ('alcaraz', 2, True, True, 1.83)
federer = ('federer', 2, False, True, 1.85)
draper = ('draper', 1, True, False, 1.93)
holger = ('holger', 1, True, True, 1.83)
nadal = ('nadal', 1, True, False, 1.85)

audrey = ('audrey', 22, True, True, 1.78)
original = ('original', 2, True, True, 1.73)
a = ('a', 1, True, True, 1.73)

# User inputs
pipeline1 = True # <<<<<<<<<<<<<<<<<<<<<<<
pipeline2 = True # <<<<<<<<<<<<<<<<<<<<<<<

test_paths = [a] # name, idx, double_backhand, right_handed <<<<<<<<<<<<<<<<<<<<<<<
player2_paths = [federer, fonseca] # <<<<<<<<<<< automate <<<<<<<<<<<<
service = ["forehand"] # <<<<<<<<< service
Amateur = False # <<<<< generate skeletal breakdown for full video
video_extensions = ('.mp4', '.mov')

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
    for player, player_vid_idx, double_hand_backhand,right_handed,height in test_paths:
        folder_path = f"{player}/{player}{player_vid_idx}"
        for video, video_name in ((f, os.path.splitext(f)[0]) for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)):
            video_path = os.path.join(folder_path, video)
            process_video_path = os.path.join(folder_path, video_name +"_processed.mp4")
            forehand_path = os.path.join(folder_path, "forehand")
            backhand_path = os.path.join(folder_path, "backhand")
        
            #Extract keypoints from the video
            json_path = os.path.join(folder_path, video_name + "_metadata.json")
            forehand_swing_json = os.path.join(forehand_path,f"{video_name}_forehand.json")
            backhand_swing_json = os.path.join(backhand_path,f"{video_name}_backhand.json")
            
            #save video coordinate info to json
            if not os.path.isfile(json_path):
                keypoints, fps = extract_keypointsv6_asimg(video_path,right_handed)
                print(f"Finished keypoint extraction for {video}")
                save_detected_keypoints_to_json(keypoints,json_path)
                forehand_segments, backhand_segments = detect_swings(keypoints, fps, double_hand_backhand, right_handed, height)
                if not right_handed:
                    for frame in keypoints:
                        for i in range(len(frame)):
                            x, y, z = frame[i]
                            if x is not None:
                                frame[i] = (1.0 - x, y, z)  # Flip x across vertical axis
                # Detect swing boundaries
                keypoints = np.array(keypoints)
                print(f"Detected {len(forehand_segments)} forehands and {len(backhand_segments)} backhands.")

                # Process each swing and extract video segments
                if 'forehand' in service and len(forehand_segments) > 0:
                    for idx, forehand in enumerate(forehand_segments):
                        output_video = os.path.join(forehand_path,f"{video_name}_forehand{idx+1}.mp4")
                        output_stick_video = os.path.join(forehand_path,f"{video_name}_forehand{idx+1}stacked.mp4")
                        forehand_segments[idx].name = f"{video_name}_forehand{idx+1}.mp4"
                        extract_motion_segment(video_path,forehand, output_video, output_stick_video, keypoints)
                    
                    save_swings_to_json(forehand_segments, forehand_swing_json)   
            
                if 'backhand' in service and len(backhand_segments) > 0:
                    # Process each swing and extract video segments
                    for idx, backhand in enumerate(backhand_segments):
                        #find previous directory
                        output_video = os.path.join(backhand_path,f"{video_name}_backhand{idx+1}.mp4")
                        output_stick_video =  os.path.join(backhand_path,f"{video_name}_backhand{idx+1}stacked.mp4")
                        backhand_segments[idx].name = f"{video_name}_backhand{idx+1}.mp4"
                        extract_motion_segment(video_path, backhand, output_video, output_stick_video, keypoints)
                            
                    save_swings_to_json(backhand_segments, backhand_swing_json)
                  
            else:
                print(f"Keypoint file exists for {video}, using saved coordinates")
                
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                keypoints = load_keypoints_from_json(json_path)
                forehand_segments, backhand_segments = detect_swings(keypoints, fps, double_hand_backhand, right_handed, height)
                print(f"Detected {len(forehand_segments)} forehands and {len(backhand_segments)} backhands.")
                # Process each swing and extract video segments
                if 'forehand' in service and len(forehand_segments) > 0:
                    for idx, forehand in enumerate(forehand_segments):
                        output_video = os.path.join(forehand_path,f"{video_name}_forehand{idx+1}.mp4")
                        output_stick_video = os.path.join(forehand_path,f"{video_name}_forehand{idx+1}stacked.mp4")
                        forehand_segments[idx].name = f"{video_name}_forehand{idx+1}.mp4"
                        extract_motion_segment(video_path,forehand, output_video, output_stick_video, keypoints)
                    
                    save_swings_to_json(forehand_segments, forehand_swing_json)   
            
                if 'backhand' in service and len(backhand_segments) > 0:
                    # Process each swing and extract video segments
                    for idx, backhand in enumerate(backhand_segments):
                        #find previous directory
                        output_video = os.path.join(backhand_path,f"{video_name}_backhand{idx+1}.mp4")
                        output_stick_video =  os.path.join(backhand_path,f"{video_name}_backhand{idx+1}stacked.mp4")
                        backhand_segments[idx].name = f"{video_name}_backhand{idx+1}.mp4"
                        extract_motion_segment(video_path, backhand, output_video, output_stick_video, keypoints)
                            
                    save_swings_to_json(backhand_segments, backhand_swing_json)
                  
        
        if Amateur:
            output_video = os.path.join(folder_path,f"{video_name}_all.mp4")
            output_stick_video = os.path.join(backhand_path,f"{video_name}_all.mp4")
            extract_all_motion_segment(video_path,output_video,output_stick_video)
                    
'''
-----------------------------------------------------------------------------------------------------------------------------------
---------------------- Pipeline Segment 2: output segmented swing comparison videos------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------
'''
N = 2 # <<<<<<< number of similar videos

if pipeline2:
    for player1, player1_vid_idx, player1_double_hand_backhand, right_handed, height in test_paths: 
        for player2, player2_vid_idx, player2_double_hand_backhand, right_handed, height2 in player2_paths: 
            player1_vid = f"{player1}{player1_vid_idx}"
            player2_vid = f"{player2}{player2_vid_idx}"
            player_compare_detail = f"{player1}{player1_vid_idx}_{player2}{player2_vid_idx}"
            player_compare = f"{player1}_{player2}"
            
            folder_path2 = f"{player2}/{player2}{player2_vid_idx}"
            for swing_type in service:
                if swing_type == "backhand" and player1_double_hand_backhand != player2_double_hand_backhand:
                    break
                folder_path1_service = f"{player1}/{player1}{player1_vid_idx}/{swing_type}"
                json_files1_service = [os.path.join(folder_path1_service, f) for f in os.listdir(folder_path1_service) if os.path.isdir(folder_path1_service) and f.lower().endswith('.json')]
                folder_path2_service = f"{player2}/{player2}{player2_vid_idx}/{swing_type}"
                json_files2_service = [os.path.join(folder_path2_service, f) for f in os.listdir(folder_path2_service) if os.path.isdir(folder_path2_service) and f.lower().endswith('.json')]
                loaded_swings1 = []
                loaded_swings2 = []
                
                for loaded_swing1 in json_files1_service:
                    loaded_swings1+= load_swings_from_json(loaded_swing1)
                    
                for loaded_swing2 in json_files2_service:
                    loaded_swings2 += load_swings_from_json(loaded_swing2)
                
                player_compare_vid = f"{player1_vid}_{player2_vid}"
                folder1 = f"{player1}/{player1_vid}/{swing_type}"
                folder2 = f"pros/{player2_vid}/{swing_type}"
                if os.path.isdir(folder1):
                    videos1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith((".mp4"))]
                else:
                    continue
                if os.path.isdir(folder2):
                    videos2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith((".mp4"))]
                else:
                    continue
                folder_path = f"{player1}/{player1_vid}/{player_compare_detail}/{swing_type}"
                vid2_swing_lens = []
                for idx1, video1 in enumerate(videos1):
                    if "_fps." in video1 or "stacked." in video1:
                        continue  

                    container1 = av.open(video1)
                    if not container1 or not container1.duration:
                        continue
                    
                    vid1_duration = container1.duration / av.time_base

                    diffs = []
                    diffs_exlude_stance = []
                    diffs_exlude_stance_swing_len = []
                    diffs_exlude_all = []
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
                        #breakpoint()
                        if not swing2 or not swing1:
                            continue
                        swing_len_diff = abs(swing2.get_swing_len() - swing1.get_swing_len())
                        
                        
                        x_movement_diff = abs(abs(swing2.x_movement) - abs(swing1.x_movement))
                        x_same_dir = swing2.x_movement * swing1.x_movement >= 0 and x_movement_diff < min(abs(swing1.x_movement), abs(swing2.x_movement))
                        
                        z_movement_diff = abs(abs(swing2.z_movement) - abs(swing1.z_movement))
                        z_same_dir = swing2.z_movement * swing1.z_movement >= 0 and z_movement_diff < min(abs(swing1.z_movement), abs(swing2.z_movement))
                        
                        similar_swing_len = abs(swing1.get_total_len() - swing2.get_total_len()) < 0.4
                        same_stance = swing1.stance == swing2.stance
                        
                        # Define a weighted criteria for comparison
                        total_diff = x_movement_diff + z_movement_diff # Scale movement difference
                        if x_same_dir:
                            diffs_exlude_all.append((total_diff, video2, swing2))
                        if x_same_dir and z_same_dir:
                            diffs_exlude_stance_swing_len.append((total_diff, video2, swing2))
                            if similar_swing_len:
                                diffs_exlude_stance.append((total_diff, video2, swing2))
                                if same_stance:
                                    diffs.append((total_diff, video2, swing2))
                    if len(diffs) >= 3: 
                        sorted_diffs = sorted(diffs, key = lambda x:x[0])
                    elif len(diffs_exlude_stance) >= 3:
                        sorted_diffs = sorted(diffs_exlude_stance, key = lambda x:x[0])
                    elif len(diffs_exlude_stance_swing_len) >= 3:
                        sorted_diffs = sorted(diffs_exlude_stance_swing_len, key = lambda x:x[0])    
                    else:
                        sorted_diffs = sorted(diffs_exlude_all, key = lambda x:x[0])  
                    
                    if len(sorted_diffs) != 0:
                        for i in range(min(len(sorted_diffs), N)):
                            best_swing2 = sorted_diffs[i][2]
                            min_video2 = sorted_diffs[i][1]
                            min_video2_names = min_video2.split('/')
                            min_video2_name = min_video2_names[-1].split('.')
                            
                            swing1.print_info()
                            best_swing2.print_info()
                            player_compare_vid_swing = f"{video1_name[0]}_{min_video2_name[0]}"
                            overlay_path = f"{player1}/{player_compare}/{player_compare_vid}/{swing_type}/{player_compare_vid_swing}_overlay.mp4"
                            comparison_path = f"{player1}/{player1_vid}/{player_compare_detail}/{swing_type}/{player_compare_vid_swing}.mp4"
                            sync_video_timings(video1, min_video2, overlay_path, comparison_path, swing_type, swing1, best_swing2)
                    