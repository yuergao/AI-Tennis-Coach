import os
import numpy as np
import os
from video_utils import *
from calc_utils import *
from swing import Swing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


'''
-----------------------------------------------------------------------------------
---------------------------------- Segment ----------------------------------------
-----------------------------------------------------------------------------------
'''
# finds the best frame according to the pose criteria. 
# called by find_most_parallel_and_furthest_segment.   
def find_best_frame(orientations, criteria, start_frame):
    best_frame = None
    best_score = float("inf")
    score1 = float("inf")
    score2 = float("inf")

    prev_orientation = None  # Initialize previous orientation

    for i, orientation in enumerate(orientations, start=start_frame):  # Loop to check pairs of consecutive frames
        if orientation:
            if prev_orientation:  # Ensure there is a previous orientation
                # Use both current and previous orientations in the criteria
                score1 = criteria(prev_orientation, i-1)
                score2 = criteria(orientation, i)

            # Check if the score is less than the current best_score
            if score1 < best_score and score2 < best_score:
                best_score = score1
                best_frame = i

        prev_orientation = orientation  # Update prev_orientation
    return best_frame  

# called by driver
# function to detect start and end indices of tennis swings based on wrist position.
def detect_swings(keypoints, fps):
    orientations = calculate_body_orientation(keypoints)
    forehand_swings = find_matching_swings_forehand(orientations,fps)
    backhand_swings = find_matching_swings_backhand(orientations,fps, forehand_swings)
    # Look for key aspects of tennis swing based on wrist position
    return [forehand_swings, backhand_swings]


'''
------------------------------------------------------------------------------------
---------------------------------- Forehand ----------------------------------------
------------------------------------------------------------------------------------
'''
def forehand_start_criteria(orientation, i):
    """
    Pose 1 Criteria
    
    Parameters:
    - orientation: A dictionary containing body landmarks' x and y coordinates.
    - i: Current frame index.
    
    Returns:
    - A score where 0 means all conditions are met, and inf means a condition failed.
    """
    
    # Condition 1: Right hand is to the right of the right shoulder
    arm_on_right_side = orientation["rw_x"] > orientation["rs_x"] and orientation["le_x"] > orientation["rs_x"] and orientation["lw_x"] > orientation["rs_x"]
    arm_on_right_side_score = 0 if arm_on_right_side else float("inf")
    
    # Condition 2: Left wirst is in front of the right shoulder
    left_hand_right_of_right_shoulder = orientation["lw_z"] > orientation["rs_z"]
    left_hand_position_score = 0 if left_hand_right_of_right_shoulder else float("inf")
    
    # Condition 3: Body is rotated to the right (left shoulder is in front of the right shoulder)
    body_rotated_right = orientation["ls_z"] > orientation["rs_z"]
    body_rotation_score = 0 if body_rotated_right else float("inf")
    
    # Condition 5: Left foot is in front of the right foot (smaller y value for the left foot)
    left_foot_in_front = orientation["la_z"] > orientation["ra_z"]
    foot_position_score = 0 if left_foot_in_front else float("inf")
    
    wrist_together = orientation["wrist_distance"]
    wrist_together_score = 0 if orientation["wrist_distance"] > 0.4 else float("inf")

    # Total score (0 means all conditions met, inf means at least one condition failed)
    score = (
        arm_on_right_side_score +
        left_hand_position_score +
        body_rotation_score +
        foot_position_score
        #+wrist_together_score
    )
    
    return score

# Pose 2 Criteria: minimize body rotation while right arm is on right side of body and left arm is not on right side of body
def forehand_end_criteria(orientation, i):
    # Minimize the body rotation (i.e., body_rotation should be as close to 0 as possible)
    #body_rotation_score = abs(orientation["body_rotation"])

    # Ensure the entire right arm is on the right side of the body
    arm_on_right_side = orientation["rw_x"] < orientation["rs_x"] 
    arm_on_right_side_score = 0 if arm_on_right_side else float("inf")  # 0 if arm is on the right side, inf if not

    # Ensure the left hand is to the left of the right shoulder
    left_hand_right_of_right_shoulder = orientation["lw_x"] < orientation["rs_x"]
    left_hand_position_score = 0 if left_hand_right_of_right_shoulder else float("inf")  # 0 if condition is met, inf otherwise
    
    #Ensure that the right hand is in front of the body 
    right_hand_in_front = orientation["rw_z"] > orientation["rs_z"]
    right_hand_in_front_score = 0 if right_hand_in_front else float("inf")
    
    # Combine all criteria (can be weighted if necessary)
    total_score = arm_on_right_side_score + left_hand_position_score + right_hand_in_front_score
    if total_score < float("inf"):
        pass
        #print(f"frame{i}: {total_score}")  # Debugging output
    return total_score


def find_matching_swings_forehand(orientations, fps, threshold=float("inf"), velocity_threshold=2):
    """
    Find all frames that meet the criteria within a specified range, ensuring no
    frames are selected within a cooldown period after a match. Includes wrist velocity as a criterion.

    Parameters:
    - orientations: List of pose orientations for each frame, including wrist coordinates.
    - fps: Frames per second of the video.
    - threshold: Maximum acceptable score to include the frame (default: infinity).
    - velocity_threshold: Minimum wrist velocity required to consider the frame (default: 5.0).

    Returns:
    - matching_swings: List of indices of frames that meet the criteria.
    """
    matching_swings = []
    cooldown_seconds = 2
    cooldown_frames = int(cooldown_seconds * fps)  # Number of frames to skip during cooldown
    last_selected_frame = -cooldown_frames  # Initialize to a value before the start

    min_body_metrics = lambda orientation: len([v for v in orientation.values() if v is not None])
    required_body_metrics = lambda orientation: len(orientation) / 2  # At least half of the metrics

    for i, orientation in enumerate(orientations):
        # Skip if within cooldown
        if i < last_selected_frame + cooldown_frames:
            continue

        if orientation:
            # Ensure the minimum number of valid metrics
            if min_body_metrics(orientation) < required_body_metrics(orientation):
                continue

            # Check forehand criteria score
            if forehand_start_criteria(orientation, i) < threshold and not backhand_start_criteria(orientation, i) < threshold:
                # Define the window for wrist velocity and scores
                window_start = i
                window_end = min(i + int(fps), len(orientations) - 1)

                # Extract right wrist coordinates for the entire window
                right_wrist_coords = [
                    orientations[k]['right_wrist'] for k in range(window_start, window_end) if orientations[k]
                ]
                
                valid_distances = [
                    orientations[k]['wrist_distance'] for k in range(window_start, window_end) if orientations[k]
                ]

                if not valid_distances:
                    continue
                
                wrists_together_check = np.max(valid_distances) > 1

                
                # Calculate wrist velocities for the window
                avg_wrist_velocity, moves_right = calculate_average_wrist_velocity(right_wrist_coords, fps, orientation['shoulder_hip_distance'])
                
                # Check if any velocity in the window exceeds the threshold
                velocity_exceeds_threshold = avg_wrist_velocity > velocity_threshold

                if velocity_exceeds_threshold and wrists_together_check:
                    scores_window = []
                    for j in range(i, min(i + int(fps / 10), len(orientations) - 1)):
                        if orientations[j]:  # Ensure valid orientation data
                            score = forehand_start_criteria(orientations[j], j)
                            scores_window.append(score)
                    


                    # Ensure scores meet the criteria
                    if scores_window and len(scores_window) > fps/15:
                        avg_score = np.mean(scores_window)
                        if avg_score < threshold:
                            start_frame = max(0, i)
                            end_frame = min(i + int(3 * fps), len(orientations) - 1)
                            contact_swing = find_best_frame(orientations[start_frame:end_frame], forehand_end_criteria, start_frame)
                            if contact_swing:
                                swing_len = contact_swing - start_frame
                                if swing_len < fps / 3:
                                    continue
                                # Passes All Check, swing identidied, create Swing
                                swing = Swing(
                                    type="forehand",
                                    start_frame= start_frame,
                                    end_frame = contact_swing,
                                    padded_start_frame=max(1, start_frame - fps),
                                    padded_end_frame=min(contact_swing + fps, len(orientations) - 1),
                                    avg_wrist_velocity= avg_wrist_velocity,
                                    movement = (orientations[max(1, start_frame - fps)]['rs_x'] - orientations[contact_swing]['rs_x']) * 100,
                                    fps=fps  
                                )
                                matching_swings.append(swing)
                                swing.print_info()
                                last_selected_frame = i  # Update the last selected frame

    return matching_swings



'''
------------------------------------------------------------------------------------
---------------------------------- Backhand ----------------------------------------
------------------------------------------------------------------------------------
'''
def backhand_start_criteria(orientation, i):
    """
    Pose 2 Criteria
    
    Parameters:
    - orientation: A dictionary containing body landmarks' x and y coordinates.
    - i: Current frame index.
    
    Returns:
    - A score where 0 means all conditions are met, and inf means a condition failed.
    """
 
    # Condition 1: both wrists to the left of the left shoulder
    arm_on_right_side = orientation["rw_x"] < orientation["ls_x"] and orientation["lw_x"] < orientation["ls_x"]
    arm_on_right_side_score = 0 
    
    # Condition 2: Left wirst behind the right shoulder
    left_hand_right_of_right_shoulder = orientation["lw_z"] < orientation["rs_z"]
    left_hand_position_score = 0 if left_hand_right_of_right_shoulder else float("inf")
    
    # Condition 3: Body is rotated to the left (right shoulder is in front of the left shoulder)
    body_rotated_right = orientation["ls_z"] < orientation["rs_z"]
    body_rotation_score = 0 if body_rotated_right else float("inf")
    
    # Condition 5: Right foot is in front of the left foot (smaller y value for the left foot)
    left_foot_in_front = orientation["la_z"] < orientation["ra_z"]
    foot_position_score = 0 if left_foot_in_front else float("inf")
    
    # Condition 7: Right wrist is below the right shoulder
    right_wrist_below_shoulder = orientation["rw_y"] > orientation["rs_y"]
    wrist_below_shoulder_score = 0 if right_wrist_below_shoulder else float("inf")
    
    left_elbow_angle = orientation["elbow_angle_left"] > 120 and orientation["elbow_angle_right"] > 120
    left_elbow_angle_score = orientation["elbow_angle_left"] if left_elbow_angle else float("inf")
    #print(orientation["elbow_angle_left"], orientation["elbow_angle_right"])
    
    wrist_together = orientation["wrist_distance"]
    wrist_together_score = 0 if orientation["wrist_distance"] < 0.2 else float("inf")
    
    left_wrist_to_shoulder_distance = orientation["lw_x"] - orientation["ls_x"] 
    left_wrist_to_shoulder_distance_score = orientation["lw_x"] - orientation["ls_x"] if left_wrist_to_shoulder_distance > 0.5 else float("inf")
    
    # Total score (0 means all conditions met, inf means at least one condition failed)
    score = (
        arm_on_right_side_score +
        left_hand_position_score +
        body_rotation_score +
        foot_position_score + 
        wrist_below_shoulder_score + 
        wrist_together_score
    )
    
    if score < float("inf"):
        #print(orientation["elbow_angle_left"], orientation["elbow_angle_right"])
        pass
        #print(f"frame: {i}, wrist together: {left_wrist_to_shoulder_distance}")
        
    return score

# backhand end Criteria: minimize body rotation while right arm is on right side of body and left arm is not on right side of body
def backhand_end_criteria(orientation, i):
    # Ensure both hands are on the right side of the body
    both_hands_on_right_side = orientation["rw_x"] > orientation["ls_x"] and orientation["lw_x"] > orientation["ls_x"]
    both_hands_on_right_side_score =  0 if both_hands_on_right_side else float("inf")  # 0 if arm is on the right side, inf if not

    # Condition 2: Left wirst in front of the right shoulder
    left_hand_front_of_right_shoulder = orientation["lw_z"] > orientation["rs_z"]
    left_hand_front_of_right_shoulder_score = 0 if left_hand_front_of_right_shoulder else float("inf")
    
    # Combine all criteria (can be weighted if necessary)
    total_score = both_hands_on_right_side_score + left_hand_front_of_right_shoulder_score
    
    return total_score


def find_matching_swings_backhand(orientations, fps, forehand_swings, threshold=float("inf"), velocity_threshold=3):
    """
    Find all backhand frames that meet the criteria, ensuring no frames are selected
    within a cooldown period after a match. Includes wrist velocity as a criterion.

    Parameters:
    - orientations: List of pose orientations for each frame.
    - fps: Frames per second of the video.
    - forehand_frames: List of forehand frame ranges to skip for backhands.
    - threshold: Maximum acceptable score to include the frame (default: infinity).
    - velocity_threshold: Minimum average wrist velocity required to consider the frame (default: 5.0).

    Returns:
    - matching_swings: List of indices of frames that meet the criteria.
    """
    matching_swings = []
    cooldown_seconds = 2
    cooldown_frames = int(cooldown_seconds * fps)  # Number of frames to skip during cooldown
    last_selected_frame = -cooldown_frames  # Initialize to a value before the start

    # Define the minimum number of body metrics required
    min_body_metrics = lambda orientation: len([v for v in orientation.values() if v is not None])
    required_body_metrics = lambda orientation: len(orientation) / 2  # At least half of the metrics

    for i, orientation in enumerate(orientations):
        # Skip if within cooldown
        if i < last_selected_frame + cooldown_frames:
            continue

        if orientation:
            # Skip the frame if less than half of the body metrics are detected
            if min_body_metrics(orientation) < required_body_metrics(orientation):
                continue

            if backhand_start_criteria(orientation, i) < threshold:
                # Define the window for wrist velocity and scores
                window_start = max(1, i - int(fps))
                window_end = min(i+ int(fps),len(orientations) - 1)

                # Extract right wrist coordinates for the entire window
                right_wrist_coords = [
                    orientations[k]['right_wrist'] for k in range(window_start + int(fps), min(window_end + int(fps),len(orientations) - 1)) if orientations[k]
                ]
                left_wrist_coords = [
                    orientations[k]['left_wrist'] for k in range(window_start+ int(fps), min(window_end + int(fps),len(orientations) - 1)) if orientations[k]
                ]
                # Calculate wrist velocities for the window
                avg_wrist_velocity_right, moves_right_rw = calculate_average_wrist_velocity(right_wrist_coords, fps, orientation['shoulder_hip_distance'])
                avg_wrist_velocity_left, moves_right_lw = calculate_average_wrist_velocity(left_wrist_coords, fps, orientation['shoulder_hip_distance'])
                avg_wrist_velocity = avg_wrist_velocity_left
                if avg_wrist_velocity > velocity_threshold:
                    #print(f"Backhand frames: {i} to {window_end}, wrist_velocity: {avg_wrist_velocity}")
                    pass

                # Check if average velocity exceeds the threshold
                velocity_exceeds_threshold = avg_wrist_velocity > velocity_threshold

                if velocity_exceeds_threshold and moves_right_rw:
                    scores_window = []
                    for j in range(window_start, window_end):
                        if orientations[j] and orientations[j]["wrist_distance"] < 0.3:  # Ensure valid orientation data
                            score = (orientations[j]["rs_z"] - orientations[j]["lw_z"]) / orientations[j]["shoulder_hip_distance"]
                            scores_window.append(score)

                    # Ensure scores meet the criteria
                    if scores_window:
                        avg_score = np.mean(scores_window)
                        scores_window.sort()
                        wrist_depth_diff = scores_window[-1]
                        if len(scores_window) > fps / 2 and avg_score > 0 and wrist_depth_diff > 1:
                            skip = False
                            for forehand_swing in forehand_swings:
                                if forehand_swing.is_within_padded_range(i):
                                    skip = True
                                    break
                            if not skip:
                                #print(f"Backhand frame {i}: {avg_score}, scores {wrist_depth_diff}")
                                start_frame = max(0, i)
                                end_frame = min(i + int(3 * fps), len(orientations) - 1)
                                contact_swing = find_best_frame(orientations[start_frame:end_frame], backhand_end_criteria, start_frame)
                                if contact_swing:
                                    swing_len = contact_swing - start_frame
                                    if swing_len < fps / 3:
                                        continue
                                    # Passes All Check, swing identidied, create Swing
                                    swing = Swing(
                                        type="backhand",
                                        avg_wrist_velocity=avg_wrist_velocity,
                                        start_frame=i,
                                        end_frame=contact_swing,
                                        padded_start_frame = window_start,
                                        padded_end_frame= min(contact_swing + fps, len(orientations) - 1),
                                        movement = (orientations[window_start]['rs_x'] - orientations[contact_swing]['rs_x']) * 100,
                                        fps=fps  
                                    )
                                    swing.print_info()
                                    matching_swings.append(swing)
                                    last_selected_frame = i  # Update the last selected frame

    return matching_swings
