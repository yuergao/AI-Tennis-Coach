import os
import cv2
import mediapipe as mp
import numpy as np
import shutil

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to extract keypoints and detect the start of the swing
def detect_swing_start(keypoints, threshold=0.5):
    """
    Detects the start of the swing based on the movement of the wrist (right wrist as an example).
    
    :param keypoints: List of keypoints for a frame.
    :param threshold: The movement threshold that signifies the start of the swing.
    :return: True if swing start is detected, False otherwise.
    """
    if keypoints is None or len(keypoints) <= 28:
        return False
    
    # Assuming right wrist is at index 28
    wrist_x, wrist_y = keypoints[28]  # Right wrist keypoint

    if wrist_x is not None and wrist_y is not None:
        # Check for a significant movement in the wrist (you can adjust this based on what you consider a "swing start")
        if wrist_y > threshold:  # You can use any condition that identifies the swing start
            return True
    return False

def calculate_arm_orientation(keypoints, left_shoulder_index=11, right_shoulder_index=12, 
                              left_elbow_index=13, right_elbow_index=14, 
                              left_wrist_index=15, right_wrist_index=16, 
                              start_frame=5):
    orientations = []
    for i in range(start_frame, len(keypoints)):
        if keypoints[i] is not None:
            try:
                # Right arm orientation (shoulder, elbow, wrist)
                rs_x, rs_y = keypoints[i][right_shoulder_index]
                re_x, re_y = keypoints[i][right_elbow_index]
                rw_x, rw_y = keypoints[i][right_wrist_index]

                # Left arm orientation (shoulder, elbow, wrist)
                ls_x, ls_y = keypoints[i][left_shoulder_index]
                le_x, le_y = keypoints[i][left_elbow_index]
                lw_x, lw_y = keypoints[i][left_wrist_index]

                # Calculate angle between shoulder, elbow, and wrist to check straightness (cosine law approach)
                # Vector from shoulder to elbow
                arm_vec_x = re_x - rs_x
                arm_vec_y = re_y - rs_y
                # Vector from elbow to wrist
                forearm_vec_x = rw_x - re_x
                forearm_vec_y = rw_y - re_y
                # Dot product to check alignment
                dot_product = arm_vec_x * forearm_vec_x + arm_vec_y * forearm_vec_y
                arm_magnitude = np.sqrt(arm_vec_x**2 + arm_vec_y**2)
                forearm_magnitude = np.sqrt(forearm_vec_x**2 + forearm_vec_y**2)

                # Cosine of the angle between vectors (ideal is 1 if perfectly straight)
                cos_angle = dot_product / (arm_magnitude * forearm_magnitude)
                straightness = 1 - abs(cos_angle)  # Closer to 0 means more straight

                # Check if the left arm is not on the right side of the body
                # Using a basic assumption that the body center is the average of both shoulders
                body_center_x = (rs_x + ls_x) / 2
                left_arm_not_on_right_side = lw_x <= body_center_x

                # Add the straightness and wrist's x position for the right arm to check its horizontal alignment
                if left_arm_not_on_right_side:
                    orientations.append((straightness, rw_x))  # Track right arm's straightness and wrist position
                else:
                    orientations.append((1, 0))  # Invalid frame where left arm is on the right side of the body
            except:
                orientations.append((1, 0))  # Default values in case of error or missing keypoints
        else:
            orientations.append((1, 0))  # Default values for missing keypoints
    return orientations

# Function to find the segment with the lowest average straightness (most extended arm)
def find_most_aligned_segment(arm_orientations, segment_length=10):
    min_avg_straightness = float('inf')
    best_segment_start = None

    for i in range(len(arm_orientations) - segment_length + 1):
        segment = arm_orientations[i:i + segment_length]
        avg_straightness = np.mean([s[0] for s in segment])  # Average straightness of the segment

        if avg_straightness < min_avg_straightness:
            min_avg_straightness = avg_straightness
            best_segment_start = i
    
    if best_segment_start is None:
        print("Error: Could not find valid aligned segments with arm straightness.")
        
    return best_segment_start

def resize_and_pad(frame, target_width, target_height):
    original_height, original_width = frame.shape[:2]
    original_aspect_ratio = original_width / original_height

    target_aspect_ratio = target_width / target_height

    if original_aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / original_aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * original_aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))

    padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

    return padded_frame

def sync_video_timings(video1_path, video2_path, output_path, top_bottom_original_path):  
    keypoints1, scales1 = extract_keypoints_and_scale(video1_path)
    keypoints2, scales2 = extract_keypoints_and_scale(video2_path)

    first_valid_frame_idx = None
    for idx in range(len(keypoints1)):
        if keypoints1[idx] is not None and keypoints2[idx] is not None:
            first_valid_frame_idx = idx
            break

    if first_valid_frame_idx is None:
        print("Error: Could not find a valid frame with keypoints for both players.")
        return

    arm_orientation1 = calculate_arm_orientation(keypoints1, start_frame=5)
    arm_orientation2 = calculate_arm_orientation(keypoints2, start_frame=5)

    aligned_segment1_start = find_most_aligned_segment(arm_orientation1)
    aligned_segment2_start = find_most_aligned_segment(arm_orientation2)

    time_offset = aligned_segment2_start - aligned_segment1_start

    print(f"Time offset between {video1_path} and {video2_path}: {time_offset} frames")

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open video files.")
        return

    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    max_fps = max(fps1, fps2)
    original_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if original_height > original_width:
        target_width = 720
        target_height = 1280
        output_video_width = target_width * 2
        output_video_height = target_height
    else:
        target_width = 1920
        target_height = 1080
        output_video_width = target_width
        output_video_height = target_height * 2

    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), max_fps, (target_width, target_height))   
    out3 = cv2.VideoWriter(top_bottom_original_path, cv2.VideoWriter_fourcc(*'mp4v'), max_fps, (output_video_width, output_video_height))

    scale_ratio = scales1[first_valid_frame_idx] / scales2[first_valid_frame_idx]

    frame_idx1 = first_valid_frame_idx
    frame_idx2 = first_valid_frame_idx 

    if time_offset > 0:
        for _ in range(time_offset):
            cap2.read()
            frame_idx2 += 1
    elif time_offset < 0:
        for _ in range(-time_offset):
            cap1.read()
            frame_idx1 += 1

    # Frame counters
    frame_interval1 = int(max_fps / fps1)
    frame_interval2 = int(max_fps / fps2)
    frame_counter1 = 0
    frame_counter2 = 0
    
    
    while True:
        # Read frames based on frame counters
        if frame_counter1 % frame_interval1 == 0:
            ret1, frame1 = cap1.read()
            if not ret1:
                frame1 = np.zeros((original_height, original_width, 3), dtype=np.uint8)  # Placeholder for missing frames

        if frame_counter2 % frame_interval2 == 0:
            ret2, frame2 = cap2.read()
            if not ret2:
                frame2 = np.zeros((original_height, original_width, 3), dtype=np.uint8)  # Placeholder for missing frames


        if not ret1 or not ret2:
            break


        if frame_idx1 < len(keypoints1) and frame_idx2 < len(keypoints2):
            if keypoints1[frame_idx1] and keypoints2[frame_idx2]:

                fixed_scale1 = scales1[frame_idx1]
                adjusted_scale2 = scales2[frame_idx2] * scale_ratio
                aligned_kp1 = align_keypoints(keypoints1[frame_idx1], fixed_scale1, fixed_scale1, target_center=(0.5, 0.5))
                aligned_kp2 = align_keypoints(keypoints2[frame_idx2], scales2[frame_idx2], adjusted_scale2, target_center=(0.5, 0.5))

                blank_frame = np.zeros_like(frame1)
                overlay_frame = draw_overlay(blank_frame, aligned_kp1, aligned_kp2)

                overlay_frame = resize_and_pad(blank_frame, target_width, target_height)
                padded_frame1 = resize_and_pad(frame1, target_width, target_height)
                padded_frame2 = resize_and_pad(frame2, target_width, target_height)

                if original_height > original_width:
                    combined_frame = np.hstack([padded_frame1, padded_frame2])
                else:
                    combined_frame = np.vstack([padded_frame1, padded_frame2])

                out.write(overlay_frame)
                out3.write(combined_frame)

        frame_idx1 += 1
        frame_idx2 += 1
        

    cap1.release()
    cap2.release()
    out.release()
    out3.release()

    print(f"Synchronized overlay video saved to {output_path}")
    print(f"Top-bottom original video saved to {top_bottom_original_path}")
    
# Function to overlay keypoints onto a blank frame
def draw_overlay(frame, keypoints1, keypoints2, color1=(0, 255, 0), color2=(0, 0, 255)):
    """
    Draws the keypoints of two sets of landmarks onto a given frame.
    
    :param frame: The blank frame onto which keypoints are drawn.
    :param keypoints1: Aligned keypoints for the first person.
    :param keypoints2: Aligned keypoints for the second person.
    :param color1: Color for the first person's keypoints.
    :param color2: Color for the second person's keypoints.
    """
    height, width, _ = frame.shape

    # Draw keypoints for the first person
    if keypoints1:
        for x1, y1 in keypoints1:
            px1, py1 = int(x1 * width), int(y1 * height)
            cv2.circle(frame, (px1, py1), 5, color1, -1)

    # Draw keypoints for the second person
    if keypoints2:
        for x2, y2 in keypoints2:
            px2, py2 = int(x2 * width), int(y2 * height)
            cv2.circle(frame, (px2, py2), 5, color2, -1)

    return frame

# Function to extract keypoints and reference scale from a video
def extract_keypoints_and_scale(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    scales = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            # Extract keypoints
            landmarks = result.pose_landmarks.landmark
            keypoints.append([(lm.x, lm.y) for lm in landmarks])

            # Calculate reference scale using shoulders and hips
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            shoulder_distance = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 +
                                        (left_shoulder.y - right_shoulder.y) ** 2)
            hip_distance = np.sqrt((left_hip.x - right_hip.x) ** 2 +
                                   (left_hip.y - right_hip.y) ** 2)

            # Use the average of shoulder and hip distances
            average_scale = (shoulder_distance + hip_distance) / 2
            scales.append(average_scale)
        else:
            keypoints.append(None)
            scales.append(None)

    cap.release()
    return keypoints, scales

# Function to align keypoints by scaling and translating
def align_keypoints(keypoints, scale, target_scale, target_center):
    if keypoints is None:
        return None

    # Calculate the centroid of the keypoints
    centroid = np.mean(keypoints, axis=0)

    # Scale and translate keypoints
    aligned_keypoints = [
        ((x - centroid[0]) * target_scale / scale + target_center[0],
         (y - centroid[1]) * target_scale / scale + target_center[1])
        for x, y in keypoints
    ]
    return aligned_keypoints

def create_aligned_overlay_video(video1_path, video2_path, annotated1_path, annotated2_path, overlay_output_path, side_by_side_annotated_path, side_by_side_original_path):
    # Extract keypoints and scales
    keypoints1, scales1 = extract_keypoints_and_scale(video1_path)
    keypoints2, scales2 = extract_keypoints_and_scale(video2_path)

    # Find the first valid frame for both players
    first_valid_frame_idx = None
    for idx in range(len(keypoints1)):
        if keypoints1[idx] is not None and keypoints2[idx] is not None:
            first_valid_frame_idx = idx
            break

    if first_valid_frame_idx is None:
        print("Error: Could not find a valid frame with keypoints for both players.")
        return

    # Initialize video capture for all videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    cap_annot1 = cv2.VideoCapture(annotated1_path)
    cap_annot2 = cv2.VideoCapture(annotated2_path)

    # Video properties
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

    # Output video writers
    overlay_out = cv2.VideoWriter(overlay_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps1, (width, height))
    side_by_side_annot_out = cv2.VideoWriter(side_by_side_annotated_path, cv2.VideoWriter_fourcc(*'mp4v'), fps2, (width * 2, height))
    side_by_side_orig_out = cv2.VideoWriter(side_by_side_original_path, cv2.VideoWriter_fourcc(*'mp4v'), max(fps1, fps2), (width * 2, height))

    frame_idx = first_valid_frame_idx
    target_center = (0.5, 0.5)  # Normalize to the center of the frame

    # Initialize scale ratio based on the first valid frame
    scale_ratio = scales1[first_valid_frame_idx] / scales2[first_valid_frame_idx]

    while cap1.isOpened() and cap2.isOpened() and cap_annot1.isOpened() and cap_annot2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret_annot1, frame_annot1 = cap_annot1.read()
        ret_annot2, frame_annot2 = cap_annot2.read()

        if not ret1 or not ret2 or not ret_annot1 or not ret_annot2 or frame_idx >= len(keypoints1) or frame_idx >= len(keypoints2):
            break

        # Create a blank frame for the overlay video
        blank_frame = np.zeros_like(frame1)

        # Align keypoints and draw overlay
        if keypoints1[frame_idx] and keypoints2[frame_idx]:
            # Keep the first player's size fixed
            fixed_scale1 = scales1[frame_idx]
            adjusted_scale2 = scales2[frame_idx] * scale_ratio

            aligned_kp1 = align_keypoints(keypoints1[frame_idx], fixed_scale1, fixed_scale1, target_center)
            aligned_kp2 = align_keypoints(keypoints2[frame_idx], scales2[frame_idx], adjusted_scale2, target_center)

            blank_frame = draw_overlay(blank_frame, aligned_kp1, aligned_kp2)

        # Write to overlay video
        overlay_out.write(blank_frame)

        # Create side-by-side videos
        side_by_side_annot = np.hstack((frame_annot1, frame_annot2))
        side_by_side_orig = np.hstack((frame1, frame2))

        side_by_side_annot_out.write(side_by_side_annot)
        side_by_side_orig_out.write(side_by_side_orig)

        frame_idx += 1

    # Release all resources
    cap1.release()
    cap2.release()
    cap_annot1.release()
    cap_annot2.release()
    overlay_out.release()
    side_by_side_annot_out.release()
    side_by_side_orig_out.release()


    

#------------------------------Established FUnctions below---------------------------------------------

def detect_swing_boundaries(keypoints):
    """Detect start and end indices of tennis swings based on wrist position."""
    hip_threshold=0.08
    prep_ignore_interval = 75
    wrist_in_front_frames = []  # List to keep track of frames when wrists are in front
    feet_threshold = 0.1

    # Look for key aspects of tennis swing based on wrist position
    for i, frame_keypoints in enumerate(keypoints):
        if frame_keypoints is None:
            continue

        # Get the keypoints for the wrists and hips
        right_wrist = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value][:2])
        left_wrist = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value][:2])
        right_hip = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value][:2])
        left_hip = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_HIP.value][:2])
        right_foot = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value][:2])
        left_foot = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][:2])



        # Calculate the hip center
        hip_center = np.array([(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2])

        # Calculate distances from wrists to hip center
        left_wrist_distance = np.linalg.norm(left_wrist - hip_center)
        right_wrist_distance = np.linalg.norm(right_wrist - hip_center)

        # Check if both wrists are within the distance threshold
        wrists_in_front = left_wrist_distance < hip_threshold and right_wrist_distance < hip_threshold

        # Calculate distance between the feet
        feet_distance = abs(left_foot[0] - right_foot[0])
        
        # Check if the feet are wide enough
        feet_wide = feet_distance > feet_threshold
        
       # Record the frame if wrists are in front and conditions are met
        if wrists_in_front and feet_wide:
            if not wrist_in_front_frames or (i - wrist_in_front_frames[-1] >= prep_ignore_interval):
                wrist_in_front_frames.append(i)
                print(f"----------prep position at {i/30:.1f} seconds---------")
                print(f"{i/30:.1f} seconds & frame{i}: {left_wrist_distance:.4f},{right_wrist_distance:.4f}/{hip_threshold} hip threshold")
                print(f"{i/30:.1f} seconds & frame{i}: {feet_distance} feet distance\n")
            
        # Debugging: Print comparison for wrists together
        #print(f"{i/30:.1f} seconds & frame{i}: {left_wrist_distance:.4f},{right_wrist_distance:.4f}/{hip_threshold} hip threshold")
        #print(f"{i/30:.1f} seconds & frame{i}: {feet_distance} feet distance\n")
    
    '''
     wrists together calculation
        wrist_threshold=0.05
        
        # Check if wrists are together (x-coordinates close to each other)
        wrists_together = abs(right_wrist[0] - left_wrist[0]) < wrist_threshold
       
        
        # Debugging: Print comparison for wrists together
        print(f"Second {i/30:.1f}: {abs(right_wrist[0] - left_wrist[0]):.4f}/{wrist_threshold} wrists together threshold")
        
        wrist_in_front = wrists_together and wrists_in_front
        
        # Detect swing start (when both wrists are in front of the body)
        if wrist_in_front and not in_swing:
            in_swing = True
            start = i
            print(f"Swing start detected at frame {i} (Both wrists in front of body)")

        # Detect swing end (when both wrists return to in front of the body again)
        elif wrist_in_front and in_swing:
            # Swing ends when both wrists are back in front of the body again
            motions.append((start, i))
            print(f"Swing end detected at frame {i} (Both wrists returned to front of body)")
            in_swing = False  # Reset for next swing
    '''

    return wrist_in_front_frames



def segment_swings(swing_frames):
    """
    Segments the keypoints into swings based on detected swing frames.
    
    Args:
        keypoints (list): List of keypoints for each frame.
        swing_frames (list): List of frame indices marking swing boundaries.
        
    Returns:
        list: List of tuples representing start and end frame indices for swings.
    """
    if not swing_frames or len(swing_frames) < 2:
        return []

    swings = [(swing_frames[i], swing_frames[i + 1]) for i in range(len(swing_frames) - 1)]
    return swings


def extract_motion_segment(video_path, start_frame, end_frame, output_path):
    """Extract a segment of the video based on frame indices."""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    if start_frame < 0 or end_frame >= total_frames or start_frame >= end_frame:
        print(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}, total_frames={total_frames}")
        return

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Jump to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    print(f"Extracting video segment: {start_frame} to {end_frame}. {(end_frame-start_frame)//30} seconds long")
    
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_idx} could not be read, ending extraction.")
            break
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Segment saved to {output_path}")
    
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


"""Extract and annotate pose keypoints from a video."""
def extract_and_annotate(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video extract_and_annotate.")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        blank_frame = np.zeros((height, width, 3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x, y, z, visibility = landmark.x, landmark.y, landmark.z, landmark.visibility
                cv2.circle(blank_frame, (int(x * width), int(y * height)), 5, (0, 255, 0), -1)

        out.write(blank_frame)
        cv2.imshow('Annotated Video', blank_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()



def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                # Collect keypoints (only the first two dimensions for position)
                frame_keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
                keypoints.append(frame_keypoints)
    
    cap.release()
    return np.array(keypoints)


def compare_swing_features(user_features, pro_features):
    """Compare the swing features of user and pro."""
    wrist_distance = 0
    elbow_distance = 0
    shoulder_distance = 0
    hip_distance = 0
    
    # Assuming the swing features are lists of positions
    # Compare each feature (wrist, elbow, etc.) by calculating Euclidean distance
    for user_wrist, pro_wrist in zip(user_features[0], pro_features[0]):
        wrist_distance += np.linalg.norm(user_wrist[0] - pro_wrist[0]) + np.linalg.norm(user_wrist[1] - pro_wrist[1])
    
    for user_elbow, pro_elbow in zip(user_features[1], pro_features[1]):
        elbow_distance += np.linalg.norm(user_elbow[0] - pro_elbow[0]) + np.linalg.norm(user_elbow[1] - pro_elbow[1])
    
    for user_shoulder, pro_shoulder in zip(user_features[2], pro_features[2]):
        shoulder_distance += np.linalg.norm(user_shoulder[0] - pro_shoulder[0]) + np.linalg.norm(user_shoulder[1] - pro_shoulder[1])
    
    for user_hip, pro_hip in zip(user_features[3], pro_features[3]):
        hip_distance += np.linalg.norm(user_hip[0] - pro_hip[0]) + np.linalg.norm(user_hip[1] - pro_hip[1])

    total_distance = wrist_distance + elbow_distance + shoulder_distance + hip_distance

    # Provide feedback based on distance
    feedback = ""
    if total_distance < 1000:  # This threshold can be adjusted based on how far you expect the swings to be
        feedback = "Your swing is very similar to the pro swing!"
    else:
        feedback = "There are significant differences between your swing and the pro swing. Work on improving your wrist and elbow movements."

    return total_distance, feedback


def extract_swing_features(keypoints):
    """Extract features from keypoints for the given swing segment."""
    wrist_positions = []
    elbow_positions = []
    shoulder_positions = []
    hip_positions = []
    
    for frame_keypoints in keypoints:
        if frame_keypoints is None:
            continue

        # Get the keypoints for the wrists, elbows, shoulders, and hips
        right_wrist = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value][:2])
        left_wrist = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value][:2])
        right_elbow = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value][:2])
        left_elbow = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value][:2])
        right_shoulder = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][:2])
        left_shoulder = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value][:2])
        right_hip = np.array(frame_keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value][:2])
        left_hip = np.array(frame_keypoints[mp_pose.PoseLandmark.LEFT_HIP.value][:2])

        # Store the keypoints for analysis
        wrist_positions.append((right_wrist, left_wrist))
        elbow_positions.append((right_elbow, left_elbow))
        shoulder_positions.append((right_shoulder, left_shoulder))
        hip_positions.append((right_hip, left_hip))

    # Return the features as a tuple of lists
    return wrist_positions, elbow_positions, shoulder_positions, hip_positions

'''-----------------------------------------------------Begin Driver-----------------------------------------------------'''
# Example Usage
'''
videos = ["Federer.mov"]
for video in videos:
    video_path = 'videos/' + video  # Path to user video
    output_path = f'results/{video}/'
    clear_output_folder(output_path)
    
    # Extract keypoints from the video
    keypoints = extract_keypoints(video_path)

    # Detect swing boundaries
    swing_frames = detect_swing_boundaries(keypoints)

    # Segment swings based on the detected frames
    swing_segments = segment_swings(swing_frames)

    print(f"Detected {len(swing_segments)} swings.")

    # Process each swing and extract video segments
    for idx, (start_frame, end_frame) in enumerate(swing_segments):
        if start_frame >= 0 and end_frame > start_frame:
            extract_motion_segment(video_path, start_frame, end_frame, f'results/{video}/swing{idx+1}.mov')
 '''
    
# Paths to your videos
folder = 'compare/'
extension = '.mov'
vid1 = 'me_horizontal'
vid2 = 'swing2'
video1 = folder + vid1 + extension
video2 = folder + vid2 + extension

#extract_and_annotate("videos/Federer1.mov", 'results/Federer1.mov')

'''
# Extract keypoints for each video (replace with actual function)
my_swing_keypoints = extract_keypoints_from_video(video1)
pro_swing_keypoints = extract_keypoints_from_video(video2)

# Extract swing features (for comparison and feedback)
my_swing_features = extract_swing_features(my_swing_keypoints)
pro_swing_features = extract_swing_features(pro_swing_keypoints)
'''

annotated_overlay = folder + vid1 + '_' + vid2 + '_overlay.mp4'
original_side_by_side = folder + vid1  + '_' +  vid2 + '_comapre.mp4'

#create_aligned_overlay_video(video1, video2, output_video)

# Call the function to sync the timing and overlay the videos
sync_video_timings(video1, video2, annotated_overlay, original_side_by_side)