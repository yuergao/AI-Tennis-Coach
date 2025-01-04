import numpy as np
from mediapipe import solutions as mp_solutions
import cv2

"""
--------------------- EXTRA UNUSED FUNCTIONS FILE FOR ARCHIVE PURPOSES --------------------
"""

mp_pose = mp_solutions.pose
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