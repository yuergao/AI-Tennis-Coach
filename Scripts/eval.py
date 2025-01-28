from segment import detect_swing_boundaries
from video_utils import extract_keypoints_from_video
import pandas as pd

def calculated_iou(segment1, segment2):
    pass

def segment_gt_matching(segment, gt_segments):

def get_segments(video_path):
    # load video
    # extract keypoints
    keypoints, fps = extract_keypoints_from_video(video_path)
    # detect swing boundaries
    forehand_segments, backhand_segments = detect_swing_boundaries(keypoints, fps)
    breakpoint()
    

if __name__ == '__main__':
    annotation_path = 'Annotations/annotations.csv'
    annot_df = pd.read_csv(annotation_path) # video_path	start_time	end_time	stroke_cat
    for video_path in annot_df['video_path'].unique():
        # todo 
        break