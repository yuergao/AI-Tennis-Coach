from segment import detect_swing_boundaries
from video_utils import extract_keypoints_from_video
import pandas as pd
import time

class Profiler:
    def __init__(self, print_name=False):
        self.start_time = None
        self.end_time = None
        self.name = None
        self.print_name = print_name
    def start(self, name=None):
        """
        Start the profiler.
        """
        self.start_time = time.time()
        self.end_time = None  # Reset end time in case the profiler is reused
        self.name = name
    def stop(self):
        """
        Stop the profiler and print the elapsed time.
        """
        if self.start_time is None:
            raise RuntimeError("Profiler has not been started. Use `start()` before `stop()`.")
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.print_name:
            print(f"{self.name} Elapsed time: {elapsed:.2f} seconds")
        return elapsed

def convert_ts_ms(timestamp_str):
    # Convert hh:mm:ss:ms string format to ms
    timestamp_str = timestamp_str.strip().split(':')
    hours, minutes, seconds, ms = map(int, timestamp_str)
    return (hours*60*60*1000) + (minutes*60*1000) + (seconds*1000) + ms

    

def calculated_iou(segment1, segment2):
    '''
    Calculate the intersection over union of two segments.
    segment1: (start_time1, end_time1) in milliseconds
    segment2: (start_time2, end_time2) in milliseconds
    '''
    start_time1, end_time1 = segment1
    start_time2, end_time2 = segment2

    intersection = max(0, min(end_time1, end_time2) - max(start_time1, start_time2))
    union = max(end_time1, end_time2) - min(start_time1, start_time2)
    return intersection / union

def segment_gt_matching(segment, gt_segments):
    highest_iou = 0
    matched_segment = None
    for gt_segment_idx, gt_segment in enumerate(gt_segments):
        iou = calculated_iou(segment, gt_segment)
        if iou > highest_iou:
            highest_iou = iou
            matched_segment = gt_segment_idx
    return matched_segment, highest_iou

# evaluate segmentation ability
def frame_idx_to_ms(frame_idx, fps):
    return frame_idx * 1000 / fps

def get_segments(video_path):
    # load video
    # extract keypoints
    keypoints, fps = extract_keypoints_from_video(video_path)
    # detect swing boundaries
    forehand_segments, backhand_segments = detect_swing_boundaries(keypoints, fps)
    segments = [*forehand_segments, *backhand_segments] # (start_frame, end_frame), need to add stroke_cat
    segments = [(frame_idx_to_ms(start_frame, fps), frame_idx_to_ms(end_frame, fps), "f" if i < len(forehand_segments) else "b") for i, (start_frame, end_frame) in enumerate(segments)]
    return segments

if __name__ == '__main__':
    annotation_path = '../Annotations/annotations.csv'
    path_to_videos = '../Annotations/videos'
    annot_df = pd.read_csv(annotation_path) # video_path	start_time	end_time	stroke_cat
    profiler = Profiler()
    processing_times = []
    for video_path in annot_df['video_path'].unique():
        # todo 
        gt_segments = [(convert_ts_ms(row['start_time']), convert_ts_ms(row['end_time'])) for _, row in annot_df[annot_df['video_path'] == video_path].iterrows()]
        gt_stroke_cats = [row['stroke_cat'] for _, row in annot_df[annot_df['video_path'] == video_path].iterrows()]
        # breakpoint()
        profiler.start()
        pred_segments = get_segments(f"{path_to_videos}/{video_path}") # TODO need to swap this with the latest segmenter
        processing_times.append(profiler.stop())
        print("processing time:", processing_times[-1])
        # breakpoint()
        # matching process
        matched_segments = {}
        unmatched_pred_segments = []
        unmatched_gt_segments = []
        for pred_seg_idx, segment in enumerate(pred_segments):
            pred_cat = segment[-1]
            pred_segment_time = segment[:-1]
            gt_segment_idx, iou = segment_gt_matching(pred_segment_time, gt_segments)
            if gt_segment_idx not in matched_segments:
                matched_segments[gt_segment_idx] = {'pred_seg_idx': pred_seg_idx, 'highest_iou': iou}

            elif matched_segments[gt_segment_idx]['highest_iou'] < iou:
                unmatched_pred_segments.append(matched_segments[gt_segment_idx]['pred_seg_idx'])
                matched_segments[gt_segment_idx] = {'pred_seg_idx': pred_seg_idx, 'highest_iou': iou}

        for gt_seg_idx, segment in enumerate(gt_segments):
            if gt_seg_idx not in matched_segments:
                unmatched_gt_segments.append(gt_seg_idx)
        # compute metrics like that quantify undersegmentation and misclassification and precision (+200ms should be okay error)
        break