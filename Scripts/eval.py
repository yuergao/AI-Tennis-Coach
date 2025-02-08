from segment import detect_swing_boundaries
from video_utils import extract_keypointsv2_asimg, extract_keypoints_from_video, extract_keypointsv3_asimg, extract_keypointsv4_asimg, extract_keypointsv5_asimg, extract_keypointsv6_asimg
import pandas as pd
import time
import cv2
import argparse

class Profiler:
    def __init__(self, print_name=False):
        self.start_time = None
        self.end_time = None
        self.name = None
        self.print_name = print_name
        self.record = []
        self.video_duration = None
    def start(self, name=None, video_duration=None):
        """
        Start the profiler.
        """
        self.start_time = time.time()
        self.end_time = None  # Reset end time in case the profiler is reused
        self.name = name
        self.video_duration = video_duration

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
        if self.video_duration:
            self.record.append((self.video_duration, elapsed))

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
    # keypoints, fps = extract_keypoints_from_video(video_path)
    keypoints, fps = extract_keypointsv6_asimg(video_path)
    # detect swing boundaries
    forehand_segments, backhand_segments = detect_swing_boundaries(keypoints, fps)
    segments = [*forehand_segments, *backhand_segments] # (start_frame, end_frame), need to add stroke_cat
    segments = [(frame_idx_to_ms(start_frame, fps), frame_idx_to_ms(end_frame, fps), "f" if i < len(forehand_segments) else "b") for i, (start_frame, end_frame) in enumerate(segments)]
    return segments

def single_match(gt_segments, pred_segments, min_iou=0.1):
    matched_segments = {}
    unmatched_pred_segments = []
    unmatched_gt_segments = []
    for pred_seg_idx, segment in enumerate(pred_segments):
        pred_cat = segment[-1]
        pred_segment_time = segment[:-1]
        gt_segment_idx, iou = segment_gt_matching(pred_segment_time, gt_segments)
        if gt_segment_idx not in matched_segments and iou >= min_iou:
            matched_segments[gt_segment_idx] = {'pred_seg_idx': pred_seg_idx, 'highest_iou': iou}

        elif gt_segment_idx in matched_segments and matched_segments[gt_segment_idx]['highest_iou'] < iou:
            unmatched_pred_segments.append(matched_segments[gt_segment_idx]['pred_seg_idx'])
            matched_segments[gt_segment_idx] = {'pred_seg_idx': pred_seg_idx, 'highest_iou': iou}

    for gt_seg_idx, segment in enumerate(gt_segments):
        if gt_seg_idx not in matched_segments:
            unmatched_gt_segments.append(gt_seg_idx)

    return matched_segments, unmatched_pred_segments, unmatched_gt_segments

def classification_metrics(time_profiles, matched_segments, gt_stroke_cats, pred_segments, verbose=False):
    # iterate over matched segments dict
    class_count_dict = {'f':0, 'b':0}
    class_count_dict_gt = {'f':0, 'b':0}
    for gt_seg_idx, pred_dict in matched_segments.items():
        gt_segment = gt_stroke_cats[gt_seg_idx]
        pred_segment = pred_segments[pred_dict['pred_seg_idx']]
        # check if the classification is correct
        if gt_segment == pred_segment[-1]:
            class_count_dict[gt_segment] += 1
        class_count_dict_gt[gt_segment] += 1


    # print the classification precision
    
    for key in class_count_dict:
        if class_count_dict_gt[key] == 0:
            # precision_dict[key] = 0
            if verbose:
                print("No precision calculated for class, due to no gt segments for", key)
        else:
            # precision_dict[key] = class_count_dict[key]/class_count_dict_gt[key]
            if verbose:
                print(f"Precision for {key}: {class_count_dict[key]/class_count_dict_gt[key]:.2f}")
    return [class_count_dict, class_count_dict_gt]

def compute_metrics(time_profiles, gt_segments, gt_stroke_cats, pred_segments, matching_algorithm, verbose=False):
    # How fast is our system?
    # Average speed -> FPS (frames processed per second)
    avg_speed = sum([x[0] for x in time_profiles.record])  / sum([x[1] for x in time_profiles.record]) 
    if verbose:
        print(f"Average processing time {avg_speed:.2f} frames/second")
    results = {}

    for iou in [0.1, 0.3, 0.5]:
        if verbose:
            print(f"---------------------- IOU THRESHOLD: {iou} ----------------------")
        matched_segments, unmatched_pred_segments, unmatched_gt_segments = matching_algorithm(gt_segments, pred_segments, min_iou = iou)

        matched_gt_percent = 100*len(matched_segments) / len(gt_segments)
        matched_pred_percent = 100 * len(matched_segments) / len(pred_segments)
        delta = len(pred_segments) - len(gt_segments)

        avg_pred_segment_length = sum([x[1] - x[0] for x in pred_segments]) / len(pred_segments)
        avg_gt_segment_length = sum([x[1] - x[0] for x in gt_segments]) / len(gt_segments)

        
        if verbose:

            # How many ground truth segments get matched (at different thresholds iou)? 
            print(f"Number of ground truth segments matched at iou {iou}: {matched_gt_percent}\%")
            # How many predicted segments have an overlap with the ground truth segment  (at different thresholds iou)?
            print(f"Number of predicted segments matched at iou {iou}: {matched_pred_percent}\%")
            # # of segments predicted vs ground truth
            print(f"# of segments predicted: {len(pred_segments)} vs # of ground truth segments: {len(gt_segments)}")
            # Observe the delta (this will point to an undersegmentation problem or oversegmentation problem)
            print(f"Delta: {len(pred_segments) - len(gt_segments)}")
            print(f"Average predicted segment length: {avg_pred_segment_length:.2f} vs average ground truth segment length: {avg_gt_segment_length:.2f}")

        # How accurate is the classification?
        precision_count_dict = classification_metrics(time_profiles, matched_segments, gt_stroke_cats, pred_segments, verbose=verbose)

        # Store results for this IoU threshold
        results[iou] = {
            "total_matched": len(matched_segments),
            "num_pred_segments": len(pred_segments),
            "num_gt_segments": len(gt_segments),
            "delta_per_video": delta,
            "avg_pred_segment_length": avg_pred_segment_length*len(pred_segments),
            "avg_gt_segment_length": avg_gt_segment_length*len(gt_segments),
            "precision_count_dict": precision_count_dict
        }
    return avg_speed, results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Play a video at 0.5x speed and show timestamps.")
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    parser.add_argument("--annotation_path", default='../Annotations/annotations.csv', type=str, help="Path to the video file")

    args = parser.parse_args()

    # annotation_path = '../Annotations/annotations.csv'
    # path_to_videos = '../Annotations/videos'
    annot_df = pd.read_csv(args.annotation_path) # video_path	start_time	end_time	stroke_cat
    profiler = Profiler()
    processing_times = []
    results = []
    speeds = []
    aggregated_results = {iou: {
        "total_matched": 0,
        "num_pred_segments": 0,
        "num_gt_segments": 0,
        "delta": 0,
        "avg_pred_segment_length": 0,
        "avg_gt_segment_length": 0,
        "correct_pred_counts": [],  # This needs special handling
        "gt_class_counts": []  # This needs special handling
    } for iou in [0.1, 0.3, 0.5]}

    for video_path in annot_df['video_path'].unique():    

        gt_segments = [(convert_ts_ms(row['start_time']), convert_ts_ms(row['end_time'])) for _, row in annot_df[annot_df['video_path'] == video_path].iterrows()]
        # get the first letter of the stroke category
        gt_stroke_cats = [row['stroke_cat'][0] for _, row in annot_df[annot_df['video_path'] == video_path].iterrows()]

        video_path=f"{args.video_path}/{video_path}"
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        profiler.start(video_duration=total_frames)
        pred_segments = get_segments(video_path)
        processing_times.append(profiler.stop())
        print("processing time:", processing_times[-1])

        avg_speed, results_for_video = compute_metrics(profiler, gt_segments, gt_stroke_cats, pred_segments, single_match, verbose=False)
        speeds.append(avg_speed)

        for iou in [0.1, 0.3, 0.5]:
            if iou in results_for_video:
                aggregated_results[iou]["total_matched"] += results_for_video[iou]["total_matched"]
                aggregated_results[iou]["num_pred_segments"] += results_for_video[iou]["num_pred_segments"]
                aggregated_results[iou]["num_gt_segments"] += results_for_video[iou]["num_gt_segments"]
                aggregated_results[iou]["delta"] += results_for_video[iou]["delta_per_video"]
                aggregated_results[iou]["avg_pred_segment_length"] += results_for_video[iou]["avg_pred_segment_length"]
                aggregated_results[iou]["avg_gt_segment_length"] += results_for_video[iou]["avg_gt_segment_length"]
                aggregated_results[iou]["correct_pred_counts"].append(results_for_video[iou]["precision_count_dict"][0])
                aggregated_results[iou]["gt_class_counts"].append(results_for_video[iou]["precision_count_dict"][1])

        results.append(results_for_video)
        
    
    print("Average throughput(fps):", sum(speeds) / len(speeds))
    print(f"Average predicted segment length: {aggregated_results[0.1]['avg_pred_segment_length'] / aggregated_results[0.1]['num_pred_segments']:.2f} vs average ground truth segment length: {aggregated_results[iou]['avg_gt_segment_length'] / aggregated_results[iou]['num_gt_segments']:.2f}")
    print(f"# of segments predicted: {aggregated_results[0.1]['num_pred_segments']} vs # of ground truth segments: {aggregated_results[0.1]['num_gt_segments']}")
    print(f"Delta: {aggregated_results[0.1]['delta']}")

    for iou in [0.1, 0.3, 0.5]:
        print(f"---------------------- IOU THRESHOLD: {iou} ----------------------")
        print(f"Percent of ground truth segments matched at iou {iou}: {100 * aggregated_results[iou]['total_matched']/ aggregated_results[iou]['num_gt_segments']}\%")
        print(f"Percent of predicted segments matched at iou {iou}: {100 * aggregated_results[iou]['total_matched'] / aggregated_results[iou]['num_pred_segments']}\%")
        print("Precision counts:")
        for key in ['f', 'b']:
            # breakpoint()
            precision_num = sum([x[key] for x in aggregated_results[iou]['correct_pred_counts']])
            prec_denom = sum([x[key] for x in aggregated_results[iou]['gt_class_counts']])
            if prec_denom == 0:
                print("No precision calculated for class, due to no gt segments for", key)
            else:
                print(f"Precision for {key}: {precision_num/prec_denom:.2f}")
        
        print("\n")
    