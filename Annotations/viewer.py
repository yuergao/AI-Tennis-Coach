import cv2
import time
import argparse

def format_timestamp(milliseconds):
    """Convert milliseconds to hh:mm:ss:ms format."""
    seconds, ms = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{int(ms):03}"

def play_video_slow(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = int(1000 / fps)  # Frame duration in milliseconds

    # Slow playback speed (e.g., 0.5x speed)
    playback_speed = 0.25
    slow_duration = int(frame_duration / playback_speed)

    paused = False
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            

            # Display the video frame
            cv2.imshow("Video Playback", frame)

        # Handle keyboard inputs
        key = cv2.waitKey(slow_duration)
        if key == ord('q'):  # Quit the video
            break
        elif key == ord('s'):
            print("pressed s")
            paused = True
            current_timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            formatted_time = format_timestamp(current_timestamp)
            print(f"Start Timestamp: {formatted_time}")
        elif key == ord('f'):
            print("pressed f")
            print(f"forehand")
        elif key == ord('b'):
            print("pressed b")
            print(f"backhand")
        elif key == ord('e'):
            print("pressed e")
            paused = True
            current_timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            formatted_time = format_timestamp(current_timestamp)
            print(f"End Timestamp: {formatted_time}")
        elif key == ord('p'):  # Pause or unpause the video
            paused = not paused
            # Get the current timestamp in milliseconds
            if paused:
                current_timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                formatted_time = format_timestamp(current_timestamp)
                
                # print(f"Timestamp: {formatted_time}")

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # ex: python viewer.py --video_path videos/Amateur/insta1.MP4
    # hit p and q to pause/resume and quit video
    parser = argparse.ArgumentParser(description="Play a video at 0.5x speed and show timestamps.")
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    # Play the video
    play_video_slow(args.video_path)

