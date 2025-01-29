# Instructions for Annotation Interface

A simple Python script to play videos at **0.25x speed** with timestamp tracking.

## Features

- **Slow Playback (0.25x speed)**
- **Keyboard Controls:**
  - `p` → Pause/Resume
  - `q` → Quit
  - `s` → Mark **Start Timestamp** ALSO will pause
  - `e` → Mark **End Timestamp** ALSO will pause
  - `f` → Log **Forehand**
  - `b` → Log **Backhand**

## Usage

```bash
python viewer.py --video_path path/to/video.mp4
```

## Requirements

- OpenCV (`cv2`)
- Python 3.x

Install dependencies:

```bash
pip install opencv-python
```

## Sample Video

Check out the sample `viewer_demo.mov` file to see it in action!

