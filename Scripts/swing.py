# swing.py
from dataclasses import dataclass
from typing import Tuple, Literal


SwingType = Literal["forehand", "backhand"]

@dataclass
class Swing:
    
    type: SwingType  # "forehand" or "backhand"
    start_frame: int  # Start timestamp
    end_frame: int  # End timestamp
    padded_start_frame: int
    padded_end_frame: int
    avg_wrist_velocity: float
    movement: float  # left (positive) right (negative)
    fps: int

    def __post_init__(self):
        """Validate that start_frame is less than end_frame."""
        if self.start_frame >= self.end_frame:
            raise ValueError(f"start_frame ({self.start_frame}) must be less than end_frame ({self.end_frame})")
        if self.padded_start_frame >= self.padded_end_frame:
            raise ValueError(f"start_frame ({self.padded_start_frame}) must be less than end_frame ({self.end_frame})")
        
    def get_swing_len(self) -> float:
        """Returns the duration of the swing in seconds."""
        return round((self.end_frame - self.start_frame)/ self.fps, 1)
    
    def get_total_len(self) -> float:
        """Returns the duration of the swing in seconds."""
        return round((self.padded_end_frame - self.padded_start_frame)/ self.fps, 1)
    
    def print_info(self) -> None:
        """Prints detailed swing information."""
        print(f"{self.type} frames: {self.padded_start_frame} {self.padded_end_frame} for {self.get_total_len()} seconds Movement: {self.movement:.1f} avg_wrist_velocity: {self.avg_wrist_velocity:.1f}")
        
        
    def to_sec(self, timestamp: int) -> float:
        """Convert a timestamp (frame count) to seconds using stored FPS."""
        return round(timestamp / self.fps, 1)
    
    
    def is_within_padded_range(self, timestamp: int) -> bool:
        """Check if a given timestamp (frame number) falls within the padded start and padded end."""
        return self.padded_start_frame <= timestamp <= self.padded_end_frame
