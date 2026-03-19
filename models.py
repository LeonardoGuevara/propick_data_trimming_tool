"""
Core data models for the Eye Tracking Data Synchronization Tool.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class SensorData:
    """Represents a sensor data file (CSV/Excel)."""
    name: str
    file_path: Path
    data: pd.DataFrame
    format_type: str  # 'csv' or 'excel'
    timestamp_col: Optional[str] = None
    start_timestamp_col: Optional[str] = None
    end_timestamp_col: Optional[str] = None
    
    def get_time_range(self) -> Tuple[int, int]:
        """Get min and max timestamps in the data."""
        if self.timestamp_col and self.timestamp_col in self.data.columns:
            valid = self.data[self.timestamp_col].dropna()
            return int(valid.min()), int(valid.max())
        elif self.start_timestamp_col and self.start_timestamp_col in self.data.columns:
            valid = self.data[self.start_timestamp_col].dropna()
            return int(valid.min()), int(valid.max())
        return 0, 0


@dataclass
class VideoData:
    """Represents a video file with metadata."""
    name: str
    file_path: Path
    is_primary: bool  # True for eye tracking video, False for additional videos
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float  # in seconds
    world_timestamps: Optional[np.ndarray] = None  # per-frame timestamps in nanoseconds
    offset_frames: int = 0  # Frame offset relative to primary video (for sync)
    start_frame: int = 0  # Frame where trimming starts
    end_frame: Optional[int] = None  # Frame where trimming ends
    camera_type: Optional[str] = None  # Camera identifier: "eyes" for primary, or user-specified for additional
    
    def get_frame_timestamp(self, frame_idx: int) -> int:
        """Get nanosecond timestamp for a frame."""
        if self.world_timestamps is not None and 0 <= frame_idx < len(self.world_timestamps):
            return int(self.world_timestamps[frame_idx])
        # Fallback: compute from fps
        return int((frame_idx / self.fps) * 1e9)
    
    def get_trimmed_range(self) -> Tuple[int, int]:
        """Get frame range for trimming."""
        end = self.end_frame if self.end_frame is not None else self.frame_count - 1
        return self.start_frame, end
    
    def get_trimmed_frame_count(self) -> int:
        """Get number of frames in trimmed range."""
        start, end = self.get_trimmed_range()
        return end - start + 1


@dataclass
class ProjectData:
    """Represents a complete project with all loaded data."""
    name: str
    project_folder: Path
    primary_video: Optional[VideoData] = None
    additional_videos: List[VideoData] = field(default_factory=list)
    sensor_data: Dict[str, SensorData] = field(default_factory=dict)
    
    def get_all_videos(self) -> List[VideoData]:
        """Get primary and all additional videos."""
        videos = []
        if self.primary_video:
            videos.append(self.primary_video)
        videos.extend(self.additional_videos)
        return videos
    
    def get_trimmed_time_range(self) -> Tuple[int, int]:
        """Get time range in nanoseconds for the trimmed primary video."""
        if not self.primary_video:
            return 0, 0
        start_frame, end_frame = self.primary_video.get_trimmed_range()
        start_ns = self.primary_video.get_frame_timestamp(start_frame)
        end_ns = self.primary_video.get_frame_timestamp(end_frame)
        return start_ns, end_ns
    
    def get_trimmed_duration(self) -> float:
        """Get duration of trimmed video in seconds."""
        if not self.primary_video:
            return 0.0
        start, end = self.primary_video.get_trimmed_range()
        return (end - start + 1) / self.primary_video.fps
