"""
Utility functions for the GUI.
"""
from typing import Optional
import numpy as np


def format_time(total_seconds: float) -> str:
    """Format time as MM:SS.mmm"""
    ms = int((total_seconds - int(total_seconds)) * 1000)
    m = int(total_seconds // 60)
    s = int(total_seconds % 60)
    return f"{m:02d}:{s:02d}.{ms:03d}"


def format_time_extended(total_seconds: float) -> str:
    """Format time as HH:MM:SS.mmm"""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    millis = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Convert seconds to frame count."""
    return int(seconds * fps)


def frames_to_seconds(frames: int, fps: float) -> float:
    """Convert frame count to seconds."""
    return frames / fps if fps > 0 else 0.0


def get_gaze_columns(df) -> tuple:
    """
    Find gaze x, y columns in a DataFrame.
    Returns: (gaze_x_col, gaze_y_col) or (None, None) if not found.
    """
    gaze_x_col = None
    gaze_y_col = None
    
    for col in df.columns:
        lc = col.lower()
        if 'gaze x' in lc or 'gaze_x' in lc or (lc == 'x'):
            gaze_x_col = col
        if 'gaze y' in lc or 'gaze_y' in lc or (lc == 'y'):
            gaze_y_col = col
    
    return gaze_x_col, gaze_y_col


def find_closest_timestamp_index(timestamp_values: np.ndarray, target_ts: int, 
                                  tolerance_ns: int = int(1e8)) -> Optional[int]:
    """
    Find index of closest timestamp within tolerance.
    
    Args:
        timestamp_values: Array of nanosecond timestamps
        target_ts: Target timestamp to find
        tolerance_ns: Maximum allowed difference in nanoseconds
    
    Returns:
        Index of closest match, or None if nothing within tolerance
    """
    if timestamp_values is None or len(timestamp_values) == 0:
        return None
    
    diffs = np.abs(timestamp_values - target_ts)
    min_diff = np.min(diffs)
    
    if min_diff > tolerance_ns:
        return None
    
    return int(np.argmin(diffs))


def scale_coordinates(coords: tuple, from_size: tuple, to_size: tuple) -> tuple:
    """
    Scale coordinates from one size to another (useful for GUI scaling).
    
    Args:
        coords: (x, y) coordinates
        from_size: (width, height) of original size
        to_size: (width, height) of new size
    
    Returns:
        Scaled (x, y) coordinates
    """
    if from_size[0] == 0 or from_size[1] == 0:
        return coords
    
    scale_x = to_size[0] / from_size[0]
    scale_y = to_size[1] / from_size[1]
    
    return (int(coords[0] * scale_x), int(coords[1] * scale_y))
