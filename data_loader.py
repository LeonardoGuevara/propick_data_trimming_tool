"""
Data loading utilities for video and sensor files.
"""
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np
import pandas as pd
from models import SensorData, VideoData


def read_data_file(file_path: Path) -> Tuple[pd.DataFrame, str]:
    """Read CSV or Excel file and return DataFrame with format type."""
    if file_path.suffix.lower() == '.csv':
        return pd.read_csv(str(file_path)), 'csv'
    else:
        return pd.read_excel(str(file_path)), 'excel'


def find_timestamp_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Identify timestamp columns in a DataFrame.
    Returns: (timestamp_col, start_timestamp_col, end_timestamp_col)
    """
    timestamp_col = None
    start_col = None
    end_col = None
    
    for col in df.columns:
        lc = col.lower()
        # Single timestamp column (for events, measurements)
        if 'timestamp' in lc and 'ns' in lc:
            if 'start' not in lc and 'end' not in lc:
                timestamp_col = col
        # Start timestamp (for events with duration)
        if 'start' in lc and 'timestamp' in lc:
            start_col = col
        # End timestamp
        if 'end' in lc and 'timestamp' in lc:
            end_col = col
    
    return timestamp_col, start_col, end_col


def load_video(video_path: Path, is_primary: bool = False) -> VideoData:
    """Load video file and extract metadata, world_timestamps if available."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    # Load world_timestamps if this is the primary video
    world_ts = None
    if is_primary:
        wt_candidates = list(video_path.parent.glob("world_timestamps*.csv"))
        if wt_candidates:
            wt_file = max(wt_candidates, key=lambda p: p.stat().st_mtime)
            try:
                wt_df, _ = read_data_file(wt_file)
                ts_col, _, _ = find_timestamp_columns(wt_df)
                if ts_col:
                    wt_df[ts_col] = pd.to_numeric(wt_df[ts_col], errors='coerce')
                    wt_df = wt_df[wt_df[ts_col].notna()].reset_index(drop=True)
                    if len(wt_df) > 0:
                        world_ts = wt_df[ts_col].to_numpy(dtype='float64')
            except Exception as e:
                print(f"Warning: Could not load world_timestamps: {e}")
    
    return VideoData(
        name=video_path.stem,
        file_path=video_path,
        is_primary=is_primary,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration=duration,
        world_timestamps=world_ts,
        start_frame=0,
        end_frame=frame_count - 1
    )


def load_sensor_data(file_path: Path) -> SensorData:
    """Load CSV or Excel sensor data file."""
    df, fmt = read_data_file(file_path)
    
    # Identify timestamp columns
    ts_col, start_col, end_col = find_timestamp_columns(df)
    
    # Convert timestamp columns to numeric
    for col in [ts_col, start_col, end_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return SensorData(
        name=file_path.stem,
        file_path=file_path,
        data=df,
        format_type=fmt,
        timestamp_col=ts_col,
        start_timestamp_col=start_col,
        end_timestamp_col=end_col
    )


def find_eye_tracking_files(folder: Path) -> dict:
    """
    Scan folder for eye tracking data files.
    If the selected folder does not contain exported CSV files directly,
    automatically try the newest child folder inside ``exports/``.

    Returns dict with keys:
    'videos', 'fixations', 'saccades', 'gaze', 'gaze_positions',
    'imu', 'world_timestamps', 'blinks', '3d_eye_states', 'resolved_folder'.
    """

    def _empty_results(resolved_folder: Path) -> dict:
        return {
            'videos': [],
            'fixations': None,
            'saccades': None,
            'gaze': None,
            'gaze_positions': None,
            'imu': None,
            'world_timestamps': None,
            'blinks': None,
            '3d_eye_states': None,
            'resolved_folder': resolved_folder,
        }

    def _pick_best_match(candidates: List[Path]) -> Optional[Path]:
        if not candidates:
            return None
        # Prefer exact stem match (e.g. fixations.csv) over suffixed variants,
        # then newest file as tie-breaker.
        return sorted(candidates, key=lambda p: (len(p.stem), -p.stat().st_mtime))[0]

    def _collect_from_folder(target_folder: Path) -> dict:
        res = _empty_results(target_folder)

        # Find videos and prioritize world.mp4 for primary eye-tracking stream
        videos = sorted(target_folder.glob('*.mp4'))
        world_videos = [v for v in videos if v.stem.lower() == 'world']
        non_world = [v for v in videos if v.stem.lower() != 'world']
        res['videos'] = world_videos + non_world

        sensor_patterns = {
            'fixations': ['fixations.csv', 'fixations*.csv', 'fixations.xlsx', 'fixations.xls'],
            'saccades': ['saccades.csv', 'saccades*.csv', 'saccades.xlsx', 'saccades.xls'],
            'gaze': ['gaze.csv', 'gaze*.csv', 'gaze.xlsx', 'gaze.xls'],
            'gaze_positions': ['gaze_positions.csv', 'gaze_positions*.csv', 'gaze_positions.xlsx', 'gaze_positions.xls'],
            'imu': ['imu.csv', 'imu*.csv', 'imu.xlsx', 'imu.xls'],
            'world_timestamps': ['world_timestamps.csv', 'world_timestamps*.csv', 'world_timestamps.xlsx', 'world_timestamps.xls'],
            'blinks': ['blinks.csv', 'blinks*.csv', 'blinks.xlsx', 'blinks.xls'],
            '3d_eye_states': ['3d_eye_states.csv', '3d_eye_states*.csv', '3d_eye_states.xlsx', '3d_eye_states.xls'],
        }

        for key, patterns in sensor_patterns.items():
            matches: List[Path] = []
            for pattern in patterns:
                matches.extend(target_folder.glob(pattern))
            res[key] = _pick_best_match(matches)

        return res

    # First pass: selected folder directly
    direct = _collect_from_folder(folder)
    direct_sensor_count = sum(
        1 for k in ['fixations', 'saccades', 'gaze', 'gaze_positions', 'imu', 'world_timestamps', 'blinks', '3d_eye_states']
        if direct.get(k) is not None
    )
    if direct_sensor_count > 0:
        return direct

    # Fallback: common Neon structure <session>/exports/<id>/
    exports_dir = folder / 'exports'
    if exports_dir.exists() and exports_dir.is_dir():
        candidates = [d for d in exports_dir.iterdir() if d.is_dir()]
        if candidates:
            best = None
            best_score = -1
            for c in candidates:
                collected = _collect_from_folder(c)
                sensor_count = sum(
                    1 for k in ['fixations', 'saccades', 'gaze', 'gaze_positions', 'imu', 'world_timestamps', 'blinks', '3d_eye_states']
                    if collected.get(k) is not None
                )
                has_world = any(v.stem.lower() == 'world' for v in collected['videos'])
                score = sensor_count * 10 + (5 if has_world else 0) + len(collected['videos'])
                if score > best_score:
                    best_score = score
                    best = collected

            if best is not None and best_score > 0:
                return best

    return direct
