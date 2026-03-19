"""
Data loading utilities for video and sensor files.
"""
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np
import pandas as pd
import subprocess
from models import SensorData, VideoData


def _parse_fps_fraction(raw_value: str) -> Optional[float]:
    """Parse ffprobe FPS values like '30000/1001' with full precision."""
    try:
        value = (raw_value or '').strip()
        if not value or value in {'0/0', 'N/A'}:
            return None
        if '/' in value:
            num_str, den_str = value.split('/', 1)
            num = float(num_str)
            den = float(den_str)
            if den == 0:
                return None
            out = num / den
            return out if out > 0 else None
        out = float(value)
        return out if out > 0 else None
    except Exception:
        return None


def _probe_precise_fps(video_path: Path, fallback_fps: float) -> float:
    """Probe a stable precise FPS from ffprobe.

    When avg_frame_rate and r_frame_rate differ notably, prefer r_frame_rate (nominal
    stream rate) because it is usually more stable for frame-index based tools.
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=avg_frame_rate,r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return fallback_fps

        lines = [line.strip() for line in (result.stdout or '').splitlines() if line.strip()]
        avg = _parse_fps_fraction(lines[0]) if len(lines) >= 1 else None
        rfr = _parse_fps_fraction(lines[1]) if len(lines) >= 2 else None

        if avg and rfr:
            if abs(avg - rfr) <= 0.01:
                return avg
            return rfr
        if avg:
            return avg
        if rfr:
            return rfr
    except Exception:
        pass
    return fallback_fps


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


def load_video(video_path: Path, is_primary: bool = False, camera_type: Optional[str] = None) -> VideoData:
    """Load video file and extract metadata, world_timestamps if available.
    
    Args:
        video_path: Path to video file
        is_primary: True if this is the primary (eye tracking) video
        camera_type: Camera identifier. If None and is_primary=True, defaults to "eyes"
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    
    fps_cv = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = _probe_precise_fps(video_path, fps_cv)
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
    
    # Set camera type: primary videos always use "eyes", others use provided value or None
    final_camera_type = "eyes" if is_primary else camera_type
    
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
        end_frame=frame_count - 1,
        camera_type=final_camera_type
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
