"""
Data export utilities for saving trimmed videos and sensor data.
"""
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import os
import shutil
import subprocess
import re
from models import ProjectData, VideoData, SensorData
from data_synchronizer import (
    export_trimmed_video_ffmpeg,
    export_trimmed_video_ffmpeg_reencode,
    create_gaze_overlay_video,
)


SENSOR_EXPORT_NAME_MAP = {
    'gaze': 'gaze_positions',
    'world_timestamps': 'timestamps',
}


def _sanitize_filename_part(value: Optional[str], default: str = "unknown") -> str:
    """Return a filesystem-safe filename part."""
    raw = (value or '').strip()
    raw = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', raw)
    raw = re.sub(r'\s+', '_', raw)
    raw = raw.strip('._')
    return raw or default


def _sensor_export_name(sensor_name: str) -> str:
    """Map internal sensor keys to exported names."""
    return SENSOR_EXPORT_NAME_MAP.get(sensor_name, sensor_name)


def _build_video_filename(video: VideoData, picker_id: str, condition: str) -> str:
    """Build <pickerID>_<camera>_<condition>.mp4 filename."""
    safe_picker = _sanitize_filename_part(picker_id)
    safe_condition = _sanitize_filename_part(condition)
    safe_camera = _sanitize_filename_part(video.camera_type or ("eyes" if video.is_primary else "unknown"))
    return f"{safe_picker}_{safe_camera}_{safe_condition}.mp4"


def _build_sensor_filename(picker_id: str, condition: str, sensor_name: str) -> str:
    """Build <pickerID>_eyes_<eye_tracking_data>_<condition>.csv filename."""
    safe_picker = _sanitize_filename_part(picker_id)
    safe_condition = _sanitize_filename_part(condition)
    safe_sensor = _sanitize_filename_part(_sensor_export_name(sensor_name))
    return f"{safe_picker}_eyes_{safe_sensor}_{safe_condition}.csv"


def _build_metadata_filename(condition: str) -> str:
    """Build export metadata filename with condition suffix."""
    safe_condition = _sanitize_filename_part(condition, default='unknown_condition')
    return f"export_metadata_{safe_condition}.json"


def preview_export_file_paths(project: ProjectData,
                              output_folder: Path,
                              picker_id: str,
                              condition: str,
                              export_gaze_overlay: bool = True,
                              enabled_video_indices: Optional[List[int]] = None) -> Dict[str, Path]:
    """Compute deterministic output paths for conflict checks before export."""
    paths: Dict[str, Path] = {}
    output_folder = Path(output_folder)

    if not project.primary_video:
        return paths

    export_primary = enabled_video_indices is None or 0 in enabled_video_indices
    primary = project.primary_video

    if export_primary:
        paths['primary_video'] = output_folder / _build_video_filename(primary, picker_id, condition)
        if export_gaze_overlay and 'gaze' in project.sensor_data:
            overlay_name = _build_video_filename(primary, picker_id, condition).replace('.mp4', '_gaze_overlay.mp4')
            paths['gaze_overlay_video'] = output_folder / overlay_name

        # Sensors are only exported when primary is selected.
        for sensor_name in project.sensor_data.keys():
            paths[f'sensor_{sensor_name}'] = output_folder / _build_sensor_filename(
                picker_id,
                condition,
                sensor_name,
            )

    for i, video in enumerate(project.additional_videos):
        video_index = i + 1
        if enabled_video_indices is not None and video_index not in enabled_video_indices:
            continue
        paths[f'additional_video_{i+1}'] = output_folder / _build_video_filename(video, picker_id, condition)

    paths['metadata'] = output_folder / _build_metadata_filename(condition)
    return paths


def safe_output_path(base_path: Path, suffix: str = "_trimmed") -> Path:
    """Generate unique output filename by appending suffix before extension."""
    stem = base_path.stem
    suffix_str = suffix
    parent = base_path.parent
    ext = base_path.suffix
    
    candidate = parent / f"{stem}{suffix_str}{ext}"
    i = 1
    while candidate.exists():
        if "_trim" in suffix_str and not suffix_str.endswith(('_1', '_2', '_3')):
            candidate = parent / f"{stem}{suffix_str}{i}{ext}"
        else:
            candidate = parent / f"{stem}{suffix_str}_{i}{ext}"
        i += 1
    return candidate


def write_data_file(df: pd.DataFrame, output_path: Path, format_type: str) -> None:
    """Write DataFrame to CSV or Excel file."""
    if format_type == 'csv':
        df.to_csv(str(output_path), index=False)
    else:
        df.to_excel(str(output_path), index=False, engine='openpyxl')


def _detect_video_codec_needs_conversion(file_path: str) -> bool:
    """Check if video uses HEVC codec (not Windows-compatible)."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=codec_name',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               str(file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            codec = result.stdout.strip().lower()
            return 'hevc' in codec or 'h265' in codec
        return False
    except Exception:
        return False


def _probe_video_bitrate_kbps(file_path: str, fallback_kbps: int = 5000) -> int:
    """Probe source video bitrate. Tries stream-level first, then container-level,
    then falls back to computing from file size and duration."""
    try:
        # 1) Stream-level bitrate (may be N/A for many H.264 MP4 files)
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=bit_rate',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               str(file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            raw = result.stdout.strip().splitlines()
            val = raw[0].strip() if raw else ''
            if val.isdigit() and int(val) > 0:
                return max(500, int(int(val) / 1000))

        # 2) Container/format-level bitrate (reliable for most MP4 files)
        cmd2 = ['ffprobe', '-v', 'error',
                '-show_entries', 'format=bit_rate,duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(file_path)]
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=10)
        if result2.returncode == 0:
            vals = [v.strip() for v in result2.stdout.strip().splitlines() if v.strip()]
            # format=bit_rate comes first, then duration
            if vals and vals[0].isdigit() and int(vals[0]) > 0:
                total_kbps = max(500, int(int(vals[0]) / 1000))
                # Subtract typical audio overhead (~128 kbps) to get video-only estimate
                return max(300, total_kbps - 128)

        # 3) Derive from file size / duration as last resort
        probe_dur = ['ffprobe', '-v', 'error',
                     '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1',
                     str(file_path)]
        dur_result = subprocess.run(probe_dur, capture_output=True, text=True, timeout=10)
        if dur_result.returncode == 0:
            dur_str = dur_result.stdout.strip()
            try:
                dur_sec = float(dur_str)
                file_bytes = Path(file_path).stat().st_size
                if dur_sec > 0 and file_bytes > 0:
                    total_kbps = int((file_bytes * 8) / (dur_sec * 1000))
                    return max(300, total_kbps - 128)
            except (ValueError, OSError):
                pass
    except Exception:
        pass
    return fallback_kbps


def _parse_fps_fraction(raw_value: str) -> Optional[float]:
    """Parse ffprobe frame rate fractions like '30000/1001' with full precision."""
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


def _probe_video_fps_precise(file_path: str, fallback_fps: float = 30.0) -> float:
    """Probe precise FPS using ffprobe avg_frame_rate/r_frame_rate."""
    # Keep a single source of truth: if the loader already provided a valid FPS,
    # use it directly so playback panel, preview, and exporter stay consistent.
    if fallback_fps and fallback_fps > 0:
        return float(fallback_fps)

    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=avg_frame_rate,r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = [line.strip() for line in (result.stdout or '').splitlines() if line.strip()]
            for candidate in lines:
                parsed = _parse_fps_fraction(candidate)
                if parsed is not None:
                    return parsed
    except Exception:
        pass
    return fallback_fps if fallback_fps and fallback_fps > 0 else 30.0


def _compute_primary_trim_duration_sec(video: VideoData, start_frame: int, end_frame: int,
                                       start_ns: int, end_ns: int) -> float:
    """Compute primary trim duration with max precision.

    Prefers timestamps (including the last frame interval) when available, then falls back
    to frame_count / precise fps.
    """
    frame_count = max(0, end_frame - start_frame + 1)
    if frame_count <= 0:
        return 0.0

    # Highest-precision path: world timestamps + one frame interval for end-exclusive time.
    if video.world_timestamps is not None and len(video.world_timestamps) > end_frame:
        try:
            if end_frame + 1 < len(video.world_timestamps):
                end_exclusive_ns = int(video.world_timestamps[end_frame + 1])
            else:
                frame_ns = int((1e9 / video.fps)) if video.fps and video.fps > 0 else int(1e9 / 30.0)
                end_exclusive_ns = int(end_ns) + frame_ns
            duration = (end_exclusive_ns - int(start_ns)) / 1e9
            if duration > 0:
                return duration
        except Exception:
            pass

    fps_precise = _probe_video_fps_precise(str(video.file_path), video.fps)
    return frame_count / fps_precise if fps_precise > 0 else frame_count / 30.0


def _is_valid_video_output(file_path: Path) -> bool:
    """Check if an output video is readable (guards against broken MP4 with missing moov atom)."""
    try:
        if not file_path.exists() or file_path.stat().st_size < 1000:
            return False

        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return False

        codec = (result.stdout or '').strip().splitlines()
        return len(codec) > 0 and codec[0] != ''
    except Exception:
        return False


def export_project(project: ProjectData, output_folder: Path, 
                   export_gaze_overlay: bool = True,
                   progress_callback=None,
                   picker_id: Optional[str] = None,
                   condition: Optional[str] = None,
                   enabled_video_indices: Optional[list] = None,
                   existing_file_policy: str = "overwrite") -> Dict[str, str]:
    """Export trimmed videos and sensor data.
    
    Args:
        project: ProjectData with all loaded media
        output_folder: Output directory
        export_gaze_overlay: Whether to create gaze overlay video
        progress_callback: Callback for progress messages
        picker_id: Picker ID for naming (e.g., "P001")
        condition: Condition for naming (e.g., "left_direction")
        enabled_video_indices: List of enabled video indices (None = export all).
                              Index 0 is primary video, 1+ are additional videos.
        existing_file_policy: How to handle existing files: "overwrite" or "skip".
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    results = {}

    def progress(msg: str):
        if progress_callback:
            progress_callback(msg)
        print(msg)

    def _generate_custom_filename(video: VideoData, is_sensor: bool = False, 
                                   sensor_type: Optional[str] = None) -> str:
        """Generate custom filename based on picker_id, camera, condition.
        
        For videos: <pickerID>_<camera>_<condition>.mp4
        For sensors: <pickerID>_<camera>_<sensor_type>_<condition>.csv
        """
        if not picker_id or not condition:
            # Fallback to original names if parameters not provided
            return video.name
        
        if is_sensor and sensor_type:
            return _build_sensor_filename(picker_id, condition, sensor_type)
        return _build_video_filename(video, picker_id, condition)

    def _get_frame_count(path: Path):
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_frames',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                values = [line.strip() for line in (result.stdout or '').splitlines() if line.strip()]
                for value in values:
                    if value.isdigit() and int(value) > 0:
                        return int(value)

            count_cmd = [
                'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(path),
            ]
            count_result = subprocess.run(count_cmd, capture_output=True, text=True, timeout=60)
            if count_result.returncode == 0:
                values = [line.strip() for line in (count_result.stdout or '').splitlines() if line.strip()]
                for value in values:
                    if value.isdigit() and int(value) > 0:
                        return int(value)

            import cv2
            cap = cv2.VideoCapture(str(path))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            return count if count > 0 else None
        except Exception:
            return None

    def _get_duration_sec(path: Path):
        """Return video duration, preferring stream duration over container duration.

        MP4 container duration can be slightly longer than the actual video stream because of
        muxing metadata/track offsets. That breaks FPS-sync validation for otherwise correct
        outputs, so the stream duration is the primary source of truth.
        """
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=duration:format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                return None

            values = [line.strip() for line in (result.stdout or '').splitlines() if line.strip()]
            for value in values:
                try:
                    duration = float(value)
                    if duration > 0:
                        return duration
                except Exception:
                    continue
            return None
        except Exception:
            return None

    def _get_output_fps(path: Path):
        """Read output FPS from ffprobe stream rate."""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=avg_frame_rate,r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                return None
            lines = [line.strip() for line in (result.stdout or '').splitlines() if line.strip()]
            for candidate in lines:
                parsed = _parse_fps_fraction(candidate)
                if parsed is not None and parsed > 0:
                    return parsed
            return None
        except Exception:
            return None

    def _is_timing_match(path: Path, expected_frames: int, expected_fps: float,
                         expected_duration_sec: Optional[float] = None,
                         frame_tolerance: int = 2,
                         diagnostics: Optional[Dict[str, object]] = None) -> bool:
        diag = diagnostics if isinstance(diagnostics, dict) else None
        if diag is not None:
            diag.clear()
            diag.update({
                'path': str(path),
                'expected_frames': int(expected_frames),
                'expected_fps': float(expected_fps) if expected_fps is not None else None,
                'expected_duration_sec': float(expected_duration_sec) if expected_duration_sec is not None else None,
                'frame_tolerance': int(frame_tolerance),
            })

        if not _is_valid_video_output(path):
            if diag is not None:
                diag.update({'result': False, 'failure': 'invalid_output'})
            return False

        out_frames = _get_frame_count(path)
        frame_match = (out_frames is None) or (abs(out_frames - expected_frames) <= frame_tolerance)
        if diag is not None:
            diag['out_frames'] = out_frames
            diag['frame_match'] = bool(frame_match)

        # When caller does not provide a duration target, frame-count match is enough.
        if expected_duration_sec is None and out_frames is not None and frame_match:
            if diag is not None:
                diag.update({'result': True, 'mode': 'frames_only'})
            return True

        target_duration = expected_duration_sec
        if target_duration is None and expected_fps and expected_fps > 0:
            target_duration = expected_frames / expected_fps

        if target_duration is None or target_duration <= 0:
            if diag is not None:
                diag.update({'result': out_frames is None, 'failure': 'invalid_expected_duration'})
            return out_frames is None

        out_fps = _get_output_fps(path) if expected_fps and expected_fps > 0 else None
        out_duration = None

        # Prefer frame-count/FPS-derived duration when available because it matches the actual
        # retimed stream precisely and avoids container-duration drift.
        if out_frames is not None and out_frames > 0 and out_fps is not None and out_fps > 0:
            out_duration = out_frames / out_fps

        if out_duration is None:
            out_duration = _get_duration_sec(path)

        if diag is not None:
            diag['out_fps'] = out_fps
            diag['out_duration_sec'] = out_duration
            diag['target_duration_sec'] = target_duration

        if out_duration is None:
            if diag is not None:
                diag.update({'result': False, 'failure': 'missing_output_duration'})
            return False

        duration_tolerance = max(0.04, (2.0 / expected_fps) if expected_fps and expected_fps > 0 else 0.08)
        duration_match = abs(out_duration - target_duration) <= duration_tolerance
        if diag is not None:
            diag['duration_tolerance_sec'] = duration_tolerance
            diag['duration_match'] = bool(duration_match)

        # In FPS-sync mode, also verify stream FPS when expected_duration_sec is provided.
        if expected_duration_sec is not None and expected_fps and expected_fps > 0:
            if out_fps is None:
                result = frame_match and duration_match
                if diag is not None:
                    diag.update({'result': bool(result), 'failure': None if result else 'missing_output_fps'})
                return result
            fps_tolerance = max(0.1, expected_fps * 0.01)
            fps_match = abs(out_fps - expected_fps) <= fps_tolerance
            result = frame_match and duration_match and fps_match
            if diag is not None:
                diag['fps_tolerance'] = fps_tolerance
                diag['fps_match'] = bool(fps_match)
                if not frame_match:
                    failure = 'frame_mismatch'
                elif not duration_match:
                    failure = 'duration_mismatch'
                elif not fps_match:
                    failure = 'fps_mismatch'
                else:
                    failure = None
                diag.update({'result': bool(result), 'failure': failure})
            return result

        result = frame_match and duration_match
        if diag is not None:
            diag.update({'result': bool(result), 'failure': None if result else ('frame_mismatch' if not frame_match else 'duration_mismatch')})
        return result

    def _summarize_validation(diag: Optional[Dict[str, object]]) -> str:
        if not diag:
            return "no validation details"
        failure = diag.get('failure') or 'unknown'
        return (
            f"validation={failure}; expected(frames={diag.get('expected_frames')}, "
            f"fps={diag.get('expected_fps')}, dur={diag.get('expected_duration_sec')}) "
            f"actual(frames={diag.get('out_frames')}, fps={diag.get('out_fps')}, dur={diag.get('out_duration_sec')})"
        )

    def _summarize_encoder_diagnostics(diag: Optional[Dict[str, object]]) -> str:
        if not diag:
            return "no encoder diagnostics"
        status = diag.get('status') or 'unknown'
        category = diag.get('failure_category') or 'unknown'
        attempts = diag.get('attempts') or []
        last_attempt = attempts[-1] if attempts else {}
        encoder = last_attempt.get('encoder') or diag.get('selected_encoder') or 'unknown'
        stderr = (last_attempt.get('stderr') or diag.get('last_error') or '').strip()
        stderr = stderr.replace('\n', ' ')[:220]
        return f"status={status}; category={category}; encoder={encoder}; stderr={stderr or 'n/a'}"

    def _clamp_trim_range(video, start_frame_val, end_frame_val):
        total_frames = max(int(video.frame_count or 0), 1)
        start_clamped = max(0, int(start_frame_val or 0))
        end_raw = total_frames - 1 if end_frame_val is None else int(end_frame_val)
        end_clamped = min(max(0, end_raw), total_frames - 1)
        if end_clamped < start_clamped:
            start_clamped = 0
            end_clamped = total_frames - 1
        return start_clamped, end_clamped

    def _prepare_output_for_write(path: Path) -> bool:
        """Prepare destination path for write according to overwrite policy."""
        if existing_file_policy != "overwrite":
            return True
        if not path.exists():
            return True
        try:
            path.unlink()
            return True
        except Exception as e:
            progress(f"  [FAIL] Cannot overwrite existing file {path.name}: {e}")
            return False

    if not project.primary_video:
        progress("Error: No primary video loaded")
        return results

    existing_file_policy = (existing_file_policy or "overwrite").strip().lower()
    if existing_file_policy not in {"overwrite", "skip"}:
        existing_file_policy = "overwrite"

    primary_video = project.primary_video
    raw_start_frame, raw_end_frame = primary_video.get_trimmed_range()
    start_frame, end_frame = _clamp_trim_range(primary_video, raw_start_frame, raw_end_frame)
    if (start_frame, end_frame) != (raw_start_frame, raw_end_frame):
        progress(
            f"  Primary trim range adjusted to available frames: "
            f"{raw_start_frame}-{raw_end_frame} -> {start_frame}-{end_frame}"
        )

    start_ns = primary_video.get_frame_timestamp(start_frame)
    end_ns = primary_video.get_frame_timestamp(end_frame)

    def get_next_export_count(folder: Path, base_name: str) -> int:
        count = 1
        while True:
            test_path = folder / f"{base_name}_trim{count}.mp4"
            if not test_path.exists():
                break
            count += 1
        return count

    export_count = get_next_export_count(output_folder, primary_video.name)

    # Check if primary video should be exported (enabled_video_indices check)
    export_primary = enabled_video_indices is None or 0 in enabled_video_indices

    # ===== PRIMARY VIDEO EXPORT =====
    progress("=" * 60)
    
    if not export_primary:
        progress("Skipping primary video (disabled in playback panel)")
    else:
        progress("Exporting primary video...")
        
        # Generate custom filename or use default
        if picker_id and condition:
            custom_filename = _generate_custom_filename(primary_video, is_sensor=False)
            primary_out = output_folder / custom_filename
        else:
            primary_out = output_folder / f"{primary_video.name}.mp4"
            primary_out = safe_output_path(primary_out, f"_trim{export_count}")

        if existing_file_policy == "skip" and primary_out.exists():
            progress(f"  [SKIP] Primary video already exists: {primary_out.name}")
            ffmpeg_ok = False
        else:
            if not _prepare_output_for_write(primary_out):
                ffmpeg_ok = False
                progress("  [FAIL] Primary export skipped due to file overwrite error")
            else:
                primary_fps_precise = _probe_video_fps_precise(str(primary_video.file_path), primary_video.fps)
                start_sec = start_frame / primary_fps_precise
                expected_primary_frames = end_frame - start_frame + 1
                expected_primary_duration_sec = expected_primary_frames / primary_fps_precise
                ffmpeg_ok = False

                # Primary export is always re-encoded for frame-accurate cuts.
                # Stream-copy can start on previous keyframes and produce visible decode artifacts
                # or timeline jumps at the beginning of trimmed clips.
                first_encode_diag = {}
                first_validate_diag = {}
                retry_encode_diag = {}
                retry_validate_diag = {}

                if primary_video.fps and primary_video.fps > 0 and expected_primary_frames > 0:
                    if export_trimmed_video_ffmpeg_reencode(
                        str(primary_video.file_path), str(primary_out), start_sec,
                        expected_primary_frames, primary_video.fps, pad_frames=0,
                        constant_bitrate=False,
                        diagnostics=first_encode_diag,
                    ):
                        out_frames = _get_frame_count(primary_out)
                        if _is_timing_match(
                            primary_out,
                            expected_primary_frames,
                            primary_fps_precise,
                            expected_duration_sec=expected_primary_duration_sec,
                            frame_tolerance=2,
                            diagnostics=first_validate_diag,
                        ):
                            ffmpeg_ok = True
                            results['primary_video'] = str(primary_out)
                            progress(f"  [OK] FFmpeg re-encode successful ({out_frames}/{expected_primary_frames} frames)")
                        elif _is_valid_video_output(primary_out) and out_frames is None:
                            ffmpeg_ok = True
                            results['primary_video'] = str(primary_out)
                            progress(f"  [OK] FFmpeg re-encode successful (expected {expected_primary_frames} frames)")
                        else:
                            progress(f"  [DIAG] Primary validation mismatch: {_summarize_validation(first_validate_diag)}")
                    else:
                        progress(f"  [DIAG] Primary encode failed (attempt 1): {_summarize_encoder_diagnostics(first_encode_diag)}")

                if not ffmpeg_ok:
                    progress("  Primary re-encode did not validate on first try; retrying with safer settings")
                    if export_trimmed_video_ffmpeg_reencode(
                        str(primary_video.file_path), str(primary_out), start_sec,
                        expected_primary_frames, primary_video.fps, pad_frames=0,
                        constant_bitrate=False,
                        diagnostics=retry_encode_diag,
                    ):
                        out_frames = _get_frame_count(primary_out)
                        if _is_timing_match(
                            primary_out,
                            expected_primary_frames,
                            primary_fps_precise,
                            expected_duration_sec=expected_primary_duration_sec,
                            frame_tolerance=2,
                            diagnostics=retry_validate_diag,
                        ):
                            ffmpeg_ok = True
                            results['primary_video'] = str(primary_out)
                            progress(f"  [OK] FFmpeg re-encode retry successful")
                        else:
                            progress(f"  [DIAG] Primary validation mismatch (retry): {_summarize_validation(retry_validate_diag)}")
                    else:
                        progress(f"  [DIAG] Primary encode failed (retry): {_summarize_encoder_diagnostics(retry_encode_diag)}")

                if not ffmpeg_ok:
                    progress("  [FAIL] Failed to export primary video; continuing with sensor/additional exports")

        if ffmpeg_ok and export_gaze_overlay and 'gaze' in project.sensor_data:
            progress("Creating gaze overlay video...")
            if picker_id and condition:
                custom_overlay_filename = _generate_custom_filename(primary_video, is_sensor=False)
                custom_overlay_filename = custom_overlay_filename.replace(".mp4", "_gaze_overlay.mp4")
                gaze_overlay_out = output_folder / custom_overlay_filename
            else:
                gaze_overlay_out = output_folder / f"{primary_video.name}.mp4"
                gaze_overlay_out = safe_output_path(gaze_overlay_out, f"_trim{export_count}_gaze_overlay")
            if existing_file_policy == "skip" and gaze_overlay_out.exists():
                progress(f"  [SKIP] Gaze overlay already exists: {gaze_overlay_out.name}")
            else:
                if _prepare_output_for_write(gaze_overlay_out):
                    if create_gaze_overlay_video(str(primary_out), str(gaze_overlay_out), primary_video,
                                                 project.sensor_data['gaze'], start_frame, end_frame):
                        results['gaze_overlay_video'] = str(gaze_overlay_out)

    # ===== SENSOR DATA EXPORT (CSV Files) =====
    if not export_primary:
        progress("Skipping sensor data export because primary video is disabled")
    else:
        progress("Exporting sensor data...")
        for sensor_name, sensor_data in project.sensor_data.items():
            if picker_id and condition:
                sensor_out = output_folder / _generate_custom_filename(
                    primary_video, is_sensor=True, sensor_type=sensor_name
                )
            else:
                original_name = sensor_data.file_path.stem
                ext = sensor_data.file_path.suffix
                sensor_out = output_folder / f"{original_name}{ext}"
                sensor_out = safe_output_path(sensor_out, f"_trim{export_count}")

            if existing_file_policy == "skip" and sensor_out.exists():
                progress(f"  [SKIP] Sensor file already exists: {sensor_out.name}")
                continue

            if not _prepare_output_for_write(sensor_out):
                continue

            progress(f"  Trimming {sensor_name}...")
            from data_synchronizer import trim_sensor_data
            trimmed_df = trim_sensor_data(sensor_data, start_ns, end_ns)

            try:
                write_data_file(trimmed_df, sensor_out, sensor_data.format_type)
                results[sensor_name] = str(sensor_out)
                progress(f"    [OK] Saved {len(trimmed_df)} rows")
            except Exception as e:
                progress(f"  Error saving {sensor_name}: {e}")

    # ===== ADDITIONAL VIDEOS EXPORT =====
    progress("=" * 60)
    progress("Exporting additional videos...")

    primary_frame_count = end_frame - start_frame + 1
    primary_duration_sec = _compute_primary_trim_duration_sec(
        primary_video,
        start_frame,
        end_frame,
        start_ns,
        end_ns,
    )

    for i, video in enumerate(project.additional_videos):
        # Check if this video should be exported (enabled_video_indices: i+1 because 0 is primary)
        video_index = i + 1
        if enabled_video_indices is not None and video_index not in enabled_video_indices:
            progress(f"  Video {i+2}: {video.name} (skipped - disabled)")
            continue
        
        progress(f"  Video {i+2}: {video.name}")

        raw_video_start = video.start_frame if video.start_frame is not None else 0
        raw_video_end = video.end_frame if video.end_frame is not None else video.frame_count - 1
        video_start_frame, video_end_frame = _clamp_trim_range(video, raw_video_start, raw_video_end)
        if (video_start_frame, video_end_frame) != (raw_video_start, raw_video_end):
            progress(
                f"    Trim range adjusted to available frames: "
                f"{raw_video_start}-{raw_video_end} -> {video_start_frame}-{video_end_frame}"
            )
        source_len = max(0, video_end_frame - video_start_frame + 1)
        source_fps_precise = _probe_video_fps_precise(str(video.file_path), video.fps)
        additional_reference_bitrate = _probe_video_bitrate_kbps(str(video.file_path))

        # Single synchronization mode: keep exact selected frames and adjust FPS so
        # duration matches primary video duration (no padding/truncation).
        target_frames = source_len
        if source_fps_precise > 0 and primary_duration_sec > 0:
            adjusted_out_fps = source_len / primary_duration_sec
        else:
            adjusted_out_fps = source_fps_precise or 30.0
        progress(
            f"    FPS-Sync mode: {source_len} source frames, "
            f"output fps {adjusted_out_fps:.9f} (source {source_fps_precise:.9f}) "
            f"-> duration {primary_duration_sec:.6f}s"
        )

        progress(f"    Frame range: {video_start_frame}-{video_end_frame} ({source_len} frames)")

        # Generate custom filename or use default
        if picker_id and condition:
            custom_filename = _generate_custom_filename(video, is_sensor=False)
            video_out = output_folder / custom_filename
        else:
            video_out = output_folder / f"{video.name}.mp4"
            video_out = safe_output_path(video_out, f"_trim{export_count}")

        if existing_file_policy == "skip" and video_out.exists():
            progress(f"    [SKIP] Additional video already exists: {video_out.name}")
            continue

        if not _prepare_output_for_write(video_out):
            continue

        ffmpeg_add_ok = False
        start_sec_add = video_start_frame / source_fps_precise if source_fps_precise > 0 else 0.0
        first_encode_diag = {}
        first_validate_diag = {}
        retry_encode_diag = {}
        retry_validate_diag = {}

        # Exact-frame sync export: preserve selected frame count and use adjusted FPS.
        # Prefer FFmpeg re-encode first to keep codec/bitrate behavior close to source.
        progress("    Exact-frame FPS sync export (codec/bitrate-preserving)")
        if source_fps_precise > 0 and target_frames > 0:
            if export_trimmed_video_ffmpeg_reencode(
                str(video.file_path),
                str(video_out),
                start_sec_add,
                target_frames,
                source_fps_precise,
                pad_frames=0,
                target_bitrate_kbps=additional_reference_bitrate,
                constant_bitrate=False,
                output_fps=adjusted_out_fps,
                diagnostics=first_encode_diag,
            ):
                if _is_timing_match(
                    video_out,
                    target_frames,
                    adjusted_out_fps,
                    expected_duration_sec=primary_duration_sec,
                    frame_tolerance=2,
                    diagnostics=first_validate_diag,
                ):
                    ffmpeg_add_ok = True
                    progress("    [OK] FFmpeg sync export successful")
                else:
                    progress(f"    [DIAG] Sync validation mismatch: {_summarize_validation(first_validate_diag)}")
            else:
                progress(f"    [DIAG] Sync encode failed (attempt 1): {_summarize_encoder_diagnostics(first_encode_diag)}")

        if not ffmpeg_add_ok:
            progress("    FFmpeg sync export did not validate; retrying with safer bitrate")
            safer_bitrate = min(int(additional_reference_bitrate or 8000), 12000)
            if source_fps_precise > 0 and target_frames > 0:
                if export_trimmed_video_ffmpeg_reencode(
                    str(video.file_path),
                    str(video_out),
                    start_sec_add,
                    target_frames,
                    source_fps_precise,
                    pad_frames=0,
                    target_bitrate_kbps=safer_bitrate,
                    constant_bitrate=False,
                    output_fps=adjusted_out_fps,
                    diagnostics=retry_encode_diag,
                ):
                    if _is_timing_match(
                        video_out,
                        target_frames,
                        adjusted_out_fps,
                        expected_duration_sec=primary_duration_sec,
                        frame_tolerance=2,
                        diagnostics=retry_validate_diag,
                    ):
                        ffmpeg_add_ok = True
                        progress("    [OK] FFmpeg safer retry successful")
                    else:
                        progress(f"    [DIAG] Sync validation mismatch (retry): {_summarize_validation(retry_validate_diag)}")
                else:
                    progress(f"    [DIAG] Sync encode failed (retry): {_summarize_encoder_diagnostics(retry_encode_diag)}")

        if not ffmpeg_add_ok:
            progress("    [FAIL] FFmpeg sync export failed for this video (OpenCV fallback disabled - it produces large, choppy output)")
            try:
                video_out.unlink(missing_ok=True)
            except Exception:
                pass

        if ffmpeg_add_ok:
            results[f'additional_video_{i+1}'] = str(video_out)
        else:
            progress(f"    [FAIL] Additional video failed for {video.name}")

    progress("=" * 60)
    progress("Export complete!")
    return results


def export_project_metadata(project: ProjectData,
                            output_folder: Path,
                            condition: str,
                            existing_file_policy: str = "overwrite",
                            enabled_video_indices: Optional[List[int]] = None,
                            export_results: Optional[Dict[str, str]] = None) -> Path:
    """Export metadata file describing the project."""
    export_results = export_results or {}

    def _video_label(video: VideoData) -> str:
        camera = (video.camera_type or ("eyes" if video.is_primary else "unknown")).strip()
        return f"{camera} ({video.name})"

    def _probe_exported_video_stats(path: Path):
        """Return measured (fps, duration_sec) from an exported file when possible."""
        fps_val = None
        dur_val = None
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=avg_frame_rate,r_frame_rate:format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                lines = [line.strip() for line in (result.stdout or '').splitlines() if line.strip()]
                # stream rates first, duration last
                if lines:
                    for candidate in lines[:2]:
                        parsed = _parse_fps_fraction(candidate)
                        if parsed is not None and parsed > 0:
                            fps_val = parsed
                            break
                    try:
                        dur_val = float(lines[-1])
                    except Exception:
                        dur_val = None
        except Exception:
            pass
        return fps_val, dur_val

    metadata = {
        'primary_video_file': str(project.primary_video.file_path) if project.primary_video else None,
        'additional_video_files': [str(v.file_path) for v in project.additional_videos],
        'sensor_data_files': {
            sensor_name: str(sensor_data.file_path)
            for sensor_name, sensor_data in project.sensor_data.items()
        },
        'video_trimming': {},
    }

    # Include trim information for all exported videos (primary + enabled additionals).
    if project.primary_video and 'primary_video' in export_results:
        video = project.primary_video
        start_frame, end_frame = video.get_trimmed_range()
        start_ns = video.get_frame_timestamp(start_frame)
        end_ns = video.get_frame_timestamp(end_frame)
        measured_fps, measured_duration = _probe_exported_video_stats(Path(export_results['primary_video']))
        metadata['video_trimming'][_video_label(video)] = {
            'trim_start_frame': int(start_frame),
            'trim_end_frame': int(end_frame),
            'trim_start_timestamp_ns': int(start_ns),
            'trim_end_timestamp_ns': int(end_ns),
            'fps': float(measured_fps if measured_fps is not None else video.fps),
            'duration_seconds': float(
                measured_duration if measured_duration is not None else ((end_frame - start_frame + 1) / max(video.fps, 1e-9))
            ),
            'source_video_file': str(video.file_path),
            'exported_video_file': str(export_results['primary_video']),
        }

    for i, video in enumerate(project.additional_videos):
        result_key = f'additional_video_{i+1}'
        if result_key not in export_results:
            continue
        video_index = i + 1
        if enabled_video_indices is not None and video_index not in enabled_video_indices:
            continue

        start_frame, end_frame = video.get_trimmed_range()
        start_ns = video.get_frame_timestamp(start_frame)
        end_ns = video.get_frame_timestamp(end_frame)
        measured_fps, measured_duration = _probe_exported_video_stats(Path(export_results[result_key]))
        metadata['video_trimming'][_video_label(video)] = {
            'trim_start_frame': int(start_frame),
            'trim_end_frame': int(end_frame),
            'trim_start_timestamp_ns': int(start_ns),
            'trim_end_timestamp_ns': int(end_ns),
            'fps': float(measured_fps if measured_fps is not None else video.fps),
            'duration_seconds': float(
                measured_duration if measured_duration is not None else ((end_frame - start_frame + 1) / max(video.fps, 1e-9))
            ),
            'source_video_file': str(video.file_path),
            'exported_video_file': str(export_results[result_key]),
        }
    
    import json

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    existing_file_policy = (existing_file_policy or "overwrite").strip().lower()
    if existing_file_policy not in {"overwrite", "skip"}:
        existing_file_policy = "overwrite"

    metadata_path = output_folder / _build_metadata_filename(condition)

    if existing_file_policy == "skip" and metadata_path.exists():
        return metadata_path

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata_path
