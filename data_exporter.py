"""
Data export utilities for saving trimmed videos and sensor data.
"""
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import os
import shutil
import subprocess
import re
from models import ProjectData, VideoData, SensorData
from data_synchronizer import (
    export_trimmed_video_ffmpeg,
    export_trimmed_video_ffmpeg_reencode,
    export_trimmed_video_opencv,
    create_gaze_overlay_video,
)


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
    """Probe source video bitrate for sizing decisions."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=bit_rate',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               str(file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            raw = result.stdout.strip().splitlines()
            if raw and raw[0].isdigit():
                return max(500, int(int(raw[0]) / 1000))
    except Exception:
        pass
    return fallback_kbps


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
                   progress_callback=None) -> Dict[str, str]:
    """Export trimmed videos and sensor data."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    results = {}

    def progress(msg: str):
        if progress_callback:
            progress_callback(msg)
        print(msg)

    def _get_frame_count(path: Path):
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            return count
        except Exception:
            return None

    def _clamp_trim_range(video, start_frame_val, end_frame_val):
        total_frames = max(int(video.frame_count or 0), 1)
        start_clamped = max(0, int(start_frame_val or 0))
        end_raw = total_frames - 1 if end_frame_val is None else int(end_frame_val)
        end_clamped = min(max(0, end_raw), total_frames - 1)
        if end_clamped < start_clamped:
            start_clamped = 0
            end_clamped = total_frames - 1
        return start_clamped, end_clamped

    if not project.primary_video:
        progress("Error: No primary video loaded")
        return results

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

    # ===== PRIMARY VIDEO EXPORT =====
    progress("=" * 60)
    progress("Exporting primary video...")
    primary_out = output_folder / f"{primary_video.name}.mp4"
    primary_out = safe_output_path(primary_out, f"_trim{export_count}")

    start_sec = start_frame / primary_video.fps
    expected_primary_frames = end_frame - start_frame + 1
    ffmpeg_ok = False
    
    # FIX #1: Detect and force H.264 for HEVC videos
    primary_video_is_hevc = _detect_video_codec_needs_conversion(str(primary_video.file_path))
    
    if primary_video_is_hevc:
        progress(f"  Detected HEVC codec (not Windows-compatible), forcing re-encode to H.264")
        if primary_video.fps and primary_video.fps > 0 and expected_primary_frames > 0:
            if export_trimmed_video_ffmpeg_reencode(
                str(primary_video.file_path), str(primary_out), start_sec,
                expected_primary_frames, primary_video.fps, pad_frames=0
            ):
                out_frames = _get_frame_count(primary_out)
                if _is_valid_video_output(primary_out) and out_frames is not None and abs(out_frames - expected_primary_frames) <= 1:
                    ffmpeg_ok = True
                    results['primary_video'] = str(primary_out)
                    progress(f"  [OK] FFmpeg re-encode successful ({out_frames}/{expected_primary_frames} frames)")
                elif _is_valid_video_output(primary_out) and out_frames is None:
                    ffmpeg_ok = True
                    results['primary_video'] = str(primary_out)
                    progress(f"  [OK] FFmpeg re-encode successful (expected {expected_primary_frames} frames)")
    else:
        # Try stream copy for compatible codecs
        if primary_video.fps and primary_video.fps > 0 and expected_primary_frames > 0:
            duration_candidates = []
            for frame_delta in [-1, 0, 1, -2, 2]:
                candidate_frames = max(expected_primary_frames + frame_delta, 1)
                duration_candidates.append(candidate_frames / primary_video.fps)
        else:
            duration_candidates = [max((end_frame - start_frame) / primary_video.fps, 0.0)]

        for attempt, duration_sec in enumerate(duration_candidates, start=1):
            progress(f"  FFmpeg attempt {attempt}: duration={duration_sec:.6f}s")
            if not export_trimmed_video_ffmpeg(str(primary_video.file_path), str(primary_out), start_sec, duration_sec):
                continue
            out_frames = _get_frame_count(primary_out)
            if _is_valid_video_output(primary_out) and (out_frames is None or abs(out_frames - expected_primary_frames) <= 1):
                ffmpeg_ok = True
                results['primary_video'] = str(primary_out)
                progress(f"  [OK] Primary video exported successfully")
                break
            progress(f"  Frame mismatch on attempt {attempt}: got {out_frames}, expected {expected_primary_frames}")

        if not ffmpeg_ok:
            progress("  Stream-copy attempts failed; using FFmpeg re-encode fallback")
            if export_trimmed_video_ffmpeg_reencode(
                str(primary_video.file_path), str(primary_out), start_sec,
                expected_primary_frames, primary_video.fps, pad_frames=0
            ):
                out_frames = _get_frame_count(primary_out)
                if _is_valid_video_output(primary_out) and out_frames is not None and abs(out_frames - expected_primary_frames) <= 1:
                    ffmpeg_ok = True
                    results['primary_video'] = str(primary_out)
                    progress(f"  [OK] FFmpeg re-encode successful")

    if not ffmpeg_ok:
        progress("  [FAIL] Failed to export primary video; continuing with sensor/additional exports")

    if ffmpeg_ok and export_gaze_overlay and 'gaze' in project.sensor_data:
        progress("Creating gaze overlay video...")
        gaze_overlay_out = output_folder / f"{primary_video.name}.mp4"
        gaze_overlay_out = safe_output_path(gaze_overlay_out, f"_trim{export_count}_gaze_overlay")
        if create_gaze_overlay_video(str(primary_out), str(gaze_overlay_out), primary_video,
                                     project.sensor_data['gaze'], start_frame, end_frame):
            results['gaze_overlay_video'] = str(gaze_overlay_out)

    # ===== SENSOR DATA EXPORT (CSV Files) =====
    progress("Exporting sensor data...")
    for sensor_name, sensor_data in project.sensor_data.items():
        progress(f"  Trimming {sensor_name}...")
        from data_synchronizer import trim_sensor_data
        trimmed_df = trim_sensor_data(sensor_data, start_ns, end_ns)

        original_name = sensor_data.file_path.stem
        ext = sensor_data.file_path.suffix
        sensor_out = output_folder / f"{original_name}{ext}"
        sensor_out = safe_output_path(sensor_out, f"_trim{export_count}")

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
    primary_duration_sec = primary_frame_count / primary_video.fps if primary_video.fps else 0.0
    additional_reference_bitrate = None
    if project.additional_videos:
        additional_reference_bitrate = _probe_video_bitrate_kbps(str(project.additional_videos[0].file_path))

    for i, video in enumerate(project.additional_videos):
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

        if video.fps and video.fps > 0:
            target_frames = int(round(primary_duration_sec * video.fps))
        else:
            target_frames = primary_frame_count

        if source_len > target_frames:
            video_end_frame = video_start_frame + target_frames - 1
            source_len = target_frames

        if video_start_frame > video_end_frame:
            video_start_frame = 0
            video_end_frame = min(video.frame_count - 1, max(target_frames - 1, 0))
            source_len = max(0, video_end_frame - video_start_frame + 1)

        progress(f"    Frame range: {video_start_frame}-{video_end_frame} ({source_len} frames)")

        video_out = output_folder / f"{video.name}.mp4"
        video_out = safe_output_path(video_out, f"_trim{export_count}")

        start_sec_add = video_start_frame / video.fps if video.fps and video.fps > 0 else 0.0
        ffmpeg_add_ok = False

        # FIX #1: Detect and force H.264 for HEVC videos
        video_is_hevc = _detect_video_codec_needs_conversion(str(video.file_path))
        
        if video_is_hevc:
            progress(f"    Detected HEVC codec (not Windows-compatible), forcing re-encode to H.264")
            if video.fps and video.fps > 0 and target_frames > 0:
                pad_frames = max(0, target_frames - source_len)
                if export_trimmed_video_ffmpeg_reencode(
                    str(video.file_path), str(video_out), start_sec_add, target_frames, video.fps,
                    pad_frames=pad_frames,
                    target_bitrate_kbps=additional_reference_bitrate,
                    constant_bitrate=True,
                ):
                    out_frames = _get_frame_count(video_out)
                    if _is_valid_video_output(video_out) and out_frames is not None and abs(out_frames - target_frames) <= 1:
                        ffmpeg_add_ok = True
                        progress(f"    [OK] FFmpeg re-encode successful")
                    elif _is_valid_video_output(video_out) and out_frames is None:
                        ffmpeg_add_ok = True
                        progress(f"    [OK] FFmpeg re-encode successful")

                if not ffmpeg_add_ok:
                    progress("    HEVC CBR re-encode did not produce a valid output; retrying with safer VBR settings")
                    safer_bitrate = min(int(additional_reference_bitrate or 8000), 12000)
                    if export_trimmed_video_ffmpeg_reencode(
                        str(video.file_path), str(video_out), start_sec_add, target_frames, video.fps,
                        pad_frames=pad_frames,
                        target_bitrate_kbps=safer_bitrate,
                        constant_bitrate=False,
                    ):
                        out_frames = _get_frame_count(video_out)
                        if _is_valid_video_output(video_out) and (out_frames is None or abs(out_frames - target_frames) <= 1):
                            ffmpeg_add_ok = True
                            progress(f"    [OK] FFmpeg safer retry successful")

                if not ffmpeg_add_ok:
                    progress("    FFmpeg retries failed; falling back to OpenCV export")
                    success, frames_written, expected_frames = export_trimmed_video_opencv(
                        str(video.file_path),
                        str(video_out),
                        video_start_frame,
                        video_end_frame,
                        video.fps,
                        target_frame_count=target_frames,
                    )
                    if success and _is_valid_video_output(video_out):
                        ffmpeg_add_ok = True
                        progress(f"    [OK] OpenCV fallback successful ({frames_written}/{expected_frames} frames)")
        else:
            # Try stream copy for compatible codecs
            duration_candidates = []
            if video.fps and video.fps > 0 and target_frames > 0:
                for frame_delta in [-1, 0, 1, -2, 2]:
                    candidate_frames = max(target_frames + frame_delta, 1)
                    duration_candidates.append(candidate_frames / video.fps)
            else:
                duration_candidates.append(primary_duration_sec)

            for attempt, duration_sec_add in enumerate(duration_candidates, start=1):
                progress(f"    FFmpeg attempt {attempt}: duration={duration_sec_add:.6f}s")
                if not export_trimmed_video_ffmpeg(str(video.file_path), str(video_out), start_sec_add, duration_sec_add):
                    continue
                out_frames = _get_frame_count(video_out)
                if _is_valid_video_output(video_out) and (out_frames is None or abs(out_frames - target_frames) <= 1):
                    ffmpeg_add_ok = True
                    progress(f"    [OK] FFmpeg export successful")
                    break

            if not ffmpeg_add_ok:
                progress("    Stream-copy failed; using FFmpeg re-encode")
                pad_frames = max(0, target_frames - source_len)
                if export_trimmed_video_ffmpeg_reencode(
                    str(video.file_path), str(video_out), start_sec_add, target_frames, video.fps,
                    pad_frames=pad_frames,
                    target_bitrate_kbps=additional_reference_bitrate,
                    constant_bitrate=True,
                ):
                    out_frames = _get_frame_count(video_out)
                    if _is_valid_video_output(video_out) and out_frames is not None and abs(out_frames - target_frames) <= 1:
                        ffmpeg_add_ok = True
                        progress(f"    [OK] FFmpeg re-encode successful")

                if not ffmpeg_add_ok:
                    success, frames_written, expected_frames = export_trimmed_video_opencv(
                        str(video.file_path),
                        str(video_out),
                        video_start_frame,
                        video_end_frame,
                        video.fps,
                        target_frame_count=target_frames,
                    )
                    if success and _is_valid_video_output(video_out):
                        ffmpeg_add_ok = True
                        progress(f"    [OK] OpenCV fallback successful ({frames_written}/{expected_frames} frames)")

        if ffmpeg_add_ok:
            results[f'additional_video_{i+1}'] = str(video_out)
        else:
            progress(f"    [FAIL] Additional video failed for {video.name}")

    progress("=" * 60)
    progress("Export complete!")
    return results


def export_project_metadata(project: ProjectData, output_folder: Path, trim_index: Optional[int] = None) -> Path:
    """Export metadata file describing the project."""
    metadata = {
        'project_name': project.name,
        'primary_video': project.primary_video.name if project.primary_video else None,
        'primary_video_file': str(project.primary_video.file_path) if project.primary_video else None,
        'additional_video_files': [str(v.file_path) for v in project.additional_videos],
        'sensor_data_files': {
            sensor_name: str(sensor_data.file_path)
            for sensor_name, sensor_data in project.sensor_data.items()
        },
    }
    
    if project.primary_video:
        start_frame, end_frame = project.primary_video.get_trimmed_range()
        start_ns, end_ns = project.get_trimmed_time_range()
        metadata.update({
            'trim_start_frame': start_frame,
            'trim_end_frame': end_frame,
            'trim_start_timestamp_ns': start_ns,
            'trim_end_timestamp_ns': end_ns,
            'fps': project.primary_video.fps,
            'duration_seconds': project.get_trimmed_duration(),
        })
    
    import json

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if trim_index is None:
        existing_indices = []
        for path in output_folder.glob('export_metadata_trim*.json'):
            match = re.search(r'_trim(\d+)$', path.stem)
            if match:
                try:
                    existing_indices.append(int(match.group(1)))
                except ValueError:
                    pass
        trim_index = (max(existing_indices) + 1) if existing_indices else 1

    # Guarantee uniqueness in case trim_index is already present
    candidate_index = max(1, int(trim_index))
    metadata_path = output_folder / f"export_metadata_trim{candidate_index}.json"
    while metadata_path.exists():
        candidate_index += 1
        metadata_path = output_folder / f"export_metadata_trim{candidate_index}.json"

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata_path
