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
                   sync_fps: bool = False,
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

    def _get_duration_sec(path: Path):
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                return None
            value = (result.stdout or '').strip().splitlines()
            if not value:
                return None
            return float(value[0])
        except Exception:
            return None

    def _is_timing_match(path: Path, expected_frames: int, expected_fps: float,
                         expected_duration_sec: Optional[float] = None,
                         frame_tolerance: int = 2) -> bool:
        if not _is_valid_video_output(path):
            return False

        out_frames = _get_frame_count(path)
        if out_frames is not None and abs(out_frames - expected_frames) <= frame_tolerance:
            return True

        target_duration = expected_duration_sec
        if target_duration is None and expected_fps and expected_fps > 0:
            target_duration = expected_frames / expected_fps

        if target_duration is None or target_duration <= 0:
            return out_frames is None

        out_duration = _get_duration_sec(path)
        if out_duration is None:
            return False

        duration_tolerance = max(0.04, (2.0 / expected_fps) if expected_fps and expected_fps > 0 else 0.08)
        return abs(out_duration - target_duration) <= duration_tolerance

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

    primary_fps_precise = _probe_video_fps_precise(str(primary_video.file_path), primary_video.fps)
    start_sec = start_frame / primary_fps_precise
    expected_primary_frames = end_frame - start_frame + 1
    expected_primary_duration_sec = expected_primary_frames / primary_fps_precise
    ffmpeg_ok = False

    # Primary export is always re-encoded for frame-accurate cuts.
    # Stream-copy can start on previous keyframes and produce visible decode artifacts
    # or timeline jumps at the beginning of trimmed clips.
    if primary_video.fps and primary_video.fps > 0 and expected_primary_frames > 0:
        if export_trimmed_video_ffmpeg_reencode(
            str(primary_video.file_path), str(primary_out), start_sec,
            expected_primary_frames, primary_video.fps, pad_frames=0,
            constant_bitrate=False,
        ):
            out_frames = _get_frame_count(primary_out)
            if _is_timing_match(
                primary_out,
                expected_primary_frames,
                primary_fps_precise,
                expected_duration_sec=expected_primary_duration_sec,
                frame_tolerance=2,
            ):
                ffmpeg_ok = True
                results['primary_video'] = str(primary_out)
                progress(f"  [OK] FFmpeg re-encode successful ({out_frames}/{expected_primary_frames} frames)")
            elif _is_valid_video_output(primary_out) and out_frames is None:
                ffmpeg_ok = True
                results['primary_video'] = str(primary_out)
                progress(f"  [OK] FFmpeg re-encode successful (expected {expected_primary_frames} frames)")

    if not ffmpeg_ok:
        progress("  Primary re-encode did not validate on first try; retrying with safer settings")
        if export_trimmed_video_ffmpeg_reencode(
            str(primary_video.file_path), str(primary_out), start_sec,
            expected_primary_frames, primary_video.fps, pad_frames=0,
            constant_bitrate=False,
        ):
            out_frames = _get_frame_count(primary_out)
            if _is_timing_match(
                primary_out,
                expected_primary_frames,
                primary_fps_precise,
                expected_duration_sec=expected_primary_duration_sec,
                frame_tolerance=2,
            ):
                ffmpeg_ok = True
                results['primary_video'] = str(primary_out)
                progress(f"  [OK] FFmpeg re-encode retry successful")

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
    primary_duration_sec = _compute_primary_trim_duration_sec(
        primary_video,
        start_frame,
        end_frame,
        start_ns,
        end_ns,
    )

    for i, video in enumerate(project.additional_videos):
        progress(f"  Video {i+2}: {video.name}")
        # Probe each video's own bitrate to avoid using a mismatched reference value
        additional_reference_bitrate = _probe_video_bitrate_kbps(str(video.file_path))

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

        # ------ determine target_frames and adjusted output FPS ------
        if sync_fps:
            # FPS-Sync mode: keep the exact trim range; adjust output FPS so that
            # duration equals the primary video duration (no padding / truncation).
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
        else:
            # Trim mode: compute target_frames from primary duration × this video's fps;
            # shorter clips are padded, longer ones truncated.
            adjusted_out_fps = None  # use source fps
            if source_fps_precise > 0:
                target_frames = int(round(primary_duration_sec * source_fps_precise))
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

        start_sec_add = video_start_frame / source_fps_precise if source_fps_precise > 0 else 0.0
        ffmpeg_add_ok = False

        # Detect HEVC (must re-encode regardless of mode)
        video_is_hevc = _detect_video_codec_needs_conversion(str(video.file_path))

        # In FPS-Sync mode we always re-encode (stream copy cannot change declared FPS),
        # regardless of the source codec.
        needs_reencode = video_is_hevc or sync_fps

        if video_is_hevc:
            progress(f"    Detected HEVC codec (not Windows-compatible), forcing re-encode to H.264")

        if needs_reencode:
            # ---- Re-encode path (HEVC conversion OR FPS-Sync mode) ----
            pad = 0 if sync_fps else max(0, target_frames - source_len)
            if source_fps_precise > 0 and target_frames > 0:
                if export_trimmed_video_ffmpeg_reencode(
                    str(video.file_path), str(video_out), start_sec_add, target_frames, source_fps_precise,
                    pad_frames=pad,
                    target_bitrate_kbps=additional_reference_bitrate,
                    # VBR/CRF path avoids avoidable bloat versus strict CBR.
                    constant_bitrate=False,
                    output_fps=adjusted_out_fps,
                ):
                    out_frames = _get_frame_count(video_out)
                    tolerance = 2 if sync_fps else 4
                    expected_duration = primary_duration_sec if sync_fps else None
                    if _is_timing_match(
                        video_out,
                        target_frames,
                        adjusted_out_fps if sync_fps else source_fps_precise,
                        expected_duration_sec=expected_duration,
                        frame_tolerance=tolerance,
                    ):
                        ffmpeg_add_ok = True
                        progress(f"    [OK] FFmpeg re-encode successful")

                if not ffmpeg_add_ok:
                    progress("    Initial re-encode did not produce a valid output; retrying with safer VBR settings")
                    safer_bitrate = min(int(additional_reference_bitrate or 8000), 12000)
                    if export_trimmed_video_ffmpeg_reencode(
                        str(video.file_path), str(video_out), start_sec_add, target_frames, source_fps_precise,
                        pad_frames=pad,
                        target_bitrate_kbps=safer_bitrate,
                        constant_bitrate=False,
                        output_fps=adjusted_out_fps,
                    ):
                        out_frames = _get_frame_count(video_out)
                        tolerance = 2 if sync_fps else 4
                        expected_duration = primary_duration_sec if sync_fps else None
                        if _is_timing_match(
                            video_out,
                            target_frames,
                            adjusted_out_fps if sync_fps else source_fps_precise,
                            expected_duration_sec=expected_duration,
                            frame_tolerance=tolerance,
                        ):
                            ffmpeg_add_ok = True
                            progress(f"    [OK] FFmpeg safer retry successful")

                if not ffmpeg_add_ok:
                    progress("    FFmpeg retries failed; falling back to OpenCV export")
                    success, frames_written, expected_frames = export_trimmed_video_opencv(
                        str(video.file_path),
                        str(video_out),
                        video_start_frame,
                        video_end_frame,
                        source_fps_precise,
                        target_frame_count=target_frames,
                    )
                    if success and _is_valid_video_output(video_out):
                        ffmpeg_add_ok = True
                        progress(f"    [OK] OpenCV fallback successful ({frames_written}/{expected_frames} frames)")
        else:
            # ---- Trim mode, H.264: try stream copy first ----
            duration_candidates = []
            if video.fps and video.fps > 0 and target_frames > 0:
                for frame_delta in [-1, 0, 1, -2, 2]:
                    candidate_frames = max(target_frames + frame_delta, 1)
                    duration_candidates.append(candidate_frames / source_fps_precise)
            else:
                duration_candidates.append(primary_duration_sec)

            for attempt, duration_sec_add in enumerate(duration_candidates, start=1):
                progress(f"    FFmpeg attempt {attempt}: duration={duration_sec_add:.9f}s")
                if not export_trimmed_video_ffmpeg(str(video.file_path), str(video_out), start_sec_add, duration_sec_add):
                    continue
                out_frames = _get_frame_count(video_out)
                if _is_timing_match(
                    video_out,
                    target_frames,
                    source_fps_precise,
                    expected_duration_sec=None,
                    frame_tolerance=4,
                ):
                    ffmpeg_add_ok = True
                    progress(f"    [OK] FFmpeg export successful")
                    break

            if not ffmpeg_add_ok:
                progress("    Stream-copy failed; using FFmpeg re-encode")
                pad_frames = max(0, target_frames - source_len)
                if export_trimmed_video_ffmpeg_reencode(
                    str(video.file_path), str(video_out), start_sec_add, target_frames, source_fps_precise,
                    pad_frames=pad_frames,
                    target_bitrate_kbps=additional_reference_bitrate,
                    constant_bitrate=False,
                ):
                    out_frames = _get_frame_count(video_out)
                    if _is_timing_match(
                        video_out,
                        target_frames,
                        source_fps_precise,
                        expected_duration_sec=None,
                        frame_tolerance=4,
                    ):
                        ffmpeg_add_ok = True
                        progress(f"    [OK] FFmpeg re-encode successful")

                if not ffmpeg_add_ok:
                    success, frames_written, expected_frames = export_trimmed_video_opencv(
                        str(video.file_path),
                        str(video_out),
                        video_start_frame,
                        video_end_frame,
                        source_fps_precise,
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
