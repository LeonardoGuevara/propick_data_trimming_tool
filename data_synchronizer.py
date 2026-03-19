"""
Video and sensor data synchronization utilities.
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from models import VideoData, SensorData, ProjectData


_FFMPEG_ENCODER_CACHE = {}


def _ffmpeg_has_encoder(encoder_name: str) -> bool:
    import subprocess
    import re

    cached = _FFMPEG_ENCODER_CACHE.get(encoder_name)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            _FFMPEG_ENCODER_CACHE[encoder_name] = False
            return False

        found = bool(re.search(rf'\b{re.escape(encoder_name)}\b', result.stdout or ''))
        _FFMPEG_ENCODER_CACHE[encoder_name] = found
        return found
    except Exception:
        _FFMPEG_ENCODER_CACHE[encoder_name] = False
        return False


def _classify_ffmpeg_failure(stderr_text: str, timed_out: bool = False) -> str:
    """Return a stable diagnostic category for FFmpeg failures."""
    if timed_out:
        return "timeout"

    text = (stderr_text or "").lower()
    if not text:
        return "unknown"

    if "permission denied" in text or "access is denied" in text:
        return "filesystem_permission"
    if "no space left on device" in text or "disk full" in text:
        return "filesystem_space"
    if "cannot open" in text or "error opening output" in text:
        return "filesystem_output_open"
    if "no such file or directory" in text:
        return "filesystem_path"

    if "no capable devices found" in text or "nvenc" in text and "cannot" in text:
        return "nvenc_unavailable"
    if "too many concurrent sessions" in text:
        return "nvenc_session_limit"
    if "cuda" in text and "error" in text:
        return "gpu_runtime_error"

    if "invalid argument" in text:
        return "invalid_arguments"
    if "error while decoding" in text or "invalid data found" in text or "corrupt" in text:
        return "decode_error"

    return "ffmpeg_error"


def synchronize_additional_video(primary_video: VideoData, additional_video: VideoData, 
                                  sync_frame: int) -> int:
    """
    Synchronize an additional video with the primary video.
    
    Args:
        primary_video: The eye tracking video
        additional_video: Another video to sync
        sync_frame: Frame in primary video to use as sync point
    
    Returns:
        Offset in frames for the additional video
    """
    # Get timestamp from primary video at sync point
    primary_ts = primary_video.get_frame_timestamp(sync_frame)
    
    # Find closest frame in additional video with similar timestamp
    # This is a simple approach - could be refined based on actual timestamps
    additional_ts_at_0 = additional_video.get_frame_timestamp(0)
    ts_diff = primary_ts - additional_ts_at_0
    
    # Convert timestamp difference to frames using additional video's fps
    offset_frames = int(ts_diff / (1e9 / additional_video.fps))
    
    return offset_frames


def trim_sensor_data(sensor_data: SensorData, start_ns: int, end_ns: int) -> pd.DataFrame:
    """
    Trim sensor data to fall within time range.
    
    Args:
        sensor_data: The sensor data to trim
        start_ns: Start timestamp in nanoseconds
        end_ns: End timestamp in nanoseconds
    
    Returns:
        Trimmed DataFrame
    """
    df = sensor_data.data.copy()
    
    # Case 1: Single timestamp column
    if sensor_data.timestamp_col and sensor_data.timestamp_col in df.columns:
        mask = (df[sensor_data.timestamp_col] >= start_ns) & (df[sensor_data.timestamp_col] <= end_ns)
        return df.loc[mask].reset_index(drop=True)
    
    # Case 2: Start and end timestamp columns (events with duration)
    if sensor_data.start_timestamp_col and sensor_data.end_timestamp_col:
        if sensor_data.start_timestamp_col in df.columns and sensor_data.end_timestamp_col in df.columns:
            # Include events that overlap with trim range
            mask = (df[sensor_data.end_timestamp_col] >= start_ns) & (df[sensor_data.start_timestamp_col] <= end_ns)
            return df.loc[mask].reset_index(drop=True)
    
    # No timestamp columns found
    return df.reset_index(drop=True)


def map_frame_to_additional_video(primary_frame: int, primary_video: VideoData, 
                                   additional_video: VideoData) -> int:
    """
    Map a frame index from primary video to corresponding frame in additional video.
    Takes into account potential fps differences and synchronization offset.
    """
    # Get timestamp from primary video
    primary_ts = primary_video.get_frame_timestamp(primary_frame)
    
    # Find closest frame in additional video with this timestamp
    if additional_video.world_timestamps is not None and len(additional_video.world_timestamps) > 0:
        # Find closest timestamp
        ts_array = additional_video.world_timestamps
        closest_idx = np.argmin(np.abs(ts_array - primary_ts))
        return int(closest_idx)
    else:
        # Fallback: use frame count and duration ratio
        primary_duration = primary_video.frame_count / primary_video.fps
        additional_duration = additional_video.frame_count / additional_video.fps
        ratio = additional_duration / primary_duration if primary_duration > 0 else 1.0
        return min(int(primary_frame * ratio), additional_video.frame_count - 1)


def export_trimmed_video_ffmpeg(input_path: str, output_path: str, start_sec: float, 
                                 duration_sec: float) -> bool:
    """
    Export trimmed video using FFmpeg with stream copy (preserves quality and audio).
    
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    from pathlib import Path
    
    try:
        # Check input file exists
        if not Path(input_path).exists():
            print(f"Error: Input file not found: {input_path}")
            return False
        
        # Ensure output path is string with proper quotes for Windows
        output_path_str = str(output_path)
        
        # Build command - use -ss before -i for faster seeking, then -c copy for stream copy
        # Both video and audio streams are copied without modification for maximum quality/minimum size
        end_sec = start_sec + duration_sec
        # Use -t (duration) when seeking with -ss to ensure correct output length
        # Add -fflags +genpts for more accurate frame count reporting
        cmd = [
            'ffmpeg',
            '-ss', f'{start_sec:.9f}',  # Seek start position
            '-i', str(input_path),
            '-t', f'{duration_sec:.9f}',  # Use duration
            '-fflags', '+genpts',  # Generate presentation timestamps for accurate frame count
            '-avoid_negative_ts', 'make_zero',  # Normalize output timeline to start at 0
            '-c:v', 'copy',  # Copy video stream unchanged (preserves original codec)
            '-c:a', 'copy',  # Copy audio stream unchanged (preserves original codec)
            '-y',  # Overwrite output
            output_path_str
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"FFmpeg error (code {result.returncode}):")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            return False
        
        # Verify output file was created and has content
        if not Path(output_path_str).exists():
            print(f"Error: Output file was not created")
            return False
        
        output_size = Path(output_path_str).stat().st_size
        if output_size < 1000:
            print(f"Warning: Output file is suspiciously small: {output_size} bytes")
            return False
        
        return True
    except FileNotFoundError:
        print(f"Error: ffmpeg not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print(f"Error: FFmpeg timed out")
        return False
    except Exception as e:
        print(f"FFmpeg error: {type(e).__name__}: {e}")
        return False


def export_trimmed_video_ffmpeg_reencode(input_path: str, output_path: str, start_sec: float,
                                         target_frames: int, fps: float, pad_frames: int = 0,
                                         target_bitrate_kbps: int | None = None,
                                         constant_bitrate: bool = False,
                                         output_fps: float | None = None,
                                         diagnostics: Optional[Dict[str, Any]] = None) -> bool:
    """
    Export trimmed video using FFmpeg re-encoding to guarantee frame creation when stream-copy
    seeking cannot hit the exact target reliably.

    When ``output_fps`` is supplied the selected source frames are preserved exactly and only
    their timing is changed so the total duration matches the primary video exactly.

    Attempts to match the original codec and bitrate to minimise quality loss and file size.

    Args:
        diagnostics: Optional dict populated with failure/success details for UI logging.

    Returns:
        True if successful, False otherwise
    """
    import subprocess
    from pathlib import Path

    diag = diagnostics if isinstance(diagnostics, dict) else None
    if diag is not None:
        diag.clear()
        diag.update({
            'status': 'starting',
            'input_path': str(input_path),
            'output_path': str(output_path),
            'start_sec': float(start_sec),
            'target_frames': int(target_frames),
            'source_fps': float(fps),
            'attempts': [],
        })

    try:
        if not Path(input_path).exists():
            print(f"Error: Input file not found: {input_path}")
            if diag is not None:
                diag.update({
                    'status': 'failed',
                    'failure_category': 'input_missing',
                    'last_error': f'Input file not found: {input_path}',
                })
            return False
        if target_frames <= 0 or fps <= 0:
            print(f"Error: Invalid target_frames ({target_frames}) or fps ({fps})")
            if diag is not None:
                diag.update({
                    'status': 'failed',
                    'failure_category': 'invalid_parameters',
                    'last_error': f'Invalid target_frames ({target_frames}) or fps ({fps})',
                })
            return False

        # Probe original video for codec and bitrate information.
        # Use a three-level probe: stream -> container -> file-size/duration.
        source_bitrate_kbps = None

        try:
            # Level 1: stream-level bitrate (often N/A for H.264 in MP4)
            probe_stream_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,bit_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(input_path)
            ]
            probe_result = subprocess.run(probe_stream_cmd, capture_output=True, text=True, timeout=10)
            if probe_result.returncode == 0:
                lines = [l.strip() for l in probe_result.stdout.splitlines() if l.strip()]
                if lines:
                    codec_name = lines[0].lower()
                    if 'hevc' in codec_name or 'h265' in codec_name:
                        print("Original codec is HEVC (not Windows-compatible), converting to H.264")
                        if diag is not None:
                            diag['input_codec'] = codec_name
                if len(lines) >= 2 and lines[1].isdigit() and int(lines[1]) > 0:
                    source_bitrate_kbps = max(500, int(int(lines[1]) / 1000))
        except Exception as e:
            print(f"Warning: Could not probe original codec: {e}")

        if source_bitrate_kbps is None:
            # Level 2: container/format-level bitrate
            try:
                probe_fmt_cmd = [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=bit_rate',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(input_path)
                ]
                fmt_result = subprocess.run(probe_fmt_cmd, capture_output=True, text=True, timeout=10)
                if fmt_result.returncode == 0:
                    val = fmt_result.stdout.strip()
                    if val.isdigit() and int(val) > 0:
                        total_kbps = max(500, int(int(val) / 1000))
                        source_bitrate_kbps = max(300, total_kbps - 128)  # subtract audio overhead
            except Exception:
                pass

        if source_bitrate_kbps is None:
            # Level 3: derive from file size and duration
            try:
                dur_sec = target_frames / fps  # rough estimate
                file_bytes = Path(input_path).stat().st_size
                if dur_sec > 0 and file_bytes > 0:
                    total_kbps = int((file_bytes * 8) / (dur_sec * 1000))
                    source_bitrate_kbps = max(300, total_kbps - 128)
            except Exception:
                pass

        if source_bitrate_kbps is None:
            source_bitrate_kbps = 5000  # conservative last-resort fallback

        # Cap effective_bitrate to the source so the output is never larger than the source.
        # If the caller requested a specific bitrate it still must not exceed the source.
        requested_kbps = int(target_bitrate_kbps) if target_bitrate_kbps else source_bitrate_kbps
        effective_bitrate = max(500, min(requested_kbps, source_bitrate_kbps))
        print(f"Source bitrate: {source_bitrate_kbps} kbps, effective encode bitrate: {effective_bitrate} kbps")
        output_file = Path(output_path)

        # Effective output FPS: use override when provided (FPS-Sync mode), else keep source fps
        effective_out_fps = output_fps if (output_fps is not None and output_fps > 0) else fps
        sync_retime_mode = output_fps is not None and output_fps > 0 and abs(effective_out_fps - fps) > 1e-6

        source_segment_duration_sec = target_frames / fps
        target_duration_sec = target_frames / effective_out_fps if (output_fps is not None and output_fps > 0) else source_segment_duration_sec
        timeout_sec = int(max(1800, min(14400, target_duration_sec * 6 + 600)))

        if sync_retime_mode:
            # PTS-STARTPTS normalises the first decoded frame's timestamp to 0 before
            # scaling, so setpts works correctly for any source codec/PTS offset.
            pts_scale = fps / effective_out_fps
            video_filter = f'setpts={pts_scale:.12f}*(PTS-STARTPTS)'
        else:
            video_filter = f'fps={effective_out_fps:.9f}'

        if pad_frames > 0:
            pad_seconds = pad_frames / effective_out_fps
            video_filter = f'{video_filter},tpad=stop_mode=add:stop_duration={pad_seconds:.9f}'

        base_cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-nostats',
            '-err_detect', 'ignore_err',
            '-fflags', '+discardcorrupt',
            '-ss', f'{start_sec:.9f}',
            '-t', f'{source_segment_duration_sec:.9f}',
            '-i', str(input_path),
        ]

        encoder_candidates = []
        if _ffmpeg_has_encoder('h264_nvenc'):
            encoder_candidates.append('h264_nvenc')
        encoder_candidates.append('libx264')

        maxrate_vbr = effective_bitrate  # hard cap: never exceed source bitrate
        bufsize = max(effective_bitrate * 2, 1000)
        last_error = ""
        last_failure_category = "unknown"

        if diag is not None:
            diag.update({
                'status': 'encoding',
                'effective_out_fps': float(effective_out_fps),
                'sync_retime_mode': bool(sync_retime_mode),
                'source_segment_duration_sec': float(source_segment_duration_sec),
                'target_duration_sec': float(target_duration_sec),
                'source_bitrate_kbps': int(source_bitrate_kbps),
                'effective_bitrate_kbps': int(effective_bitrate),
                'encoder_candidates': list(encoder_candidates),
                'video_filter': video_filter,
                'timeout_sec': int(timeout_sec),
            })

        for video_encoder in encoder_candidates:
            cmd = list(base_cmd)
            cmd += [
                '-vf', video_filter,
                '-frames:v', str(int(target_frames)),
                '-r', f'{effective_out_fps:.9f}',
                '-c:v', video_encoder,
                '-pix_fmt', 'yuv420p',
            ]

            if video_encoder == 'h264_nvenc':
                cmd += ['-preset', 'p4']
                if constant_bitrate:
                    cmd += [
                        '-rc', 'cbr',
                        '-b:v', f'{effective_bitrate}k',
                        '-minrate', f'{effective_bitrate}k',
                        '-maxrate', f'{effective_bitrate}k',
                        '-bufsize', f'{bufsize}k',
                    ]
                else:
                    # CRF-like quality control with hard bitrate ceiling
                    cmd += [
                        '-rc', 'vbr',
                        '-cq', '23',
                        '-maxrate', f'{maxrate_vbr}k',
                        '-bufsize', f'{bufsize}k',
                    ]
            else:
                if constant_bitrate:
                    cmd += [
                        '-preset', 'faster',
                        '-b:v', f'{effective_bitrate}k',
                        '-minrate', f'{effective_bitrate}k',
                        '-maxrate', f'{effective_bitrate}k',
                        '-bufsize', f'{bufsize}k',
                        '-x264-params', 'nal-hrd=cbr:force-cfr=1',
                    ]
                else:
                    # CRF 23 (default quality) with maxrate cap so output never inflates
                    cmd += [
                        '-preset', 'faster',
                        '-crf', '23',
                        '-maxrate', f'{maxrate_vbr}k',
                        '-bufsize', f'{bufsize}k',
                    ]

            if sync_retime_mode:
                # Audio cannot stay aligned when playback speed changes unless it is time-stretched.
                # To keep exact frames, exact duration, and reasonable file size, omit audio.
                cmd += [
                    '-an',
                    '-avoid_negative_ts', 'make_zero',
                    '-movflags', '+faststart',
                    '-y',
                    str(output_path),
                ]
            else:
                cmd += [
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-avoid_negative_ts', 'make_zero',
                    '-movflags', '+faststart',
                    '-y',
                    str(output_path),
                ]

            attempt_diag = {
                'encoder': video_encoder,
                'command': ' '.join(cmd),
            }
            if diag is not None:
                diag['attempts'].append(attempt_diag)

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                last_error = f"encoder {video_encoder} timed out after {timeout_sec}s"
                last_failure_category = _classify_ffmpeg_failure(last_error, timed_out=True)
                attempt_diag.update({
                    'status': 'timeout',
                    'failure_category': last_failure_category,
                    'stderr': last_error,
                })
                try:
                    output_file.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            if result.returncode != 0:
                stderr_text = (result.stderr or '').strip()
                last_error = stderr_text[:1200]
                last_failure_category = _classify_ffmpeg_failure(stderr_text)
                attempt_diag.update({
                    'status': 'failed',
                    'returncode': int(result.returncode),
                    'failure_category': last_failure_category,
                    'stderr': stderr_text[:4000],
                })
                try:
                    output_file.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            if output_file.exists() and output_file.stat().st_size >= 1000:
                if video_encoder == 'h264_nvenc':
                    print("FFmpeg re-encode used NVENC acceleration")
                attempt_diag.update({
                    'status': 'success',
                    'returncode': int(result.returncode),
                    'output_size_bytes': int(output_file.stat().st_size),
                })
                if diag is not None:
                    diag.update({
                        'status': 'success',
                        'selected_encoder': video_encoder,
                        'output_size_bytes': int(output_file.stat().st_size),
                    })
                return True

            last_error = "output missing or too small"
            last_failure_category = "output_invalid"
            attempt_diag.update({
                'status': 'failed',
                'failure_category': last_failure_category,
                'stderr': last_error,
            })
            try:
                output_file.unlink(missing_ok=True)
            except Exception:
                pass

        print("FFmpeg re-encode error: all encoder attempts failed")
        if last_error:
            print(f"  stderr: {last_error}")
        if diag is not None:
            diag.update({
                'status': 'failed',
                'failure_category': last_failure_category,
                'last_error': last_error,
            })
        return False
    except Exception as e:
        print(f"FFmpeg re-encode error: {type(e).__name__}: {e}")
        if diag is not None:
            diag.update({
                'status': 'failed',
                'failure_category': 'python_exception',
                'last_error': f'{type(e).__name__}: {e}',
            })
        try:
            from pathlib import Path
            Path(output_path).unlink(missing_ok=True)
        except Exception:
            pass
        return False


def export_trimmed_video_opencv(input_path: str, output_path: str, start_frame: int,
                                 end_frame: int, fps: float, target_frame_count: int = None,
                                 output_fps: float = None) -> tuple:
    """
    Export trimmed video using OpenCV (slower, larger file, but always available).
    Each frame is explicitly read from the specified range to ensure trimming works.
    
    Returns:
        Tuple: (success: bool, frames_written: int, expected_frames: int)
    """
    import cv2
    from pathlib import Path
    
    try:
        # Ensure output has correct extension
        output_path = str(Path(output_path).with_suffix('.mp4'))
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open input video: {input_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate frame range
        if start_frame < 0:
            start_frame = 0
        if end_frame >= total_frames:
            end_frame = total_frames - 1
        
        if start_frame > end_frame:
            print(f"Error: Invalid frame range {start_frame} to {end_frame}")
            cap.release()
            return False
        
        effective_fps = output_fps if (output_fps is not None and output_fps > 0) else fps

        print(f"OpenCV export: input={input_path}, output={output_path}, reading frames {start_frame}-{end_frame} (total {end_frame - start_frame + 1} frames) from {total_frames} total, fps={effective_fps}, target_frame_count={target_frame_count}")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))
        
        if not out.isOpened():
            print(f"Warning: mp4v codec failed, trying MJPEG fallback")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video writer for {output_path}")
            cap.release()
            return False
        
        # Extract and write frames in the specified range
        frames_written = 0
        expected_frames = max(0, end_frame - start_frame + 1)
        
        print(f"OpenCV: Seeking to frame {start_frame}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read and process each frame in the target range
        for frame_idx in range(start_frame, end_frame + 1):
            # If target_frame_count provided and we've already written enough, stop (truncate)
            if target_frame_count is not None and frames_written >= target_frame_count:
                break

            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame at position {frame_idx}")
                break
            
            # Check frame dimensions
            if frame.shape[:2] != (height, width):
                print(f"Warning: Frame shape mismatch at {frame_idx}: {frame.shape[:2]} vs {(height, width)}")
                # Resize if needed
                frame = cv2.resize(frame, (width, height))
            
            # Write frame (VideoWriter.write() returns None in OpenCV Python bindings,
            # so don't treat its return value as a success flag).
            try:
                out.write(frame)
            except Exception as e:
                print(f"Warning: Could not write frame {frame_idx}: {e}")
                continue

            frames_written += 1
            
            # Show progress every 100 frames
            if frames_written % 100 == 0:
                print(f"  OpenCV progress: {frames_written} frames written...")
        
        # If target_frame_count provided, pad with black frames to reach target while writer still open
        if target_frame_count is not None and frames_written < target_frame_count:
            try:
                import numpy as _np
                # Write black frames to reach the target count
                black_frame = _np.zeros((height, width, 3), dtype='uint8')
                while frames_written < target_frame_count:
                    out.write(black_frame)
                    frames_written += 1
            except Exception as e:
                print(f"Warning: Failed to pad with black frames: {e}")

        # If we somehow wrote more frames than the target, abort and clean up
        if target_frame_count is not None and frames_written > target_frame_count:
            print(f"Error: frames_written ({frames_written}) > target_frame_count ({target_frame_count}). Aborting export and cleaning up output file.")
            try:
                out.release()
            except Exception:
                pass
            try:
                cap.release()
            except Exception:
                pass
            try:
                Path(output_path).unlink()
            except Exception:
                pass
            return (False, frames_written, expected_frames)

        # Release resources
        out.release()
        cap.release()

        # Verify output file was created and has data
        output_size = Path(output_path).stat().st_size if Path(output_path).exists() else 0
        print(f"OpenCV export complete: {frames_written}/{expected_frames} frames written, file size: {output_size} bytes")

        # Safety: never exceed target_frame_count if provided
        if target_frame_count is not None and frames_written > target_frame_count:
            print(f"Warning: frames_written ({frames_written}) exceeded target_frame_count ({target_frame_count}), truncating output is required but cannot be performed post-write")
        
        if output_size < 10000:  # Less than 10KB is suspicious (likely corrupted)
            print(f"Error: Output file is suspiciously small ({output_size} bytes). Deleting failed export.")
            try:
                Path(output_path).unlink()  # Delete the corrupted file
            except Exception as e:
                print(f"Could not delete file: {e}")
            return False
        
        # Warn if we didn't write enough frames (significant failure)
        if frames_written < expected_frames * 0.8:  # Allow 20% margin
            print(f"Error: Only wrote {frames_written} of {expected_frames} frames (80% threshold)")
            try:
                Path(output_path).unlink()
            except:
                pass
            return (False, frames_written, expected_frames)

        return (True, frames_written, expected_frames)
    except Exception as e:
        print(f"OpenCV export error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Try to clean up any partial file
        try:
            if Path(output_path).exists():
                Path(output_path).unlink()
        except:
            pass
        return (False, 0, max(0, end_frame - start_frame + 1))


def create_gaze_overlay_video(input_path: str, output_path: str, primary_video: VideoData,
                              gaze_data: SensorData, start_frame: int, end_frame: int) -> bool:
    """
    Create a copy of video with gaze point overlays.
    
    Returns:
        True if successful, False otherwise
    """
    import cv2
    import subprocess
    from pathlib import Path
    
    try:
        # Find gaze x, y columns
        gaze_x_col = None
        gaze_y_col = None
        time_col = gaze_data.timestamp_col
        
        if not time_col:
            return False
        
        for col in gaze_data.data.columns:
            lc = col.lower()
            if 'gaze x' in lc or 'gaze_x' in lc:
                gaze_x_col = col
            if 'gaze y' in lc or 'gaze_y' in lc:
                gaze_y_col = col
        
        if not gaze_x_col or not gaze_y_col:
            return False
        
        # Pre-index gaze data for fast per-frame nearest timestamp lookup.
        gaze_df = gaze_data.data[[time_col, gaze_x_col, gaze_y_col]].copy()
        gaze_df[time_col] = pd.to_numeric(gaze_df[time_col], errors='coerce')
        gaze_df = gaze_df[gaze_df[time_col].notna()].sort_values(time_col).reset_index(drop=True)
        if len(gaze_df) == 0:
            return False

        gaze_times = gaze_df[time_col].to_numpy(dtype='float64')
        gaze_x_vals = gaze_df[gaze_x_col].to_numpy()
        gaze_y_vals = gaze_df[gaze_y_col].to_numpy()
        match_tolerance_ns = 1e8  # 0.1 seconds

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_overlay = str(Path(output_path).with_suffix('.overlay_tmp.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_overlay, fourcc, fps, (width, height))
        
        # Process each frame
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Map frame index to original primary video frame
            original_frame_idx = start_frame + frame_idx
            frame_timestamp_ns = primary_video.get_frame_timestamp(original_frame_idx)
            
            # Find nearest gaze sample by timestamp using binary search
            idx = int(np.searchsorted(gaze_times, frame_timestamp_ns, side='left'))
            best_idx = None
            best_diff = None

            if 0 <= idx < len(gaze_times):
                best_idx = idx
                best_diff = abs(gaze_times[idx] - frame_timestamp_ns)
            if idx > 0:
                prev_diff = abs(gaze_times[idx - 1] - frame_timestamp_ns)
                if best_diff is None or prev_diff < best_diff:
                    best_idx = idx - 1
                    best_diff = prev_diff

            if best_idx is not None and best_diff is not None and best_diff < match_tolerance_ns:
                gx = gaze_x_vals[best_idx]
                gy = gaze_y_vals[best_idx]
                if pd.notna(gx) and pd.notna(gy):
                    cv2.circle(frame, (int(gx), int(gy)), 15, (255, 0, 0), 2)
            
            out.write(frame)
        
        out.release()
        cap.release()

        def _probe_video_bitrate_kbps(video_path: str) -> int:
            try:
                candidates = [
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                ]
                for probe_cmd in candidates:
                    probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
                    if probe.returncode == 0:
                        raw = probe.stdout.strip()
                        if raw and raw.upper() != 'N/A':
                            bps = int(float(raw))
                            kbps = max(500, int(round(bps / 1000)))
                            return kbps

                stat_cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=size,duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', video_path,
                ]
                stat_probe = subprocess.run(stat_cmd, capture_output=True, text=True, timeout=30)
                if stat_probe.returncode == 0:
                    parts = [p.strip() for p in stat_probe.stdout.splitlines() if p.strip()]
                    if len(parts) >= 2:
                        size_bytes = float(parts[0])
                        duration_sec = float(parts[1])
                        if duration_sec > 0:
                            bps = (size_bytes * 8.0) / duration_sec
                            kbps = max(500, int(round(bps / 1000)))
                            return kbps
            except Exception:
                pass
            return 6000

        target_kbps = _probe_video_bitrate_kbps(input_path)
        # Increase bitrate slightly to maintain quality with added gaze circles
        target_kbps = int(target_kbps * 1.05)  # 5% increase for overlay content
        
        encode_candidates = []
        if _ffmpeg_has_encoder('h264_nvenc'):
            encode_candidates.append('h264_nvenc')
        encode_candidates.append('libx264')

        last_stderr = ""
        target_duration_sec = frame_count / max(fps, 0.001)
        overlay_timeout_sec = int(max(600, min(21600, target_duration_sec * 4 + 300)))
        ffmpeg_ok = False

        for encoder in encode_candidates:
            final_cmd = [
                'ffmpeg',
                '-i', temp_overlay,
                '-i', str(input_path),
                '-map', '0:v:0',
                '-map', '1:a?',
                '-c:v', encoder,
                '-b:v', f'{target_kbps}k',
                '-minrate', f'{int(target_kbps * 0.9)}k',
                '-maxrate', f'{int(target_kbps * 1.1)}k',
                '-bufsize', f'{target_kbps}k',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
            ]

            if encoder == 'h264_nvenc':
                final_cmd += ['-preset', 'p4', '-rc', 'vbr']
            else:
                final_cmd += ['-preset', 'veryfast', '-crf', '18']

            final_cmd += [
                '-y',
                str(output_path),
            ]

            try:
                ffmpeg_res = subprocess.run(final_cmd, capture_output=True, text=True, timeout=overlay_timeout_sec)
            except subprocess.TimeoutExpired:
                last_stderr = f"{encoder} timed out after {overlay_timeout_sec}s"
                continue

            if ffmpeg_res.returncode == 0:
                ffmpeg_ok = True
                break

            last_stderr = (ffmpeg_res.stderr or '')[:500]

        try:
            Path(temp_overlay).unlink(missing_ok=True)
        except Exception:
            pass

        if not ffmpeg_ok:
            print("Gaze overlay FFmpeg compression failed")
            if last_stderr:
                print(last_stderr)
            return False

        return True
    except Exception as e:
        print(f"Gaze overlay error: {e}")
        return False
