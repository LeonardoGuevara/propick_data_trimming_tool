# Eye Tracking Data Synchronization Tool

Desktop GUI tool for synchronizing, trimming, and exporting:
- Primary eye-tracking scene video (`world.mp4`)
- Up to 5 additional videos (e.g., chest camera, trolley camera)
- Sensor data files (`fixations`, `saccades`, `gaze_positions`, `imu`, `world_timestamps`, etc.)

## Requirements

- Python 3.10+
- FFmpeg and ffprobe available in PATH (recommended for fast, high-quality exports)

Install FFmpeg on Windows:
```powershell
winget install ffmpeg
```

Install Python dependencies:
```powershell
python -m pip install -r requirements.txt
```

## Run

From the project root:
```powershell
python main.py
```

## Typical Workflow

1. **Load Eye Tracking Data** from the File menu.
   - You can select either the export folder (e.g. `exports/001`) or a parent session folder.
   - The app auto-resolves the best `exports/<id>` folder when needed.
2. **Add Video File** for additional camera feeds.
3. Sync and set trim ranges in the panel.
4. Click **Export Trimmed Data**.
5. In export preview:
   - Confirm trim summary for all videos.
   - Optional checkbox: **Create gaze overlay video (slower)** (enabled by default).

## Export Behavior

- Primary and additional videos are exported to the selected output folder.
- Additional videos are matched to primary duration (seconds):
  - Longer trims are truncated.
  - Shorter trims are padded.
- HEVC/H.265 inputs are converted to H.264 for Windows compatibility.
- For compatible sources, stream copy is used when possible for speed.
- Robust fallback chain is used when needed (re-encode and OpenCV fallback).

## Performance Notes

- **Fast path**: plain trim via stream copy (very fast).
- **Slow path**: gaze overlay (decodes frames, draws overlays, re-encodes video).
- Re-encode path includes speed improvements and can use GPU encoder (`h264_nvenc`) when available, with automatic CPU fallback.

## Output Files

Example output naming:
- `world_trim1.mp4`
- `world_trim1_gaze_overlay.mp4` (if overlay enabled)
- `chest_1_trim1.mp4`, `Trolley_1_trim1.mp4`
- `fixations_trim1.csv`, `saccades_trim1.csv`, etc.
- `export_metadata_trim1.json`

Subsequent exports increment trim index (`trim2`, `trim3`, ...), avoiding overwrite.

## Metadata Schema

Each `export_metadata_trim#.json` includes:
- `project_name`
- `primary_video`, `primary_video_file`
- `additional_video_files` (full source paths)
- `sensor_data_files` (mapping: sensor key → full source path)
- trim timing/frame details (`trim_start_frame`, `trim_end_frame`, nanosecond timestamps, fps, duration)

## Troubleshooting

- If CSV files are missing after export, verify sensor files were loaded (shown in the status text after loading).
- If FFmpeg is unavailable, some operations fall back to OpenCV and may be slower/larger.
