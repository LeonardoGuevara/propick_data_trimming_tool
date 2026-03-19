# ProPick Data Trimming Tool

Desktop GUI tool for trimming, synchronizing, and exporting eye-tracking session data:

- Primary eye-tracking scene video (`world.mp4`) with optional gaze overlay
- Up to 5 additional camera videos (chest, trolley, or custom labels)
- Sensor data files (`fixations`, `saccades`, `gaze`, `gaze_positions`, `imu`, `world_timestamps`, `blinks`, `3d_eye_states`)

## Requirements

- Python 3.10+
- FFmpeg and ffprobe available in PATH

Install FFmpeg on Windows:
```powershell
winget install ffmpeg
```

Install Python dependencies:
```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

## Typical Workflow

1. **File → Load Eye Tracking Data** — select either the Pupil Neon/Labs export folder (e.g. `exports/001`) or its parent session folder. The app auto-resolves the best child folder scored on sensor file count + video presence.
2. **File → Add Video File** — add additional camera feeds (`.mp4`, `.avi`, `.mov`). A camera-type dialog appears:
   - `chest`, `trolley_picker`, `trolley_crop`, or `custom` (free-text label)
   - Up to 5 additional videos can be loaded.
3. Set trim ranges in the **Sync & Trim** panel using frame-accurate start/end controls.
4. Click **Export Trimmed Data** — a preview dialog shows per-video trim summaries and FPS-sync equations before you confirm.
5. Choose an output folder and enter **Picker ID** + **Condition** in the naming dialog.

## Export Naming

All output files follow the pattern:

```
{pickerID}_{camera}_{condition}.mp4
```

**Condition** is selected from a preset list or entered as free text:
- `left_direction`, `right_direction`, `no_intervention`, `generic_intervention`, `personalised_intervention`

**Camera** is always `eyes` for the primary video and the user-specified label (e.g. `chest`, `trolley_picker`) for additional videos.

Examples for `picker_id = "P001"`, `condition = "left_direction"`:

| File | Description |
|---|---|
| `P001_eyes_left_direction.mp4` | Primary trimmed video |
| `P001_eyes_left_direction_gaze_overlay.mp4` | Gaze overlay (optional) |
| `P001_eyes_fixations_left_direction.csv` | Fixations sensor data |
| `P001_eyes_gaze_positions_left_direction.csv` | Gaze positions sensor data |
| `P001_eyes_timestamps_left_direction.csv` | World timestamps |
| `P001_chest_left_direction.mp4` | Additional video (camera_type = chest) |
| `export_metadata_left_direction.json` | Export metadata |

If any output file already exists, a prompt asks whether to overwrite, skip, or cancel the entire export.

## Export Behavior

### Primary video

- Re-encoded to H.264 (`h264_nvenc` with automatic CPU fallback to `libx264`).
- HEVC/H.265 source is explicitly detected and re-encoded for Windows compatibility.
- Bitrate is capped to the source bitrate (output never exceeds source quality).

### Additional videos — FPS-Sync

Each additional video is re-encoded so its **output duration exactly matches the primary video's duration**, while keeping the exact user-selected frame range intact:

```
adjusted_fps = selected_frame_count / primary_duration_sec
```

Audio is dropped on FPS-synced outputs (audio tempo cannot be preserved with frame-rate changes). The preview dialog shows the equation per video before confirming:

```
chest (GX010042) (source 59.940060 fps):
  3596 frames (60.000s) → output at 59.850149 fps (−0.089911 delta) → 60.082s
```

### Encoder fallback chain

1. `h264_nvenc` (GPU) — tried first if available
2. `libx264` (CPU) — fallback
3. OpenCV frame-by-frame — **disabled** in the current pipeline (produces large, low-quality output)

After encoding, every output file is validated against expected frame count, FPS, and duration. If validation fails, the export is retried at a lower target bitrate before reporting failure.

### Gaze overlay

An optional gaze overlay video can be generated from the exported primary clip. Each frame is matched to the nearest gaze sample (within 0.1 s) and a red filled circle (radius 15, white border) is drawn at the normalized gaze coordinate. The overlay is muxed back with the original audio.

### Sensor data

All sensor CSV/Excel files are trimmed to the primary video's trim window (nanosecond timestamps). Interval sensors (fixations, saccades, blinks) use an overlap filter; point sensors (gaze, IMU) use an inclusive range filter.

## Metadata Schema

Each `export_metadata_{condition}.json` records:

```json
{
  "primary_video_file": "<absolute source path>",
  "additional_video_files": ["<path>", "..."],
  "sensor_data_files": {
    "fixations": "<path>",
    "saccades": "<path>",
    "gaze": "<path>",
    "..."
  },
  "video_trimming": {
    "eyes (world)": {
      "trim_start_frame": 120,
      "trim_end_frame": 3600,
      "trim_start_timestamp_ns": 1234567890,
      "trim_end_timestamp_ns": 9876543210,
      "fps": 30.000333,
      "duration_seconds": 114.93,
      "source_video_file": "<absolute source path>",
      "exported_video_file": "<absolute output path>"
    },
    "chest (GX010042)": { "..." }
  }
}
```

`fps` and `duration_seconds` in `video_trimming` are measured from the actual exported file via ffprobe when available.

## Troubleshooting

- **Sensor CSVs missing** — verify sensor files were found after loading (status text updates to list loaded sensor types).
- **Export validation failure** — check the `[DIAG]` log lines printed to the console. They include expected vs. actual frame count/FPS/duration and the specific check that failed, plus the full FFmpeg encoder attempt log (encoder tried, command, failure category, stderr).
- **NVENC errors** — the tool automatically falls back to `libx264`. Common NVENC failure categories: `nvenc_unavailable` (no CUDA-capable GPU), `nvenc_session_limit` (too many concurrent GPU sessions), `gpu_runtime_error` (CUDA runtime error).
- **Permission / disk space errors** — check the output folder. The failure classifier will report `filesystem_permission` or `filesystem_space` in the `[DIAG]` output.
