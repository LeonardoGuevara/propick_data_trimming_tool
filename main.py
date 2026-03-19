"""
Main application window for Eye Tracking Data Synchronization Tool.
"""
import sys
import re
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from models import ProjectData, VideoData, SensorData
from data_loader import find_eye_tracking_files, load_video, load_sensor_data
from multi_video_player import MultiVideoPlayerWidget
from sync_trim_panel import SyncTrimPanel
from data_exporter import export_project, export_project_metadata, preview_export_file_paths
from gui_utils import format_time, format_fps, video_display_name, ExportNamingDialog, CameraSelectionDialog


class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""

    @staticmethod
    def _video_display_name(video: VideoData) -> str:
        """Return camera-based label for UI."""
        return video_display_name(video)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracking Data Synchronization Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        self.project = None
        self.video_player = None
        self.sync_panel = None
        self.video_visibility = {}  # Track which videos are visible
        self.additional_video_ids = {}  # Map video index to video_id in multi_video_player
        
        self._init_ui()
        self._create_menu()
    
    def _init_ui(self):
        """Initialize the main UI."""
        # Central widget
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)
        
        # Left side: Large video display (takes up most of the space)
        left_layout = QtWidgets.QVBoxLayout()
        
        # Global play controls (at top)
        self.global_controls_layout = QtWidgets.QHBoxLayout()
        
        self.global_play_button = QtWidgets.QPushButton("Play All")
        self.global_play_button.setEnabled(False)
        self.global_play_button.clicked.connect(self._on_global_play)
        self.global_controls_layout.addWidget(self.global_play_button)
        
        self.global_pause_button = QtWidgets.QPushButton("Pause All")
        self.global_pause_button.setEnabled(False)
        self.global_pause_button.clicked.connect(self._on_global_pause)
        self.global_controls_layout.addWidget(self.global_pause_button)

        self.global_frame_back_button = QtWidgets.QPushButton("All ◄ Frame")
        self.global_frame_back_button.setEnabled(False)
        self.global_frame_back_button.clicked.connect(self._on_global_frame_back)
        self.global_controls_layout.addWidget(self.global_frame_back_button)

        self.global_frame_forward_button = QtWidgets.QPushButton("Frame ► All")
        self.global_frame_forward_button.setEnabled(False)
        self.global_frame_forward_button.clicked.connect(self._on_global_frame_forward)
        self.global_controls_layout.addWidget(self.global_frame_forward_button)
        
        self.global_controls_layout.addStretch()
        left_layout.addLayout(self.global_controls_layout)
        
        # Video player area (will be populated after loading data) - takes most of left space
        self.video_container = QtWidgets.QWidget()
        self.video_container_layout = QtWidgets.QVBoxLayout(self.video_container)
        self.video_container_layout.setContentsMargins(0, 0, 0, 0)
        self.video_container.setMinimumWidth(640)
        left_layout.addWidget(self.video_container, 2)  # Stretch factor 2
        
        main_layout.addLayout(left_layout, 3)
        
        # Right side: Control panels, video visibility, and export
        right_layout = QtWidgets.QVBoxLayout()
        
        # Sync and trim panel (will be populated after loading data)
        self.panel_container = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(self.panel_container)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.panel_container)
        
        # Video visibility controls
        self.video_visibility_group = QtWidgets.QGroupBox("Video Feeds", self)
        visibility_layout = QtWidgets.QVBoxLayout(self.video_visibility_group)
        self.video_visibility_layout = visibility_layout
        right_layout.addWidget(self.video_visibility_group)
        
        right_layout.addStretch()
        
        # Export button
        self.export_button = QtWidgets.QPushButton("Export Trimmed Data")
        self.export_button.clicked.connect(self._on_export)
        self.export_button.setEnabled(False)
        right_layout.addWidget(self.export_button)
        
        # Status bar setup
        self.statusLabel = QtWidgets.QLabel("Ready. Load eye tracking data to begin.")
        self.statusLabel.setWordWrap(True)
        right_layout.addWidget(self.statusLabel)
        
        main_layout.addLayout(right_layout, 1)
        
        self.setCentralWidget(central)
    
    def _create_menu(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QtWidgets.QAction("Load Eye Tracking Data", self)
        load_action.triggered.connect(self._on_load_eye_tracking)
        file_menu.addAction(load_action)
        
        load_video_action = QtWidgets.QAction("Add Video File", self)
        load_video_action.triggered.connect(self._on_add_video)
        file_menu.addAction(load_video_action)
        
        file_menu.addSeparator()
        
        remove_video_action = QtWidgets.QAction("Remove Additional Video", self)
        remove_video_action.triggered.connect(self._on_remove_video)
        file_menu.addAction(remove_video_action)
        
        remove_data_action = QtWidgets.QAction("Clear All Data", self)
        remove_data_action.triggered.connect(self._on_clear_all_data)
        file_menu.addAction(remove_data_action)
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QtWidgets.QAction("About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _on_load_eye_tracking(self):
        """Load eye tracking data folder."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Eye Tracking Data Folder"
        )
        
        if not folder:
            return
        
        folder_path = Path(folder)
        
        try:
            # Find available files
            file_dict = find_eye_tracking_files(folder_path)
            
            if not file_dict['videos']:
                QtWidgets.QMessageBox.warning(self, "Error", "No .mp4 files found in folder")
                return
            
            resolved_folder = file_dict.get('resolved_folder', folder_path)

            # Create project
            self.project = ProjectData(
                name=resolved_folder.name,
                project_folder=resolved_folder
            )
            
            # Load primary video (first .mp4)
            video_path = file_dict['videos'][0]
            primary_video = load_video(video_path, is_primary=True)
            self.project.primary_video = primary_video
            
            # Load sensor data
            sensor_files = {
                'fixations': file_dict['fixations'],
                'saccades': file_dict['saccades'],
                'gaze': file_dict['gaze'] or file_dict['gaze_positions'],
                'imu': file_dict['imu'],
                'world_timestamps': file_dict['world_timestamps'],
                'blinks': file_dict['blinks'],
                '3d_eye_states': file_dict['3d_eye_states'],
            }
            
            for sensor_name, sensor_file in sensor_files.items():
                if sensor_file:
                    try:
                        sensor_data = load_sensor_data(sensor_file)
                        self.project.sensor_data[sensor_name] = sensor_data
                    except Exception as e:
                        print(f"Warning: Failed to load {sensor_name}: {e}")

            if not self.project.sensor_data:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Sensor Files Loaded",
                    "No sensor CSV/Excel files were detected in the selected folder.\n\n"
                    "Tip: Select the exported data folder (e.g. exports/000) that contains files like\n"
                    "fixations.csv, saccades.csv, gaze_positions.csv, imu.csv, and world_timestamps.csv."
                )
            
            # Setup UI with loaded data
            self._setup_with_project()
            
            self.statusLabel.setText(
                f"Loaded: {self.project.name} | "
                f"Video: {self._video_display_name(primary_video)} ({primary_video.frame_count} frames, {format_fps(primary_video.fps)} fps) | "
                f"Sensor data: {', '.join(self.project.sensor_data.keys())}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
    
    def _on_add_video(self):
        """Add an additional video file."""
        if not self.project or not self.project.primary_video:
            QtWidgets.QMessageBox.warning(self, "Warning", "Load eye tracking data first")
            return
        
        if len(self.project.additional_videos) >= 5:
            QtWidgets.QMessageBox.warning(self, "Warning", "Maximum 5 additional videos allowed")
            return
        
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        
        if not file_path:
            return
        
        try:
            # Ask for camera type
            video_filename = Path(file_path).name
            camera_dialog = CameraSelectionDialog(video_filename, self)
            if camera_dialog.exec_() != QtWidgets.QDialog.Accepted:
                self.statusLabel.setText("Video import cancelled")
                return
            
            camera_type = camera_dialog.get_camera_type()
            if not camera_type:
                QtWidgets.QMessageBox.warning(self, "Error", "No camera type selected")
                return
            
            # Load video with camera type
            additional_video = load_video(Path(file_path), is_primary=False, camera_type=camera_type)
            self.project.additional_videos.append(additional_video)
            
            # Add to multi-video player with 0-based index
            video_index = len(self.project.additional_videos) - 1
            video_id = self.video_player.add_additional_video(additional_video, video_index)
            self.additional_video_ids[video_index] = video_id
            
            # Update UI
            # Ask video player to relayout remaining videos
            if self.video_player and hasattr(self.video_player, '_relayout_videos'):
                try:
                    self.video_player._relayout_videos()
                except Exception:
                    pass

            # Update UI controls and sync panel
            self._update_video_visibility_controls()
            self._update_sync_panel()
            
            self.statusLabel.setText(f"Added video: {self._video_display_name(additional_video)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")
    
    def _setup_with_project(self):
        """Setup UI components with loaded project data."""
        # Clear previous widgets
        for i in reversed(range(self.video_container_layout.count())):
            self.video_container_layout.itemAt(i).widget().setParent(None)
        
        for i in reversed(range(self.panel_container.layout().count())):
            self.panel_container.layout().itemAt(i).widget().setParent(None)
        
        # Clear previous visibility checkboxes
        while self.video_visibility_layout.count() > 0:
            self.video_visibility_layout.itemAt(0).widget().setParent(None)
        
        # Create multi-video player for primary video
        self.video_player = MultiVideoPlayerWidget(
            self.project.primary_video
        )
        # Handle move-to-beginning requests from video displays
        self.video_player.move_to_beginning_requested.connect(self._on_move_to_beginning_requested)
        self.video_player.move_to_end_requested.connect(self._on_move_to_end_requested)
        self.video_player.frame_changed.connect(self._on_frame_changed)
        self.video_container_layout.addWidget(self.video_player)
        
        # Create sync/trim panel with reference to video player
        self.sync_panel = SyncTrimPanel(self.project, video_player=self.video_player)
        self.panel_container.layout().addWidget(self.sync_panel)
        self.sync_panel.trim_changed.connect(self._on_trim_changed)
        
        # Update video visibility controls
        self._update_video_visibility_controls()
        
        # Enable export button and global controls
        self.export_button.setEnabled(True)
        self.global_play_button.setEnabled(True)
        self.global_pause_button.setEnabled(True)
        self.global_frame_back_button.setEnabled(True)
        self.global_frame_forward_button.setEnabled(True)
    
    def _update_video_visibility_controls(self):
        """Update video visibility checkboxes."""
        # Clear existing checkboxes
        while self.video_visibility_layout.count() > 0:
            self.video_visibility_layout.itemAt(0).widget().setParent(None)
        
        self.video_visibility = {}
        
        # Primary video
        primary_checkbox = QtWidgets.QCheckBox(self._video_display_name(self.project.primary_video))
        primary_checkbox.setChecked(True)
        primary_checkbox.stateChanged.connect(lambda state: self._on_video_visibility_changed("primary", state == QtCore.Qt.Checked))
        self.video_visibility_layout.addWidget(primary_checkbox)
        self.video_visibility["primary"] = primary_checkbox
        
        # Additional videos
        for i, video in enumerate(self.project.additional_videos):
            video_id = f"video_{i}"
            video_checkbox = QtWidgets.QCheckBox(self._video_display_name(video))
            video_checkbox.setChecked(True)
            video_checkbox.stateChanged.connect(lambda state, vid_id=video_id: self._on_video_visibility_changed(vid_id, state == QtCore.Qt.Checked))
            self.video_visibility_layout.addWidget(video_checkbox)
            self.video_visibility[video_id] = video_checkbox
    
    def _update_sync_panel(self):
        """Update sync/trim panel layout."""
        if self.sync_panel:
            self.sync_panel.setParent(None)
        
        self.sync_panel = SyncTrimPanel(self.project, video_player=self.video_player)
        self.panel_container.layout().addWidget(self.sync_panel)
        self.sync_panel.trim_changed.connect(self._on_trim_changed)
    
    def _on_frame_changed(self, frame_num: int):
        """Handle frame change from video player."""
        if self.sync_panel:
            self.sync_panel.update_current_frame(frame_num)
    
    
    def _on_video_visibility_changed(self, video_id: str, visible: bool):
        """Handle video visibility toggle."""
        if self.video_player:
            self.video_player.set_video_visibility(video_id, visible)

    def _on_move_to_beginning_requested(self, video_id: str) -> None:
        """Handle move-to-beginning request from a video display.

        Move the video slider to the trim start frame.
        """
        if not self.video_player or not self.project:
            return

        # Move the video to its trim start frame
        if video_id == 'primary':
            if self.project.primary_video and 'primary' in self.video_player.video_displays:
                if self.video_player.video_displays['primary'].playing:
                    self.video_player.video_displays['primary'].toggle_play()
                start_frame = self.project.primary_video.start_frame
                self.video_player.video_displays['primary'].set_frame(int(start_frame))
        else:
            try:
                idx = int(video_id.split('_', 1)[1])
            except Exception:
                return
            if 0 <= idx < len(self.project.additional_videos):
                vid = self.project.additional_videos[idx]
                if video_id in self.video_player.video_displays:
                    if self.video_player.video_displays[video_id].playing:
                        self.video_player.video_displays[video_id].toggle_play()
                    start_frame = vid.start_frame
                    self.video_player.video_displays[video_id].set_frame(int(start_frame))

        # Update sync panel display (labels) and notify listeners
        try:
            self.sync_panel._update_display()
            self.sync_panel.trim_changed.emit()
        except Exception:
            pass

    def _on_move_to_end_requested(self, video_id: str) -> None:
        """Handle move-to-end request from a specific video display only."""
        if not self.video_player or not self.project:
            return

        if video_id == 'primary':
            if self.project.primary_video and 'primary' in self.video_player.video_displays:
                if self.video_player.video_displays['primary'].playing:
                    self.video_player.video_displays['primary'].toggle_play()
                end_frame = self.project.primary_video.end_frame
                end_frame = self.project.primary_video.frame_count - 1 if end_frame is None else end_frame
                self.video_player.video_displays['primary'].set_frame(int(end_frame))
        else:
            try:
                idx = int(video_id.split('_', 1)[1])
            except Exception:
                return
            if 0 <= idx < len(self.project.additional_videos):
                vid = self.project.additional_videos[idx]
                if video_id in self.video_player.video_displays:
                    if self.video_player.video_displays[video_id].playing:
                        self.video_player.video_displays[video_id].toggle_play()
                    end_frame = vid.end_frame
                    end_frame = vid.frame_count - 1 if end_frame is None else end_frame
                    self.video_player.video_displays[video_id].set_frame(int(end_frame))

        try:
            self.sync_panel._update_display()
            self.sync_panel.trim_changed.emit()
        except Exception:
            pass
    
    def _on_trim_changed(self):
        """Handle trim settings change."""
        if self.video_player and self.project.primary_video:
            # Force video player to update display
            primary_frame = self.video_player.get_primary_frame()
            self.video_player._on_primary_frame_changed(primary_frame)
    
    def _on_global_play(self):
        """Handle global play all button."""
        if self.video_player:
            self.video_player.play_all()
    
    def _on_global_pause(self):
        """Handle global pause all button."""
        if self.video_player:
            self.video_player.pause_all()

    def _on_global_frame_back(self):
        """Step all videos one frame backward simultaneously."""
        if self.video_player:
            self.video_player.step_all_frames(-1)

    def _on_global_frame_forward(self):
        """Step all videos one frame forward simultaneously."""
        if self.video_player:
            self.video_player.step_all_frames(1)
    
    def _on_export(self):
        """Handle export button click."""
        if not self.project or not self.project.primary_video:
            QtWidgets.QMessageBox.warning(self, "Warning", "No data loaded")
            return
        
        # Ask for output folder
        output_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder"
        )
        
        if not output_folder:
            return
        
        try:
            # Build export preview for ALL videos (primary + additional) showing resultant trimmed frames/seconds
            primary = self.project.primary_video
            p_start, p_end = primary.get_trimmed_range()
            primary_frame_count = p_end - p_start + 1
            primary_duration_sec = primary_frame_count / primary.fps
            
            # Collect info for all videos
            all_videos_info = []
            
            # Add primary video info
            all_videos_info.append({
                'index': 0,
                'name': self._video_display_name(primary),
                'target_frames': primary_frame_count,
                'target_seconds': primary_duration_sec,
                'fps': primary.fps,
                'delta_frames': 0,
                'is_primary': True
            })
            
            # Add additional videos with per-video target frames calculated from primary duration
            for i, video in enumerate(self.project.additional_videos):
                # Get the actual trimmed range for this video (use defaults if not explicitly trimmed)
                v_start = video.start_frame if video.start_frame is not None else 0
                v_end = video.end_frame if video.end_frame is not None else video.frame_count - 1
                
                # Source trimmed frames (from sync panel - what user actually trimmed)
                trimmed_frames = max(0, v_end - v_start + 1)
                trimmed_seconds = trimmed_frames / video.fps if video.fps > 0 else 0
                
                # Target frames for this additional video = primary duration (seconds) * this video's fps
                target_frames = int(round(primary_duration_sec * video.fps)) if video.fps > 0 else primary_frame_count
                target_seconds = target_frames / video.fps if video.fps > 0 else primary_duration_sec
                delta_frames = target_frames - trimmed_frames
                
                all_videos_info.append({
                    'index': i + 1,
                    'name': self._video_display_name(video),
                    'source_trimmed_frames': trimmed_frames,
                    'source_trimmed_seconds': trimmed_seconds,
                    'target_frames': target_frames,
                    'target_seconds': target_seconds,
                    'fps': video.fps,
                    'delta_frames': delta_frames,
                    'is_primary': False
                })
            
            # ---- Build FPS-Sync preview text (only export mode) ------
            def _build_preview() -> str:
                lines = []
                lines.append("Export Preview  —  FPS Sync Mode:")
                lines.append("=" * 88)
                lines.append(
                    f"{all_videos_info[0]['name']}:  {primary_frame_count} frames  "
                    f"({primary_duration_sec:.3f}s  at {format_fps(primary.fps)} fps)"
                )
                for info in all_videos_info[1:]:
                    src_frames = info['source_trimmed_frames']
                    src_sec    = info['source_trimmed_seconds']
                    src_fps    = info['fps']
                    adj_fps = src_frames / primary_duration_sec if primary_duration_sec > 0 else src_fps
                    out_sec = src_frames / adj_fps if adj_fps > 0 else 0.0
                    fps_delta = adj_fps - src_fps
                    fps_delta_text = f"+{format_fps(abs(fps_delta))}" if fps_delta >= 0 else f"-{format_fps(abs(fps_delta))}"
                    lines.append(
                        f"{info['name']}  (source {format_fps(src_fps)} fps):  "
                        f"{src_frames} frames ({src_sec:.3f}s)  →  output at {format_fps(adj_fps)} fps "
                        f"({fps_delta_text} delta)  →  {out_sec:.3f}s"
                    )
                lines.append("=" * 88)
                lines.append(
                    "Output FPS is adjusted so that the selected trim frames are the exact\n"
                    "first and last frames of each output video, with identical duration\n"
                    "to the primary video.  No padding or truncation is used."
                )
                return "\n".join(lines)

            # ---- Custom export dialog --------------------------------
            export_dialog = QtWidgets.QDialog(self)
            export_dialog.setWindowTitle("Export Preview")
            export_dialog.setMinimumWidth(760)
            dlg_layout = QtWidgets.QVBoxLayout(export_dialog)
            dlg_layout.setSpacing(10)

            # Preview text (monospaced)
            preview_label = QtWidgets.QLabel(_build_preview())
            preview_label.setWordWrap(False)
            preview_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            font = QtGui.QFont("Courier New", 9)
            preview_label.setFont(font)
            dlg_layout.addWidget(preview_label)

            # Gaze overlay checkbox
            overlay_checkbox = QtWidgets.QCheckBox("Create gaze overlay video (slower)")
            overlay_checkbox.setChecked(True)
            dlg_layout.addWidget(overlay_checkbox)

            # OK / Cancel buttons
            btn_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            btn_box.button(QtWidgets.QDialogButtonBox.Ok).setText("Export")
            btn_box.accepted.connect(export_dialog.accept)
            btn_box.rejected.connect(export_dialog.reject)
            dlg_layout.addWidget(btn_box)

            if export_dialog.exec_() != QtWidgets.QDialog.Accepted:
                return

            export_gaze_overlay = overlay_checkbox.isChecked()

            # ---- Collect enabled videos ----
            enabled_video_indices = []
            if self.video_visibility.get("primary") and self.video_visibility.get("primary").isChecked():
                enabled_video_indices.append(0)  # Index 0 = primary video
            for i in range(len(self.project.additional_videos)):
                video_id = f"video_{i}"
                if video_id in self.video_visibility and self.video_visibility[video_id].isChecked():
                    enabled_video_indices.append(i + 1)  # Index i+1 = additional video i

            # ---- Show naming dialog ----
            naming_dialog = ExportNamingDialog(self)
            if naming_dialog.exec_() != QtWidgets.QDialog.Accepted:
                self.statusLabel.setText("Export cancelled")
                return
            
            naming_params = naming_dialog.get_parameters()
            if not naming_params:
                return
            picker_id, condition = naming_params

            planned_paths = preview_export_file_paths(
                self.project,
                Path(output_folder),
                picker_id,
                condition,
                export_gaze_overlay=export_gaze_overlay,
                enabled_video_indices=enabled_video_indices,
            )
            existing_paths = [p for p in planned_paths.values() if p.exists()]

            existing_file_policy = "overwrite"
            if existing_paths:
                unique_existing = sorted({str(p) for p in existing_paths})
                preview_items = unique_existing[:12]
                remaining = max(0, len(unique_existing) - len(preview_items))
                details = "\n".join(f"- {Path(p).name}" for p in preview_items)
                if remaining > 0:
                    details += f"\n- ... and {remaining} more"

                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Existing Export Files Detected",
                    "Some output filenames already exist in the selected folder.\n\n"
                    "Existing files:\n"
                    f"{details}\n\n"
                    "Yes = Override existing files\n"
                    "No = Skip existing files\n"
                    "Cancel = Stop export",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.No,
                )
                if reply == QtWidgets.QMessageBox.Cancel:
                    self.statusLabel.setText("Export cancelled")
                    return
                if reply == QtWidgets.QMessageBox.No:
                    existing_file_policy = "skip"
                else:
                    existing_file_policy = "overwrite"

            # Show progress dialog
            progress = QtWidgets.QProgressDialog(
                "Exporting data...", None, 0, 100, self
            )
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.show()
            
            # Export with naming parameters
            results = export_project(
                self.project,
                Path(output_folder),
                export_gaze_overlay=export_gaze_overlay,
                progress_callback=lambda msg: None,  # Could update progress dialog here
                picker_id=picker_id,
                condition=condition,
                enabled_video_indices=enabled_video_indices,
                existing_file_policy=existing_file_policy,
            )

            metadata_path = None
            metadata_updated = len(results) > 0
            if metadata_updated:
                # Always refresh metadata when at least one output file was created/overwritten.
                metadata_path = export_project_metadata(
                    self.project,
                    Path(output_folder),
                    condition=condition,
                    existing_file_policy="overwrite",
                    enabled_video_indices=enabled_video_indices,
                    export_results=results,
                )
            
            progress.close()
            
            # Show results with detailed breakdown
            msg = "Export complete!\n\n📹 Video Files:\n"
            
            # Primary video
            if 'primary_video' in results:
                msg += f"  ✓ {Path(results['primary_video']).name}\n"
            
            # Gaze overlay
            if 'gaze_overlay_video' in results:
                msg += f"  ✓ {Path(results['gaze_overlay_video']).name}\n"
            elif not export_gaze_overlay:
                msg += "  - Skipped\n"
            
            # Additional videos
            additional_keys = sorted(
                [key for key in results.keys() if key.startswith('additional_video_')],
                key=lambda k: int(k.rsplit('_', 1)[1])
            )
            if additional_keys:
                msg += f"\n📹 Additional Videos ({len(additional_keys)}):\n"
                for key in additional_keys:
                    msg += f"  ✓ {Path(results[key]).name}\n"
            
            # Sensor data
            sensor_keys = [k for k in results.keys() if k not in ['primary_video', 'gaze_overlay_video'] 
                          and not k.startswith('additional_video_')]
            if sensor_keys:
                msg += f"\n📊 Sensor Data ({len(sensor_keys)}):\n"
                for key in sensor_keys:
                    msg += f"  ✓ {Path(results[key]).name}\n"

            msg += "\n📝 Metadata:\n"
            if metadata_path is not None:
                msg += f"  ✓ {Path(metadata_path).name}\n"
            else:
                msg += "  - Skipped (no new files created or overwritten)\n"
            
            QtWidgets.QMessageBox.information(self, "Success", msg)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
    
    def _on_remove_video(self):
        """Remove an additional video."""
        if not self.project or not self.project.additional_videos:
            QtWidgets.QMessageBox.warning(self, "Warning", "No additional videos to remove")
            return
        
        # Create dialog to select which video to remove
        video_names = [self._video_display_name(v) for v in self.project.additional_videos]
        selected, ok = QtWidgets.QInputDialog.getItem(
            self, "Remove Video", "Select video to remove:", video_names
        )
        
        if ok and selected:
            # Get the index
            video_index = video_names.index(selected)
            
            # Remove from project
            removed_video = self.project.additional_videos.pop(video_index)
            
            # Remove from video player and rebuild remaining additional displays to keep IDs in-sync
            if self.video_player:
                # Clean up all non-primary displays first
                for vid in list(self.video_player.video_displays.keys()):
                    if vid == 'primary':
                        continue
                    try:
                        disp = self.video_player.video_displays[vid]
                        disp.stop_and_cleanup()
                        disp.setParent(None)
                    except Exception:
                        pass
                    try:
                        if vid in self.video_player.video_visibility:
                            del self.video_player.video_visibility[vid]
                    except Exception:
                        pass
                    try:
                        del self.video_player.video_displays[vid]
                    except Exception:
                        pass

                # Re-add remaining additional videos so ids are 0-based and contiguous
                self.additional_video_ids = {}
                for i, v in enumerate(self.project.additional_videos):
                    try:
                        new_id = self.video_player.add_additional_video(v, i)
                        self.additional_video_ids[i] = new_id
                    except Exception:
                        pass
                # Relayout video grid
                try:
                    self.video_player._relayout_videos()
                except Exception:
                    pass
            
            # Update UI
            self._update_video_visibility_controls()
            self._update_sync_panel()
            
            self.statusLabel.setText(f"Removed video: {self._video_display_name(removed_video)}")
    
    def _on_clear_all_data(self):
        """Clear all imported data (eye tracking and videos)."""
        reply = QtWidgets.QMessageBox.question(
            self, "Clear All Data",
            "This will remove all loaded data (eye tracking and videos).\n\nContinue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # Clean up video player
            if self.video_player:
                self.video_player.cleanup()
                self.video_player = None
            
            # Clear project
            self.project = None
            self.video_visibility = {}
            self.additional_video_ids = {}
            
            # Clear UI
            for i in reversed(range(self.video_container_layout.count())):
                self.video_container_layout.itemAt(i).widget().setParent(None)
            
            for i in reversed(range(self.panel_container.layout().count())):
                self.panel_container.layout().itemAt(i).widget().setParent(None)
            
            while self.video_visibility_layout.count() > 0:
                self.video_visibility_layout.itemAt(0).widget().setParent(None)
            
            # Disable buttons
            self.export_button.setEnabled(False)
            self.global_play_button.setEnabled(False)
            self.global_pause_button.setEnabled(False)
            self.global_frame_back_button.setEnabled(False)
            self.global_frame_forward_button.setEnabled(False)
            
            self.statusLabel.setText("Ready. Load eye tracking data to begin.")
    
    def _on_about(self):
        """Show about dialog."""
        QtWidgets.QMessageBox.information(
            self, "About",
            "Eye Tracking Data Synchronization Tool\n\n"
            "A modular tool for synchronizing, trimming, and exporting "
            "eye tracking video and sensor data."
        )
    
    def closeEvent(self, event):
        """Clean up before closing."""
        if self.video_player:
            self.video_player.cleanup()
        event.accept()


def main():
    """Main entry point."""
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
