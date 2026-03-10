"""
Enhanced multi-video player widget for displaying multiple videos simultaneously.
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
from typing import Optional, Dict, List
from models import VideoData, SensorData
from gui_utils import format_time, get_gaze_columns, find_closest_timestamp_index


class SingleVideoDisplay(QtWidgets.QWidget):
    """Widget for a single video display with controls."""
    
    frame_changed = QtCore.pyqtSignal(int)  # Emits current frame number
    move_to_beginning_clicked = QtCore.pyqtSignal()
    
    def __init__(self, video: VideoData, video_name: str, is_primary: bool = False,
                 gaze_data: Optional[SensorData] = None, parent=None):
        super().__init__(parent)
        self.video = video
        self.video_name = video_name
        self.is_primary = is_primary
        
        self.cap = cv2.VideoCapture(str(video.file_path))
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video.file_path}")
        
        self.current_frame = 0
        self.playing = False
        self.speed = 1.0
        
        # Frame caching for performance
        self._cached_frame_num = -1
        self._cached_pixmap = None
        self._last_read_frame = -1  # Track last frame we actually read (not just displayed)
        self._frame_buffer = None   # Store last decoded frame for reuse
        
        # Smart playback timing
        self._last_frame_time = 0
        self._target_frame_time = 1.0 / self.video.fps if self.video.fps > 0 else 0.033
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._on_timer)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the video display widget."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Video name label
        name_label = QtWidgets.QLabel(f"<b>{self.video_name}</b>")
        layout.addWidget(name_label)
        
        # Video display
        self.video_label = QtWidgets.QLabel()
        self.video_label.setStyleSheet("background: black; border: 1px solid #555;")
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        # Control panel
        control_layout = QtWidgets.QHBoxLayout()
        
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_button)
        
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        control_layout.addWidget(self.speed_combo)
        
        self.time_label = QtWidgets.QLabel("00:00.000 / 00:00.000")
        control_layout.addWidget(self.time_label)
        
        layout.addLayout(control_layout)
        
        # Additional controls panel
        button_layout = QtWidgets.QHBoxLayout()
        
        # Frame backward button
        self.frame_backward_button = QtWidgets.QPushButton("◄◄ Frame Back")
        self.frame_backward_button.clicked.connect(self._on_frame_backward)
        button_layout.addWidget(self.frame_backward_button)
        
        # Frame forward button
        self.frame_forward_button = QtWidgets.QPushButton("Frame Fwd ►►")
        self.frame_forward_button.clicked.connect(self._on_frame_forward)
        button_layout.addWidget(self.frame_forward_button)
        
        self.move_to_beginning_button = QtWidgets.QPushButton("Move to Beginning")
        self.move_to_beginning_button.clicked.connect(self._on_move_to_beginning)
        button_layout.addWidget(self.move_to_beginning_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.video.frame_count - 1))
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(lambda: self._on_slider_moved(self.slider.value()))
        layout.addWidget(self.slider)
        
        # Frame info
        self.frame_info = QtWidgets.QLabel(f"Frame: 0 / {self.video.frame_count} | FPS: {self.video.fps:.1f}")
        layout.addWidget(self.frame_info)
        
        self.setLayout(layout)
        self._update_display()
    
    def toggle_play(self) -> None:
        """Toggle playback - COMPLETELY INDEPENDENT per video."""
        # This method must remain isolated from other videos' playback state
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.playing = True
            self.play_button.setText("Pause")
            self._restart_timer()
        
        # Ensure no cross-talk with other videos
        # (Each video has its own timer instance)
    
    def _restart_timer(self) -> None:
        """Restart timer with current speed."""
        # Use shorter interval to check more frequently - we'll skip frames if needed
        interval = max(8, int(1000 / (self.video.fps * self.speed)))
        self.timer.start(interval)
        # Reset timing
        import time
        self._last_frame_time = time.perf_counter()
        self._target_frame_time = 1.0 / (self.video.fps * self.speed) if self.video.fps > 0 else 0.033
    
    def _on_speed_changed(self, text: str) -> None:
        """Handle speed change."""
        self.speed = float(text.replace('x', ''))
        if self.playing:
            self._restart_timer()
    
    def _on_timer(self) -> None:
        """Handle timer tick for playback - always advance by 1 for smooth playback."""
        self.current_frame += 1
        
        if self.current_frame >= self.video.frame_count:
            self.current_frame = self.video.frame_count - 1
            self.timer.stop()
            self.playing = False
            self.play_button.setText("Play")
        
        self._update_display()
    
    def _on_slider_moved(self, value: int) -> None:
        """Handle slider movement."""
        self.current_frame = int(value)
        self._update_display()
    
    def set_frame(self, frame_num: int) -> None:
        """Jump to specific frame."""
        self.current_frame = max(0, min(frame_num, self.video.frame_count - 1))
        self._update_display()
    
    def _on_frame_backward(self) -> None:
        """Move backward by one frame."""
        if self.playing:
            self.toggle_play()  # Stop playback when using frame navigation
        self.set_frame(self.current_frame - 1)
    
    def _on_frame_forward(self) -> None:
        """Move forward by one frame."""
        if self.playing:
            self.toggle_play()  # Stop playback when using frame navigation
        self.set_frame(self.current_frame + 1)
    
    def _on_move_to_beginning(self) -> None:
        """Request that the trimmed start be set to the Sync/Trim panel current frame.

        Emit a signal so the top-level window can decide which frame to use (the value
        selected in the right-hand panel) and then update the display and model.
        """
        try:
            self.move_to_beginning_clicked.emit()
        except Exception:
            # Fallback to previous behaviour: use stored start_frame
            start_frame = getattr(self.video, 'start_frame', 0)
            start_frame = max(0, min(start_frame, self.video.frame_count - 1))
            self.set_frame(start_frame)
    
    def synchronize_to_frame(self, primary_frame: int, primary_video: VideoData) -> None:
        """Synchronize this video to a frame in the primary video."""
        if not hasattr(self.video, 'offset_frames'):
            return
        
        # Calculate offset based on offset_frames
        mapped_frame = primary_frame + int(self.video.offset_frames)
        mapped_frame = max(0, min(mapped_frame, self.video.frame_count - 1))
        
        self.set_frame(mapped_frame)
    
    def _update_display(self) -> None:
        """Update video frame display with optimized sequential reading."""
        # Ensure cap is valid before reading frames
        if self.cap is None:
            self.cap = cv2.VideoCapture(str(self.video.file_path))
            self._last_read_frame = -1
            self._frame_buffer = None
            self._cached_pixmap = None
            self._cached_frame_num = -1
        
        # Critical optimization: Avoid expensive seeking during sequential playback
        # cv2.VideoCapture.set() is VERY slow - we only do it when needed
        
        frame_to_process = None
        
        # Check if we're doing sequential playback (next frame in sequence)
        if self.current_frame == self._last_read_frame + 1:
            # Next frame in sequence - just read without seeking (FAST)
            ret, frame = self.cap.read()
            if ret:
                frame_to_process = frame
                self._last_read_frame = self.current_frame
        elif self.current_frame > self._last_read_frame + 1 and self.current_frame - self._last_read_frame < 10:
            # Small skip ahead - read multiple frames sequentially (faster than seeking)
            frames_to_skip = self.current_frame - self._last_read_frame - 1
            for _ in range(frames_to_skip):
                self.cap.read()  # Skip frames
            ret, frame = self.cap.read()
            if ret:
                frame_to_process = frame
                self._last_read_frame = self.current_frame
        elif self.current_frame != self._last_read_frame:
            # Frame jumped (user seeked or replaying) - must seek first (SLOW but necessary)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if ret:
                frame_to_process = frame
                self._last_read_frame = self.current_frame
        else:
            # Same frame requested again - use buffered frame
            if self._frame_buffer is not None:
                frame_to_process = self._frame_buffer
        
        if frame_to_process is None:
            return
        
        # Buffer the frame for potential reuse
        self._frame_buffer = frame_to_process
        
        # Only convert to display if frame changed or not in display cache
        if not hasattr(self, '_cached_frame_num') or self._cached_frame_num != self.current_frame:
            # Work directly on frame - no expensive copy() operation
            # Convert BGR to RGB for Qt
            frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            
            bytes_per_line = ch * w
            qimg = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self._cached_pixmap = QtGui.QPixmap.fromImage(qimg)
            self._cached_frame_num = self.current_frame
        
        # Use cached pixmap - scale with faster algorithm
        target_size = self.video_label.size()
        if target_size.width() > 0 and target_size.height() > 0:
            scaled_pixmap = self._cached_pixmap.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        else:
            scaled_pixmap = self._cached_pixmap.scaledToWidth(400, QtCore.Qt.FastTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setScaledContents(False)
        
        # Update UI less frequently during playback for better performance
        if not self.playing or self.current_frame % 3 == 0:
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame)
            self.slider.blockSignals(False)
            
            current_time = self.current_frame / self.video.fps
            total_time = self.video.duration
            self.time_label.setText(f"{format_time(current_time)} / {format_time(total_time)}")
            self.frame_info.setText(f"Frame: {self.current_frame} / {self.video.frame_count} | FPS: {self.video.fps:.1f}")
        
        self.frame_changed.emit(self.current_frame)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.playing:
            self.toggle_play()
        if self.cap:
            self.cap.release()
    
    def stop_and_cleanup(self) -> None:
        """Stop playback and release resources immediately."""
        if self.timer:
            self.timer.stop()
        if self.playing:
            self.playing = False
            self.play_button.setText("Play")
        if self.cap:
            self.cap.release()
            self.cap = None
        # Clear caches to free memory
        self._cached_pixmap = None
        self._cached_frame_num = -1
        self._frame_buffer = None
        self._last_read_frame = -1


class MultiVideoPlayerWidget(QtWidgets.QWidget):
    """Widget for displaying multiple videos side by side with INDEPENDENT controls."""
    
    frame_changed = QtCore.pyqtSignal(int)  # Emits current frame of primary video
    move_to_beginning_requested = QtCore.pyqtSignal(str)
    
    def __init__(self, primary_video: VideoData, parent=None):
        super().__init__(parent)
        self.primary_video = primary_video
        self.video_displays: Dict[str, SingleVideoDisplay] = {}
        self.video_visibility: Dict[str, bool] = {}
        
        # Synchronization state: only sync non-primary videos when they're playing
        self.sync_mode = "independent"  # Can be "independent" or "synchronized"
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize multi-video display."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Create a container with a grid layout so up to 4 videos fit the available space
        container = QtWidgets.QWidget()
        self.video_grid = QtWidgets.QGridLayout(container)
        self.video_grid.setSpacing(10)
        container.setLayout(self.video_grid)

        # Put container inside a scroll area only if content exceeds space
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)

        # Add primary video
        self._add_video("primary", self.primary_video, "Primary Video (Eye Tracking)", is_primary=True)

        self.setLayout(main_layout)
    
    def _add_video(self, video_id: str, video: VideoData, display_name: str, is_primary: bool = False):
        """Add a video display to the grid - each video is completely independent."""
        if video_id in self.video_displays:
            return  # Already added
        
        display = SingleVideoDisplay(
            video, display_name, is_primary=is_primary,
            gaze_data=None,
            parent=self
        )
        
        self.video_displays[video_id] = display
        self.video_visibility[video_id] = True
        
        # Connect frame change signal only for primary video to notify the UI
        # Other videos are completely independent
        if is_primary:
            display.frame_changed.connect(self._on_primary_frame_changed)
            # connect display's move-to-beginning to be relayed with id
            display.move_to_beginning_clicked.connect(lambda vid=video_id: self.move_to_beginning_requested.emit(vid))
        else:
            # connect non-primary displays as well
            display.move_to_beginning_clicked.connect(lambda vid=video_id: self.move_to_beginning_requested.emit(vid))
        
        # Place display into a responsive 2xN grid (primary at index 0)
        # Use a deferred relayout to avoid heavy immediate layout work
        try:
            QtCore.QTimer.singleShot(0, lambda: self._relayout_videos())
        except Exception:
            # Fallback to immediate add if timer fails
            keys = list(self.video_displays.keys())
            pos_index = keys.index(video_id)
            row = pos_index // 2
            col = pos_index % 2
            self.video_grid.addWidget(display, row, col)
    
    def add_additional_video(self, video: VideoData, video_index: int) -> str:
        """Add an additional video (video_index is 0-based)."""
        video_id = f"video_{video_index}"
        self._add_video(video_id, video, f"Video {video_index + 2} ({video.name})", is_primary=False)
        return video_id
    
    def _on_primary_frame_changed(self, frame_num: int):
        """Handle primary video frame change - emit signal but DON'T force sync to other videos."""
        # Only emit frame change for UI updates (e.g., status bar)
        # Other videos are completely independent and NOT synced
        self.frame_changed.emit(frame_num)
    
    def set_video_visibility(self, video_id: str, visible: bool) -> None:
        """Show or hide a video."""
        if video_id not in self.video_displays:
            return
        display = self.video_displays[video_id]

        if not visible:
            # Stop playback and release resources when hiding
            display.stop_and_cleanup()

        display.setVisible(visible)
        # Update stored visibility and relayout
        self.video_visibility[video_id] = visible
        try:
            self._relayout_videos()
        except Exception:
            pass
    
    def _relayout_videos(self) -> None:
        """Relayout video widgets in the grid according to insertion order and set equal sizes."""
        # Remove all widgets from the grid
        try:
            for i in reversed(range(self.video_grid.count())):
                item = self.video_grid.itemAt(i)
                if item:
                    w = item.widget()
                    if w:
                        self.video_grid.removeWidget(w)
        except Exception:
            pass

        # Build list of visible videos: primary first, then enabled additional videos
        keys = []
        if "primary" in self.video_displays and self.video_visibility.get("primary", True):
            keys.append("primary")
        for vid in list(self.video_displays.keys()):
            if vid == "primary":
                continue
            if self.video_visibility.get(vid, True):
                keys.append(vid)

        # Determine target size for all videos
        target_width = 480
        target_height = 270
        if len(keys) > 1:
            # Use the minimum size among all videos, or a fixed size
            target_width = min([self.video_displays[k].video_label.width() for k in keys] + [480])
            target_height = min([self.video_displays[k].video_label.height() for k in keys] + [270])
        # Set all video labels to the same size
        for k in keys:
            self.video_displays[k].video_label.setFixedSize(target_width, target_height)

        # Re-add visible widgets in grid (2 columns)
        for idx, video_id in enumerate(keys):
            disp = self.video_displays[video_id]
            row = idx // 2
            col = idx % 2
            self.video_grid.addWidget(disp, row, col)
    
    def get_primary_frame(self) -> int:
        """Get current frame of primary video."""
        if "primary" in self.video_displays:
            return self.video_displays["primary"].current_frame
        return 0

    def get_visible_video_ids(self) -> list:
        """Return ordered list of visible video ids (primary first if visible)."""
        ids = []
        if "primary" in self.video_displays and self.video_visibility.get("primary", True):
            ids.append("primary")
        for vid in list(self.video_displays.keys()):
            if vid == "primary":
                continue
            if self.video_visibility.get(vid, True):
                ids.append(vid)
        return ids
    
    def play_all(self) -> None:
        """Play all videos simultaneously."""
        for display in self.video_displays.values():
            if not display.playing and display.isVisible():
                display.toggle_play()
    
    def pause_all(self) -> None:
        """Pause all videos."""
        for display in self.video_displays.values():
            if display.playing:
                display.toggle_play()
    
    def is_any_playing(self) -> bool:
        """Check if any video is currently playing."""
        return any(display.playing for display in self.video_displays.values())
    
    def cleanup(self) -> None:
        """Clean up all video resources."""
        for display in self.video_displays.values():
            display.cleanup()
