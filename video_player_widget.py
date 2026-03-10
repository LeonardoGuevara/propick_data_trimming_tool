"""
Video player widget for displaying videos.
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
from typing import Optional, Callable
from models import VideoData, SensorData
from gui_utils import format_time


class VideoPlayerWidget(QtWidgets.QWidget):
    """Widget for playing video."""
    
    frame_changed = QtCore.pyqtSignal(int)  # Emits current frame number
    
    def __init__(self, video: VideoData, parent=None):
        super().__init__(parent)
        self.video = video
        
        self.cap = cv2.VideoCapture(str(video.file_path))
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video.file_path}")
        
        self.current_frame = 0
        self.playing = False
        self.speed = 1.0
        
        # Frame caching for performance optimization
        self._cached_frame_num = -1
        self._cached_pixmap = None
        self._last_read_frame = -1  # Track last frame we actually read
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
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display
        self.video_label = QtWidgets.QLabel()
        self.video_label.setStyleSheet("background: black")
        self.video_label.setMinimumSize(320, 240)
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
        
        # Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.video.frame_count - 1))
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(lambda: self._on_slider_moved(self.slider.value()))
        layout.addWidget(self.slider)
        
        self.setLayout(layout)
        # Defer initial frame load to let the UI stay responsive for large videos
        try:
            QtCore.QTimer.singleShot(0, self._update_display)
        except Exception:
            self._update_display()
    
    def toggle_play(self) -> None:
        """Toggle playback."""
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.playing = True
            self.play_button.setText("Pause")
            self._restart_timer()
    
    def _restart_timer(self) -> None:
        """Restart timer with current speed."""
        # Use shorter interval to check more frequently
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
        """Handle timer tick for playback."""
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
    
    def _update_display(self) -> None:
        """Update video frame display with optimized sequential reading."""
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
        scaled_pixmap = self._cached_pixmap.scaledToWidth(640, QtCore.Qt.FastTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update UI less frequently during playback for better performance
        if not self.playing or self.current_frame % 3 == 0:
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame)
            self.slider.blockSignals(False)
            
            current_time = self.current_frame / self.video.fps
            total_time = self.video.duration
            self.time_label.setText(f"{format_time(current_time)} / {format_time(total_time)}")
        
        self.frame_changed.emit(self.current_frame)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.playing:
            self.toggle_play()
        if self.cap:
            self.cap.release()
        # Clear caches to free memory
        self._cached_pixmap = None
        self._cached_frame_num = -1
        self._frame_buffer = None
        self._last_read_frame = -1
