"""
Synchronization and trimming control panel widget.
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Optional, Callable
from models import ProjectData, VideoData
from gui_utils import format_time, frames_to_seconds


class SyncTrimPanel(QtWidgets.QGroupBox):
    """Panel for controlling synchronization and trimming of videos."""
    
    trim_changed = QtCore.pyqtSignal()  # Emitted when trim frame changes
    
    def __init__(self, project: ProjectData, parent=None, video_player=None):
        super().__init__("Synchronization & Trimming", parent)
        self.project = project
        self.current_frame = 0
        self.additional_video_widgets = {}  # Initialize early to avoid AttributeError
        self.video_player = video_player  # Reference to video player for getting individual frame counts
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the panel UI with vertical scroll for many videos."""
        # Create a scroll area for the whole panel
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        # Main widget inside scroll area
        content_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content_widget)
        
        if not self.project.primary_video:
            layout.addWidget(QtWidgets.QLabel("No primary video loaded"))
            return
        
        # Primary video trim section
        primary_group = QtWidgets.QGroupBox("Primary Video (Eye Tracking)", self)
        primary_layout = QtWidgets.QVBoxLayout(primary_group)
        
        # Start frame controls
        start_frame_layout = QtWidgets.QHBoxLayout()
        start_frame_layout.addWidget(QtWidgets.QLabel("Trim Start:"))
        self.start_frame_label = QtWidgets.QLabel(f"0 (0.000s)")
        start_frame_layout.addWidget(self.start_frame_label)
        self.set_start_button = QtWidgets.QPushButton("Set to Current Frame")
        self.set_start_button.clicked.connect(self._on_set_start)
        start_frame_layout.addWidget(self.set_start_button)
        start_frame_layout.addStretch()
        primary_layout.addLayout(start_frame_layout)
        
        # End frame controls
        end_frame_layout = QtWidgets.QHBoxLayout()
        end_frame_layout.addWidget(QtWidgets.QLabel("Trim End:"))
        self.end_frame_label = QtWidgets.QLabel(
            f"{self.project.primary_video.frame_count - 1} ({self.project.primary_video.duration:.3f}s)"
        )
        end_frame_layout.addWidget(self.end_frame_label)
        self.set_end_button = QtWidgets.QPushButton("Set to Current Frame")
        self.set_end_button.clicked.connect(self._on_set_end)
        end_frame_layout.addWidget(self.set_end_button)
        end_frame_layout.addStretch()
        primary_layout.addLayout(end_frame_layout)
        
        # Trim duration info
        self.trim_duration_label = QtWidgets.QLabel()
        self.trim_duration_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        primary_layout.addWidget(self.trim_duration_label)
        
        # Reset button
        self.reset_primary_button = QtWidgets.QPushButton("Reset to Full Video")
        self.reset_primary_button.clicked.connect(self._on_reset_primary)
        primary_layout.addWidget(self.reset_primary_button)
        
        layout.addWidget(primary_group)
        
        # Additional videos trimming only (no synchronization controls)
        if self.project.additional_videos:
            additional_group = QtWidgets.QGroupBox("Additional Videos", self)
            additional_layout = QtWidgets.QVBoxLayout(additional_group)
            
            self.additional_video_widgets = {}
            for i, video in enumerate(self.project.additional_videos):
                # Create expandable section for each video
                video_section = QtWidgets.QGroupBox(f"Video {i+2}: {video.name}", self)
                video_section_layout = QtWidgets.QVBoxLayout(video_section)
                
                # Trim start controls
                start_frame_layout = QtWidgets.QHBoxLayout()
                start_frame_layout.addWidget(QtWidgets.QLabel("Trim Start:"))
                start_frame_label = QtWidgets.QLabel(f"0 (0.000s)")
                start_frame_layout.addWidget(start_frame_label)
                start_button = QtWidgets.QPushButton("Set to Current Frame")
                start_button.clicked.connect(lambda checked, idx=i: self._on_set_video_start(idx))
                start_frame_layout.addWidget(start_button)
                start_frame_layout.addStretch()
                video_section_layout.addLayout(start_frame_layout)
                
                # Trim end controls
                end_frame_layout = QtWidgets.QHBoxLayout()
                end_frame_layout.addWidget(QtWidgets.QLabel("Trim End:"))
                end_frame_label = QtWidgets.QLabel(f"{video.frame_count - 1} ({video.duration:.3f}s)")
                end_frame_layout.addWidget(end_frame_label)
                end_button = QtWidgets.QPushButton("Set to Current Frame")
                end_button.clicked.connect(lambda checked, idx=i: self._on_set_video_end(idx))
                end_frame_layout.addWidget(end_button)
                end_frame_layout.addStretch()
                video_section_layout.addLayout(end_frame_layout)
                
                # Trim duration info
                trim_duration_label = QtWidgets.QLabel()
                trim_duration_label.setStyleSheet("color: #0066cc; font-weight: bold;")
                video_section_layout.addWidget(trim_duration_label)
                
                # Reset button
                reset_button = QtWidgets.QPushButton("Reset to Full Video")
                reset_button.clicked.connect(lambda checked, idx=i: self._on_reset_video(idx))
                video_section_layout.addWidget(reset_button)
                
                additional_layout.addWidget(video_section)
                self.additional_video_widgets[i] = {
                    'start_frame_label': start_frame_label,
                    'end_frame_label': end_frame_label,
                    'trim_duration_label': trim_duration_label,
                    'video': video
                }
            
            layout.addWidget(additional_group)
        
        layout.addStretch()
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(scroll_area)
        scroll_area.setWidget(content_widget)
        self._update_display()
    
    def update_current_frame(self, frame_num: int) -> None:
        """Update the current frame (called from main window)."""
        self.current_frame = frame_num
    
    def _on_set_start(self) -> None:
        """Set trim start to current frame."""
        primary = self.project.primary_video
        
        # Validate: start_frame must be less than end_frame and within video range
        if self.current_frame >= primary.end_frame:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Frame",
                f"Start frame ({self.current_frame}) must be less than end frame ({primary.end_frame})"
            )
            return
        
        if self.current_frame < 0 or self.current_frame >= primary.frame_count:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Frame",
                f"Frame {self.current_frame} is outside video range [0, {primary.frame_count - 1}]"
            )
            return
        
        primary.start_frame = self.current_frame
        self._update_display()
        self.trim_changed.emit()
    
    def _on_set_end(self) -> None:
        """Set trim end to current frame."""
        primary = self.project.primary_video
        
        # Validate: end_frame must be greater than start_frame and within video range
        if self.current_frame <= primary.start_frame:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Frame",
                f"End frame ({self.current_frame}) must be greater than start frame ({primary.start_frame})"
            )
            return
        
        if self.current_frame < 0 or self.current_frame >= primary.frame_count:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Frame",
                f"Frame {self.current_frame} is outside video range [0, {primary.frame_count - 1}]"
            )
            return
        
        primary.end_frame = self.current_frame
        self._update_display()
        self.trim_changed.emit()
    
    def _on_reset_primary(self) -> None:
        """Reset primary video to full range."""
        primary = self.project.primary_video
        primary.start_frame = 0
        primary.end_frame = primary.frame_count - 1
        self._update_display()
        self.trim_changed.emit()
    
    def _on_set_video_start(self, video_index: int) -> None:
        """Set trim start for additional video to its current frame."""
        if video_index < len(self.project.additional_videos):
            video = self.project.additional_videos[video_index]
            # Find the matching SingleVideoDisplay for this video and get its current frame
            actual_frame = None
            if self.video_player:
                vid_key = f"video_{video_index}"
                if vid_key in self.video_player.video_displays:
                    actual_frame = self.video_player.video_displays[vid_key].current_frame
                else:
                    for disp in self.video_player.video_displays.values():
                        try:
                            if hasattr(disp, 'video') and hasattr(video, 'file_path'):
                                if str(disp.video.file_path) == str(video.file_path) or getattr(disp.video, 'name', None) == getattr(video, 'name', None):
                                    actual_frame = disp.current_frame
                                    break
                        except Exception:
                            continue

            if actual_frame is not None:
                # Validate: start_frame must be less than end_frame and within video range
                end_frame = video.end_frame if video.end_frame is not None else video.frame_count - 1
                if actual_frame >= end_frame:
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Frame",
                        f"Start frame ({actual_frame}) must be less than end frame ({end_frame})"
                    )
                    return
                
                if actual_frame < 0 or actual_frame >= video.frame_count:
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Frame",
                        f"Frame {actual_frame} is outside video range [0, {video.frame_count - 1}]"
                    )
                    return
                
                video.start_frame = int(actual_frame)
                self._update_display()
                self.trim_changed.emit()
    
    def _on_set_video_end(self, video_index: int) -> None:
        """Set trim end for additional video to its current frame."""
        if video_index < len(self.project.additional_videos):
            video = self.project.additional_videos[video_index]
            actual_frame = None
            if self.video_player:
                vid_key = f"video_{video_index}"
                if vid_key in self.video_player.video_displays:
                    actual_frame = self.video_player.video_displays[vid_key].current_frame
                else:
                    for disp in self.video_player.video_displays.values():
                        try:
                            if hasattr(disp, 'video') and hasattr(video, 'file_path'):
                                if str(disp.video.file_path) == str(video.file_path) or getattr(disp.video, 'name', None) == getattr(video, 'name', None):
                                    actual_frame = disp.current_frame
                                    break
                        except Exception:
                            continue

            if actual_frame is not None:
                # Validate: end_frame must be greater than start_frame and within video range
                if actual_frame <= video.start_frame:
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Frame",
                        f"End frame ({actual_frame}) must be greater than start frame ({video.start_frame})"
                    )
                    return
                
                if actual_frame < 0 or actual_frame >= video.frame_count:
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Frame",
                        f"Frame {actual_frame} is outside video range [0, {video.frame_count - 1}]"
                    )
                    return
                
                video.end_frame = int(actual_frame)
                self._update_display()
                self.trim_changed.emit()
    
    def _on_reset_video(self, video_index: int) -> None:
        """Reset additional video to full range."""
        if video_index < len(self.project.additional_videos):
            video = self.project.additional_videos[video_index]
            video.start_frame = 0
            video.end_frame = video.frame_count - 1
            self._update_display()
            self.trim_changed.emit()
    
    def _on_offset_changed(self, video: VideoData, offset: int) -> None:
        """Handle offset change for additional video."""
        video.offset_frames = offset
    
    def _on_auto_sync(self, video: VideoData) -> None:
        """Attempt automatic synchronization based on timestamps (REMOVED - trim only)."""
        pass
    
    def _update_display(self) -> None:
        """Update all display labels."""
        if not self.project.primary_video:
            return
        
        primary = self.project.primary_video
        
        start_time = frames_to_seconds(primary.start_frame, primary.fps)
        end_time = frames_to_seconds(primary.end_frame, primary.fps)
        duration = end_time - start_time
        
        self.start_frame_label.setText(f"{primary.start_frame} ({start_time:.3f}s)")
        self.end_frame_label.setText(f"{primary.end_frame} ({end_time:.3f}s)")
        trimmed_frame_count = primary.end_frame - primary.start_frame + 1
        self.trim_duration_label.setText(
            f"Trimmed Duration: {trimmed_frame_count} frames ({duration:.3f}s)"
        )
        
        # Update additional video labels
        for i, widget_dict in self.additional_video_widgets.items():
            if i < len(self.project.additional_videos):
                video = self.project.additional_videos[i]
                
                start_time = frames_to_seconds(video.start_frame, video.fps)
                end_time = frames_to_seconds(video.end_frame, video.fps)
                duration = end_time - start_time
                
                widget_dict['start_frame_label'].setText(f"{video.start_frame} ({start_time:.3f}s)")
                widget_dict['end_frame_label'].setText(f"{video.end_frame} ({end_time:.3f}s)")
                trimmed_frame_count = video.end_frame - video.start_frame + 1
                widget_dict['trim_duration_label'].setText(
                    f"Trimmed Duration: {trimmed_frame_count} frames ({duration:.3f}s)"
                )
