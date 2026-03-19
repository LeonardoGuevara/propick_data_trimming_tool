"""
Utility functions for the GUI.
"""
from typing import Optional, Tuple
import numpy as np
from PyQt5 import QtWidgets


def format_time(total_seconds: float) -> str:
    """Format time as MM:SS.mmm"""
    ms = int((total_seconds - int(total_seconds)) * 1000)
    m = int(total_seconds // 60)
    s = int(total_seconds % 60)
    return f"{m:02d}:{s:02d}.{ms:03d}"


def format_time_extended(total_seconds: float) -> str:
    """Format time as HH:MM:SS.mmm"""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    millis = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def format_fps(fps: float, decimals: int = 9) -> str:
    """Format FPS with high precision while trimming insignificant trailing zeros."""
    try:
        val = float(fps)
    except Exception:
        return "0"
    if abs(val) < 1e-12:
        return "0"
    text = f"{val:.{max(0, int(decimals))}f}".rstrip('0').rstrip('.')
    return text if text else "0"


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Convert seconds to frame count."""
    return int(seconds * fps)


def frames_to_seconds(frames: int, fps: float) -> float:
    """Convert frame count to seconds."""
    return frames / fps if fps > 0 else 0.0


def get_gaze_columns(df) -> tuple:
    """
    Find gaze x, y columns in a DataFrame.
    Returns: (gaze_x_col, gaze_y_col) or (None, None) if not found.
    """
    gaze_x_col = None
    gaze_y_col = None
    
    for col in df.columns:
        lc = col.lower()
        if 'gaze x' in lc or 'gaze_x' in lc or (lc == 'x'):
            gaze_x_col = col
        if 'gaze y' in lc or 'gaze_y' in lc or (lc == 'y'):
            gaze_y_col = col
    
    return gaze_x_col, gaze_y_col


def find_closest_timestamp_index(timestamp_values: np.ndarray, target_ts: int, 
                                  tolerance_ns: int = int(1e8)) -> Optional[int]:
    """
    Find index of closest timestamp within tolerance.
    
    Args:
        timestamp_values: Array of nanosecond timestamps
        target_ts: Target timestamp to find
        tolerance_ns: Maximum allowed difference in nanoseconds
    
    Returns:
        Index of closest match, or None if nothing within tolerance
    """
    if timestamp_values is None or len(timestamp_values) == 0:
        return None
    
    diffs = np.abs(timestamp_values - target_ts)
    min_diff = np.min(diffs)
    
    if min_diff > tolerance_ns:
        return None
    
    return int(np.argmin(diffs))


def scale_coordinates(coords: tuple, from_size: tuple, to_size: tuple) -> tuple:
    """
    Scale coordinates from one size to another (useful for GUI scaling).
    
    Args:
        coords: (x, y) coordinates
        from_size: (width, height) of original size
        to_size: (width, height) of new size
    
    Returns:
        Scaled (x, y) coordinates
    """
    if from_size[0] == 0 or from_size[1] == 0:
        return coords
    
    scale_x = to_size[0] / from_size[0]
    scale_y = to_size[1] / from_size[1]
    
    return (int(coords[0] * scale_x), int(coords[1] * scale_y))


def video_display_name(video) -> str:
    """Return a consistent camera-based label for UI display."""
    camera = (getattr(video, 'camera_type', None) or ("eyes" if getattr(video, 'is_primary', False) else "unknown")).strip()
    return f"{camera} ({video.name})"


class CameraSelectionDialog(QtWidgets.QDialog):
    """Dialog for selecting camera type for a video."""

    CAMERA_OPTIONS = ["chest", "trolley_picker", "trolley_crop", "custom"]

    def __init__(self, video_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Camera Type")
        self.setMinimumWidth(400)
        self.video_name = video_name
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        info_label = QtWidgets.QLabel(
            f"Select camera type for: {self.video_name}\n"
            f"(This parameter will be used in the exported filename)"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addSpacing(10)

        camera_layout = QtWidgets.QVBoxLayout()
        camera_layout.addWidget(QtWidgets.QLabel("Camera:"))

        self.camera_group = QtWidgets.QButtonGroup(self)
        for index, option in enumerate(self.CAMERA_OPTIONS):
            radio = QtWidgets.QRadioButton(option)
            self.camera_group.addButton(radio, index)
            camera_layout.addWidget(radio)
            if index == 0:
                radio.setChecked(True)

        layout.addLayout(camera_layout)

        custom_layout = QtWidgets.QHBoxLayout()
        custom_layout.addWidget(QtWidgets.QLabel("Custom value (if selected):"))
        self.custom_input = QtWidgets.QLineEdit()
        self.custom_input.setPlaceholderText("e.g., side_view, overhead, etc.")
        self.custom_input.setEnabled(False)
        custom_layout.addWidget(self.custom_input)
        layout.addLayout(custom_layout)

        def on_camera_changed():
            self.custom_input.setEnabled(self.camera_group.checkedId() == 3)

        for radio in self.camera_group.buttons():
            radio.toggled.connect(on_camera_changed)

        layout.addSpacing(10)

        btn_layout = QtWidgets.QHBoxLayout()
        ok_button = QtWidgets.QPushButton("OK")
        cancel_button = QtWidgets.QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_button)
        btn_layout.addWidget(cancel_button)
        layout.addLayout(btn_layout)

    def get_camera_type(self) -> Optional[str]:
        if self.camera_group.checkedId() == 3:
            custom_value = self.custom_input.text().strip()
            if not custom_value:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Validation Error",
                    "Please enter a custom camera value.",
                )
                return None
            return custom_value

        index = self.camera_group.checkedId()
        if 0 <= index < len(self.CAMERA_OPTIONS):
            return self.CAMERA_OPTIONS[index]
        return None


class ExportNamingDialog(QtWidgets.QDialog):
    """Dialog for entering custom file naming parameters."""

    CONDITION_OPTIONS = [
        "left_direction",
        "right_direction",
        "no_intervention",
        "generic_intervention",
        "personalised_intervention",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export File Naming")
        self.setMinimumWidth(400)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        info_label = QtWidgets.QLabel(
            "Provide naming parameters for exported files "
            "(name format: <pickerID>_<camera>_<condition>.mp4)"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addSpacing(10)

        picker_layout = QtWidgets.QHBoxLayout()
        picker_layout.addWidget(QtWidgets.QLabel("Picker ID:"))
        self.picker_id_input = QtWidgets.QLineEdit()
        self.picker_id_input.setPlaceholderText("e.g., P001, Picker_01, etc.")
        picker_layout.addWidget(self.picker_id_input)
        layout.addLayout(picker_layout)

        condition_layout = QtWidgets.QHBoxLayout()
        condition_layout.addWidget(QtWidgets.QLabel("Condition:"))
        self.condition_combo = QtWidgets.QComboBox()
        self.condition_combo.addItems(self.CONDITION_OPTIONS)
        self.condition_combo.setEditable(False)
        condition_layout.addWidget(self.condition_combo)
        layout.addLayout(condition_layout)

        custom_condition_layout = QtWidgets.QHBoxLayout()
        custom_condition_layout.addWidget(QtWidgets.QLabel("Custom condition (optional):"))
        self.custom_condition_input = QtWidgets.QLineEdit()
        self.custom_condition_input.setPlaceholderText("Type custom condition here")
        custom_condition_layout.addWidget(self.custom_condition_input)
        layout.addLayout(custom_condition_layout)

        layout.addSpacing(10)

        btn_layout = QtWidgets.QHBoxLayout()
        ok_button = QtWidgets.QPushButton("Export")
        cancel_button = QtWidgets.QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_button)
        btn_layout.addWidget(cancel_button)
        layout.addLayout(btn_layout)

    def get_parameters(self) -> Optional[Tuple[str, str]]:
        picker_id = self.picker_id_input.text().strip()
        custom_condition = self.custom_condition_input.text().strip()
        condition = custom_condition if custom_condition else self.condition_combo.currentText().strip()

        if not picker_id:
            QtWidgets.QMessageBox.warning(self, "Validation Error", "Picker ID cannot be empty.")
            return None

        if not condition:
            QtWidgets.QMessageBox.warning(self, "Validation Error", "Condition must be selected.")
            return None

        return picker_id, condition
