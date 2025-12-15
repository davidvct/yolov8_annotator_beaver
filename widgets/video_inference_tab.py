"""
Video inference tab widget.
"""
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                                QLabel, QSlider, QCheckBox, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt

from widgets.video_list import VideoListWidget
from widgets.video_player import VideoPlayerWidget
from utils.video_handler import VideoHandler
from utils.yolo_inference import YOLOInference
from utils.video_thread import VideoThread


class VideoInferenceTab(QWidget):
    """Main tab for video inference"""

    def __init__(self):
        super().__init__()

        # Handlers and state
        self.video_handler = VideoHandler()
        self.inference_engine = YOLOInference()
        self.video_thread = None

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        """Setup the user interface"""
        main_layout = QHBoxLayout(self)

        # Left panel: Video list
        self.video_list_widget = VideoListWidget()
        self.video_list_widget.setMaximumWidth(250)
        self.video_list_widget.setMinimumWidth(200)
        main_layout.addWidget(self.video_list_widget, stretch=0)

        # Center: Video player
        self.video_player = VideoPlayerWidget()
        self.video_player.setMinimumWidth(600)
        main_layout.addWidget(self.video_player, stretch=3)

        # Right panel: Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Video folder selection
        folder_label = QLabel("Video Folder:")
        right_layout.addWidget(folder_label)

        self.folder_path_label = QLabel("Not selected")
        self.folder_path_label.setWordWrap(True)
        self.folder_path_label.setStyleSheet("color: gray;")
        right_layout.addWidget(self.folder_path_label)

        select_folder_btn = QPushButton("Select Folder")
        select_folder_btn.clicked.connect(self.select_video_folder)
        select_folder_btn.setFocusPolicy(Qt.NoFocus)
        right_layout.addWidget(select_folder_btn)

        right_layout.addSpacing(20)

        # Model loading
        model_label = QLabel("Model:")
        right_layout.addWidget(model_label)

        self.model_path_label = QLabel("Not loaded")
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setStyleSheet("color: gray;")
        right_layout.addWidget(self.model_path_label)

        load_model_btn = QPushButton("Load Model (.pt)")
        load_model_btn.clicked.connect(self.load_model)
        load_model_btn.setFocusPolicy(Qt.NoFocus)
        right_layout.addWidget(load_model_btn)

        right_layout.addSpacing(20)

        # Confidence threshold
        conf_label = QLabel("Confidence Threshold:")
        right_layout.addWidget(conf_label)

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setFocusPolicy(Qt.NoFocus)
        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)
        right_layout.addWidget(self.confidence_slider)

        self.confidence_value_label = QLabel("0.50")
        self.confidence_value_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.confidence_value_label)

        right_layout.addSpacing(20)

        # Inference toggle
        self.inference_checkbox = QCheckBox("Enable Inference")
        self.inference_checkbox.setChecked(True)
        self.inference_checkbox.setFocusPolicy(Qt.NoFocus)
        self.inference_checkbox.stateChanged.connect(self._on_inference_toggled)
        right_layout.addWidget(self.inference_checkbox)

        right_layout.addStretch()

        right_panel.setMaximumWidth(350)
        main_layout.addWidget(right_panel, stretch=1)

    def _setup_connections(self):
        """Setup signal-slot connections"""
        # Video list signals
        self.video_list_widget.video_selected.connect(self.on_video_selected)

        # Video player signals
        self.video_player.play_clicked.connect(self.on_play)
        self.video_player.pause_clicked.connect(self.on_pause)
        self.video_player.stop_clicked.connect(self.on_stop)
        self.video_player.seek_requested.connect(self.on_seek)
        self.video_player.step_forward_clicked.connect(self.on_step_forward)
        self.video_player.step_backward_clicked.connect(self.on_step_backward)
        self.video_player.slider_moved.connect(self.on_slider_moved)

    def select_video_folder(self):
        """Open dialog to select video folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder:
            self.video_handler.set_directory(folder)
            self.folder_path_label.setText(folder)
            self.folder_path_label.setStyleSheet("color: black;")

            # Update video list
            if self.video_handler.has_videos():
                self.video_list_widget.set_videos(
                    self.video_handler.videos_dir,
                    self.video_handler.video_files
                )
            else:
                QMessageBox.warning(self, "No Videos", "No video files found in the selected folder.")

    def load_model(self):
        """Open dialog to load YOLO model"""
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "YOLO Models (*.pt)"
        )

        if model_path:
            success = self.inference_engine.load_model(model_path)
            if success:
                self.model_path_label.setText(model_path)
                self.model_path_label.setStyleSheet("color: black;")
                QMessageBox.information(self, "Success", "Model loaded successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to load model. Please check the file.")

    def on_video_selected(self, index):
        """Handle video selection from list"""
        # Stop current playback
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

        # Update video handler
        if self.video_handler.goto_video(index):
            self.video_list_widget.set_current_video(index)
            video_path = self.video_handler.get_current_video_path()

            # Reset player
            self.video_player.reset()
            self.video_player.enable_controls(True)

            # Create new thread for this video
            self.video_thread = VideoThread()
            self.video_thread.set_video(video_path)
            self.video_thread.set_inference_engine(self.inference_engine)

            # Connect thread signals
            self.video_thread.frame_ready.connect(self.video_player.display_frame)
            self.video_thread.position_changed.connect(self.video_player.update_position)
            self.video_thread.duration_changed.connect(self.video_player.set_duration)
            self.video_thread.playback_finished.connect(self.video_player.on_playback_finished)
            self.video_thread.error_occurred.connect(self._on_video_error)

    def on_play(self):
        """Handle play button"""
        if self.video_thread:
            self.video_thread.play()

    def on_pause(self):
        """Handle pause button"""
        if self.video_thread:
            self.video_thread.pause()

    def on_stop(self):
        """Handle stop button"""
        if self.video_thread:
            self.video_thread.stop()

    def on_seek(self, position_ms):
        """Handle seek request"""
        if self.video_thread:
            self.video_thread.seek(position_ms)

    def on_step_forward(self):
        """Handle step forward button"""
        if self.video_thread:
            self.video_thread.step_forward()

    def on_step_backward(self):
        """Handle step backward button"""
        if self.video_thread:
            self.video_thread.step_backward()

    def on_slider_moved(self, position_ms):
        """Handle slider being moved (preview frame)"""
        if self.video_thread:
            self.video_thread.get_frame_at_position(position_ms)

    def _on_confidence_changed(self, value):
        """Handle confidence slider change"""
        confidence = value / 100.0
        self.confidence_value_label.setText(f"{confidence:.2f}")
        self.inference_engine.set_confidence(confidence)

    def _on_inference_toggled(self, state):
        """Handle inference checkbox toggle"""
        enabled = state == Qt.Checked
        self.inference_engine.set_enabled(enabled)

    def _on_video_error(self, error_msg):
        """Handle video error"""
        QMessageBox.critical(self, "Video Error", error_msg)

    def get_session_state(self) -> dict:
        """
        Get current state for session saving.

        Returns:
            Dictionary containing video tab state
        """
        return {
            "video_folder": self.video_handler.videos_dir,
            "model_path": self.inference_engine.model_path,
            "inference_threshold": self.inference_engine.confidence,
            "inference_enabled": self.inference_engine.enabled,
            "current_video_index": self.video_handler.get_current_index()
        }

    def restore_session_state(self, data: dict) -> list:
        """
        Restore state from session data.

        Args:
            data: Dictionary containing video tab state

        Returns:
            List of warning messages for paths that don't exist
        """
        import os
        warnings = []

        # Stop any current playback
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

        # Restore video folder
        video_folder = data.get("video_folder")
        if video_folder:
            if os.path.exists(video_folder):
                self.video_handler.set_directory(video_folder)
                self.folder_path_label.setText(video_folder)
                self.folder_path_label.setStyleSheet("color: black;")

                # Update video list
                if self.video_handler.has_videos():
                    self.video_list_widget.set_videos(
                        self.video_handler.videos_dir,
                        self.video_handler.video_files
                    )
            else:
                warnings.append(f"Video folder not found: {video_folder}")
        else:
            # Reset to default (no folder)
            self.video_handler.videos_dir = None
            self.video_handler.video_files = []
            self.video_handler.current_index = 0
            self.folder_path_label.setText("Not selected")
            self.folder_path_label.setStyleSheet("color: gray;")
            self.video_list_widget.clear()
            self.video_player.reset()

        # Restore model
        model_path = data.get("model_path")
        if model_path:
            if os.path.exists(model_path):
                success = self.inference_engine.load_model(model_path)
                if success:
                    self.model_path_label.setText(model_path)
                    self.model_path_label.setStyleSheet("color: black;")
                else:
                    warnings.append(f"Failed to load model: {model_path}")
            else:
                warnings.append(f"Model file not found: {model_path}")
        else:
            # Reset to default (no model)
            self.inference_engine.model = None
            self.inference_engine.model_path = None
            self.model_path_label.setText("Not loaded")
            self.model_path_label.setStyleSheet("color: gray;")

        # Restore confidence threshold
        threshold = data.get("inference_threshold", 0.5)
        slider_value = int(threshold * 100)
        self.confidence_slider.setValue(slider_value)
        self.confidence_value_label.setText(f"{threshold:.2f}")
        self.inference_engine.set_confidence(threshold)

        # Restore inference enabled state
        enabled = data.get("inference_enabled", True)
        self.inference_checkbox.setChecked(enabled)
        self.inference_engine.set_enabled(enabled)

        # Restore current video selection
        video_index = data.get("current_video_index", 0)
        if self.video_handler.has_videos() and video_index < self.video_handler.get_total_videos():
            self.on_video_selected(video_index)

        return warnings

    def closeEvent(self, event):
        """Handle widget close"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()
