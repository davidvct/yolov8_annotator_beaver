"""
Video list widget for browsing video files.
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
                                QLabel, QLineEdit, QHBoxLayout, QMenu, QApplication)
from PySide6.QtCore import Qt, Signal, QMimeData, QUrl


class VideoListWidget(QWidget):
    """Widget displaying a list of videos (filenames only for fast loading)"""

    video_selected = Signal(int)  # Emits the index of the selected video

    def __init__(self):
        super().__init__()

        self.video_files = []
        self.videos_dir = None
        self.current_index = -1

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Videos")
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title)

        # Search/filter box
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search videos...")
        self.filter_input.textChanged.connect(self._filter_videos)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_input)
        layout.addLayout(filter_layout)

        # Video count label
        self.count_label = QLabel("0 videos")
        self.count_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(self.count_label)

        # Video list
        self.list_widget = QListWidget()
        self.list_widget.setSpacing(2)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.list_widget.setStyleSheet("QListWidget::item:selected { background-color: #1e90ff; color: white; }")
        
        # Enable custom context menu
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.list_widget)

    def set_videos(self, videos_dir, video_files):
        """
        Set the list of videos to display

        Args:
            videos_dir: Directory containing the videos
            video_files: List of video filenames
        """
        self.videos_dir = videos_dir
        self.video_files = video_files
        self.current_index = -1

        # Clear existing items
        self.list_widget.clear()
        self.filter_input.clear()

        # Update count
        self.count_label.setText(f"{len(video_files)} videos")

        # Add items with just filenames
        for i, filename in enumerate(video_files):
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, i)  # Store original index
            self.list_widget.addItem(item)

    def _filter_videos(self, text):
        """Filter video list based on search text"""
        search_text = text.lower()

        visible_count = 0
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item:
                filename = item.text().lower()
                matches = search_text in filename
                item.setHidden(not matches)
                if matches:
                    visible_count += 1

        # Update count label
        if search_text:
            self.count_label.setText(f"{visible_count} of {len(self.video_files)} videos")
        else:
            self.count_label.setText(f"{len(self.video_files)} videos")

    def _on_item_clicked(self, item):
        """Handle item click event"""
        index = item.data(Qt.UserRole)
        self.video_selected.emit(index)

    def set_current_video(self, index):
        """
        Highlight the current video in the list

        Args:
            index: Index of the current video
        """
        self.current_index = index

        # Clear previous selection
        self.list_widget.clearSelection()

        # Find and select the item with this index
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.UserRole) == index:
                item.setSelected(True)
                self.list_widget.scrollToItem(item, QListWidget.PositionAtCenter)
                break

    def clear(self):
        """Clear the video list"""
        self.list_widget.clear()
        self.video_files = []
        self.videos_dir = None
        self.current_index = -1
        self.count_label.setText("0 videos")

    def _show_context_menu(self, pos):
        """Show context menu for list items"""
        item = self.list_widget.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        
        copy_filename_action = menu.addAction("Copy Filename")
        copy_filename_action.triggered.connect(lambda: self._copy_filename(item))
        
        copy_path_action = menu.addAction("Copy Full Path")
        copy_path_action.triggered.connect(lambda: self._copy_full_path(item))
        
        copy_file_action = menu.addAction("Copy File")
        copy_file_action.triggered.connect(lambda: self._copy_file(item))

        menu.exec_(self.list_widget.mapToGlobal(pos))

    def _copy_filename(self, item):
        """Copy just the filename to clipboard"""
        filename = item.text()
        clipboard = QApplication.clipboard()
        clipboard.setText(filename)

    def _copy_full_path(self, item):
        """Copy full file path to clipboard"""
        filename = item.text()
        if self.videos_dir:
            full_path = f"{self.videos_dir}/{filename}"
            # Normalize path separators
            full_path = full_path.replace("/", "\\")
            clipboard = QApplication.clipboard()
            clipboard.setText(full_path)

    def _copy_file(self, item):
        """Copy file object to clipboard (pasteable in Explorer)"""
        filename = item.text()
        if self.videos_dir:
            full_path = f"{self.videos_dir}/{filename}"
            # Normalize path separators for Windows
            full_path = full_path.replace("/", "\\")
            
            mime_data = QMimeData()
            url = QUrl.fromLocalFile(full_path)
            mime_data.setUrls([url])
            
            clipboard = QApplication.clipboard()
            clipboard.setMimeData(mime_data)
