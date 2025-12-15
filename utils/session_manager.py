"""
Session manager for saving and loading application state.
"""
import json
import os
from typing import Optional
from pathlib import Path


class SessionManager:
    """Manages saving and loading of session files."""

    SESSION_VERSION = "1.0"

    # Store last session path in user's home directory
    LAST_SESSION_FILE = os.path.join(str(Path.home()), ".yolov8_annotator_last_session.txt")

    @staticmethod
    def get_default_session() -> dict:
        """Return default session structure with empty values."""
        return {
            "version": SessionManager.SESSION_VERSION,
            "annotation_tab": {
                "images_folder": None,
                "labels_folder": None,
                "current_image_index": 0
            },
            "video_tab": {
                "video_folder": None,
                "model_path": None,
                "inference_threshold": 0.5,
                "inference_enabled": True,
                "current_video_index": 0
            }
        }

    @staticmethod
    def validate_session(data: dict) -> bool:
        """Validate that session data has the required structure."""
        if not isinstance(data, dict):
            return False

        if "version" not in data:
            return False

        # Check annotation_tab structure
        if "annotation_tab" not in data:
            return False
        annotation_tab = data["annotation_tab"]
        required_annotation_keys = ["images_folder", "labels_folder", "current_image_index"]
        if not all(key in annotation_tab for key in required_annotation_keys):
            return False

        # Check video_tab structure
        if "video_tab" not in data:
            return False
        video_tab = data["video_tab"]
        required_video_keys = [
            "video_folder", "model_path", "inference_threshold",
            "inference_enabled", "current_video_index"
        ]
        if not all(key in video_tab for key in required_video_keys):
            return False

        return True

    @staticmethod
    def save_session(filepath: str, session_data: dict) -> bool:
        """
        Save session data to a JSON file.

        Args:
            filepath: Path to save the session file
            session_data: Session data dictionary

        Returns:
            True if save was successful, False otherwise
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            return True
        except (IOError, OSError, TypeError) as e:
            print(f"Error saving session: {e}")
            return False

    @staticmethod
    def load_session(filepath: str) -> Optional[dict]:
        """
        Load session data from a JSON file.

        Args:
            filepath: Path to the session file

        Returns:
            Session data dictionary if successful, None otherwise
        """
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not SessionManager.validate_session(data):
                print(f"Invalid session file structure: {filepath}")
                return None

            return data
        except (IOError, OSError, json.JSONDecodeError) as e:
            print(f"Error loading session: {e}")
            return None

    @staticmethod
    def save_last_session_path(filepath: str) -> None:
        """
        Save the path to the last opened session file.

        Args:
            filepath: Path to the session file
        """
        try:
            with open(SessionManager.LAST_SESSION_FILE, 'w', encoding='utf-8') as f:
                f.write(filepath)
        except (IOError, OSError) as e:
            print(f"Error saving last session path: {e}")

    @staticmethod
    def get_last_session_path() -> Optional[str]:
        """
        Get the path to the last opened session file.

        Returns:
            Path to the last session file, or None if not found
        """
        if not os.path.exists(SessionManager.LAST_SESSION_FILE):
            return None

        try:
            with open(SessionManager.LAST_SESSION_FILE, 'r', encoding='utf-8') as f:
                filepath = f.read().strip()
                # Only return the path if the file still exists
                if filepath and os.path.exists(filepath):
                    return filepath
                return None
        except (IOError, OSError) as e:
            print(f"Error loading last session path: {e}")
            return None

    @staticmethod
    def clear_last_session_path() -> None:
        """Clear the last session path (e.g., when creating a new session)."""
        try:
            if os.path.exists(SessionManager.LAST_SESSION_FILE):
                os.remove(SessionManager.LAST_SESSION_FILE)
        except (IOError, OSError) as e:
            print(f"Error clearing last session path: {e}")
