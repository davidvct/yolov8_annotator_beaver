"""
YOLO model inference utilities.
"""
import numpy as np
from typing import Optional


class YOLOInference:
    """Handles YOLO model loading and inference"""

    def __init__(self):
        self.models = {}  # Dictionary to store loaded models: {slot_index: model}
        self.item_paths = {} # Dictionary to store model paths: {slot_index: path}
        self.active_slot = 0 # Currently active slot
        self.confidence = 0.5
        self.enabled = True

    def load_model(self, path: str, slot_index: int = 0) -> bool:
        """
        Load a YOLO model from .pt file into a specific slot

        Args:
            path: Path to .pt model file
            slot_index: Index of the slot (e.g., 0 or 1)

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            self.models[slot_index] = model
            self.item_paths[slot_index] = path
            
            # If this is the only model or we are loading into the active slot, it's ready.
            # But we don't necessarily force switch unless requested. 
            # For simplicity, if we load into the active slot, it updates immediately.
            return True
        except Exception as e:
            print(f"Error loading model into slot {slot_index}: {e}")
            # If load failed, clear that slot
            if slot_index in self.models:
                del self.models[slot_index]
            if slot_index in self.item_paths:
                del self.item_paths[slot_index]
            return False

    def unload_model(self, slot_index: int = 0):
        """Unload the model from a specific slot"""
        if slot_index in self.models:
            del self.models[slot_index]
        if slot_index in self.item_paths:
            del self.item_paths[slot_index]

    def set_active_slot(self, slot_index: int):
        """Set the active model slot"""
        self.active_slot = slot_index

    def predict(self, frame: np.ndarray):
        """
        Run inference on a frame using the active model

        Args:
            frame: Input frame (numpy array in BGR format from OpenCV)

        Returns:
            Results object from ultralytics, or None if active model not loaded
        """
        if not self.enabled:
            return None
            
        model = self.models.get(self.active_slot)
        if model is None:
            return None

        try:
            results = model(frame, conf=self.confidence, verbose=False)
            return results[0] if results else None
        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    def draw_results(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw bounding boxes, labels, confidence scores, and masks on frame

        Args:
            frame: Input frame
            results: Results object from predict()

        Returns:
            Annotated frame
        """
        if results is None:
            return frame

        try:
            # Use ultralytics built-in plotting
            annotated_frame = results.plot()
            return annotated_frame
        except Exception as e:
            print(f"Error drawing results: {e}")
            return frame

    def set_confidence(self, value: float):
        """
        Set confidence threshold

        Args:
            value: Confidence threshold (0.0 to 1.0)
        """
        self.confidence = max(0.0, min(1.0, value))

    def set_enabled(self, enabled: bool):
        """
        Enable or disable inference

        Args:
            enabled: True to enable, False to disable
        """
        self.enabled = enabled

    def is_loaded(self, slot_index: Optional[int] = None) -> bool:
        """
        Check if a model is loaded. 
        If slot_index is provided, checks that specific slot.
        Otherwise checks the active slot.
        """
        target_slot = slot_index if slot_index is not None else self.active_slot
        return target_slot in self.models

    def get_model_path(self, slot_index: Optional[int] = None) -> Optional[str]:
        """
        Get the path of the loaded model.
        If slot_index is provided, returns path for that slot.
        Otherwise returns path for active slot.
        """
        target_slot = slot_index if slot_index is not None else self.active_slot
        return self.item_paths.get(target_slot)
