"""
Undo/Redo manager for annotation operations.
"""
import copy
from typing import List
from models.annotation import Annotation


class UndoRedoManager:
    """Manages undo/redo operations for annotations"""

    def __init__(self, max_history: int = 50):
        """
        Initialize the undo/redo manager.

        Args:
            max_history: Maximum number of states to keep in history
        """
        self.max_history = max_history
        self.history: List[List[Annotation]] = []
        self.current_index = -1
        self.saved_index = -1  # Track which state was last saved

    def push_state(self, annotations: List[Annotation]) -> None:
        """
        Push a new state to the history.

        Args:
            annotations: Current list of annotations to save
        """
        # Deep copy the annotations to preserve the state
        state = self._deep_copy_annotations(annotations)

        # Remove any states after current_index (when new action is performed after undo)
        self.history = self.history[:self.current_index + 1]

        # Add new state
        self.history.append(state)

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            # Adjust saved_index if we removed a state before it
            if self.saved_index > 0:
                self.saved_index -= 1
            elif self.saved_index == 0:
                self.saved_index = -1  # Saved state was removed
        else:
            self.current_index += 1

    def can_undo(self) -> bool:
        """Check if undo is available"""
        return self.current_index > 0

    def can_redo(self) -> bool:
        """Check if redo is available"""
        return self.current_index < len(self.history) - 1

    def undo(self) -> List[Annotation]:
        """
        Undo to the previous state.

        Returns:
            The previous state of annotations, or current state if undo is not available
        """
        if self.can_undo():
            self.current_index -= 1
            return self._deep_copy_annotations(self.history[self.current_index])

        # Return current state if undo is not available
        if self.current_index >= 0 and self.current_index < len(self.history):
            return self._deep_copy_annotations(self.history[self.current_index])

        return []

    def redo(self) -> List[Annotation]:
        """
        Redo to the next state.

        Returns:
            The next state of annotations, or current state if redo is not available
        """
        if self.can_redo():
            self.current_index += 1
            return self._deep_copy_annotations(self.history[self.current_index])

        # Return current state if redo is not available
        if self.current_index >= 0 and self.current_index < len(self.history):
            return self._deep_copy_annotations(self.history[self.current_index])

        return []

    def clear(self) -> None:
        """Clear all history"""
        self.history.clear()
        self.current_index = -1
        self.saved_index = -1

    def mark_saved(self) -> None:
        """Mark the current state as saved"""
        self.saved_index = self.current_index

    def is_saved(self) -> bool:
        """Check if the current state matches the saved state"""
        return self.current_index == self.saved_index

    def _deep_copy_annotations(self, annotations: List[Annotation]) -> List[Annotation]:
        """
        Create a deep copy of annotations list.

        Args:
            annotations: List of annotations to copy

        Returns:
            Deep copy of the annotations
        """
        copied_annotations = []
        for ann in annotations:
            # Create a new annotation with copied data
            new_ann = Annotation(
                class_id=ann.class_id,
                points=copy.deepcopy(ann.points),
                class_name=ann.class_name
            )
            new_ann.selected = ann.selected
            copied_annotations.append(new_ann)

        return copied_annotations
