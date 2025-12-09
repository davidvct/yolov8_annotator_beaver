"""
Data models for managing annotations in the UI.
"""
from typing import List, Tuple
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor


class Annotation:
    """Represents a polygon annotation with UI state"""

    def __init__(self, class_id: int, points: List[Tuple[float, float]], class_name: str = ""):
        """
        Args:
            class_id: Integer class ID
            points: List of (x, y) tuples in normalized coordinates (0-1)
            class_name: Human-readable class name
        """
        self.class_id = class_id
        self.points = points  # Normalized coordinates (0-1)
        self.class_name = class_name
        self.selected = False
        self.color = self._generate_color(class_id)

    def _generate_color(self, class_id: int) -> QColor:
        """Generate a consistent color for each class ID"""
        colors = [
            QColor(255, 0, 0, 150),      # Red
            QColor(0, 255, 0, 150),      # Green
            QColor(0, 0, 255, 150),      # Blue
            QColor(255, 255, 0, 150),    # Yellow
            QColor(255, 0, 255, 150),    # Magenta
            QColor(0, 255, 255, 150),    # Cyan
            QColor(255, 128, 0, 150),    # Orange
            QColor(128, 0, 255, 150),    # Purple
            QColor(0, 128, 255, 150),    # Sky Blue
            QColor(255, 192, 203, 150),  # Pink
        ]
        return colors[class_id % len(colors)]

    def get_pixel_points(self, img_width: int, img_height: int) -> List[QPointF]:
        """Convert normalized points to pixel coordinates as QPointF"""
        return [QPointF(x * img_width, y * img_height) for x, y in self.points]

    def contains_point(self, point: Tuple[float, float], img_width: int, img_height: int,
                      tolerance: float = 10.0) -> bool:
        """Check if a pixel point is near any vertex of the polygon"""
        pixel_points = [(x * img_width, y * img_height) for x, y in self.points]
        px, py = point

        for vx, vy in pixel_points:
            distance = ((px - vx) ** 2 + (py - vy) ** 2) ** 0.5
            if distance <= tolerance:
                return True
        return False

    def get_nearest_vertex(self, point: Tuple[float, float], img_width: int,
                          img_height: int) -> int:
        """Get the index of the nearest vertex to the given pixel point"""
        pixel_points = [(x * img_width, y * img_height) for x, y in self.points]
        px, py = point

        min_distance = float('inf')
        nearest_index = -1

        for i, (vx, vy) in enumerate(pixel_points):
            distance = ((px - vx) ** 2 + (py - vy) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_index = i

        return nearest_index

    def update_vertex(self, vertex_index: int, new_point: Tuple[float, float],
                     img_width: int, img_height: int) -> None:
        """Update a vertex position (new_point in pixel coordinates)"""
        if 0 <= vertex_index < len(self.points):
            # Convert pixel to normalized
            norm_x = new_point[0] / img_width
            norm_y = new_point[1] / img_height
            # Clamp to 0-1 range
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            self.points[vertex_index] = (norm_x, norm_y)

    def is_inside_polygon(self, point: Tuple[float, float], img_width: int,
                         img_height: int) -> bool:
        """Check if a pixel point is inside the polygon using ray casting algorithm"""
        pixel_points = [(x * img_width, y * img_height) for x, y in self.points]
        px, py = point
        n = len(pixel_points)
        inside = False

        p1x, p1y = pixel_points[0]
        for i in range(1, n + 1):
            p2x, p2y = pixel_points[i % n]
            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or px <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def update_class(self, class_id: int, class_name: str = "") -> None:
        """Update the class ID and name, and regenerate the color"""
        self.class_id = class_id
        self.class_name = class_name
        self.color = self._generate_color(class_id)
