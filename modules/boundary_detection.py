import cv2
import numpy as np

def detect_arena_boundary(frame):
    """
    Detect the boundary of the arena using color thresholding or line detection.
    Returns:
        boundary_contours: list of contours representing the boundary
        annotated_frame: the input frame with boundary marked (for debugging)
    """
    # Convert to HSV (for robust color-based thresholding)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Example threshold: This depends on the boundary color in your scenario
    # Lower and upper bounds for a distinctive boundary color
    lower_color = np.array([0, 0, 200])   # (dummy values)
    upper_color = np.array([180, 30, 255]) # (dummy values)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Morphological operations to clean the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Optionally, pick the largest contour or certain shape criteria
    boundary_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  # threshold depends on actual boundary size
            boundary_contours.append(cnt)
    
    # For debug annotation
    annotated_frame = frame.copy()
    cv2.drawContours(annotated_frame, boundary_contours, -1, (0, 255, 0), 3)

    return boundary_contours, annotated_frame
