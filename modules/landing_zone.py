import cv2
import numpy as np

def detect_safe_landing_zone(color_frame, depth_frame=None):
    """
    Identify candidate landing regions based on:
      1) Minimal obstacles (using color or segmentation).
      2) Flatness (using depth or 3D reconstruction).
    Returns:
      safe_zones: list of bounding boxes or contours representing safe areas
      annotated_frame: for visualization
    """

    # 1. (Optional) Segment out ground vs. obstacles
    #    For demonstration, let's do a very naive approach:
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Morphological close to remove small specks
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Depth-based filtering (if depth_frame is provided)
    #    Example: measure variance of depth within the contour
    safe_zones = []
    annotated_frame = color_frame.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500: 
            continue  # skip small areas

        # Approx bounding rect
        x, y, w, h = cv2.boundingRect(cnt)

        # If depth is available, check flatness
        if depth_frame is not None:
            depth_roi = depth_frame[y:y+h, x:x+w]
            mean, stddev = cv2.meanStdDev(depth_roi)
            if stddev[0][0] < 5.0:  # example threshold; depends on sensor noise
                # This area is "flat enough" for a potential landing
                safe_zones.append((x, y, w, h))
        else:
            # If no depth, we just treat large open areas as "safe"
            safe_zones.append((x, y, w, h))

    # Annotate safe zones in green
    for (x, y, w, h) in safe_zones:
        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return safe_zones, annotated_frame
