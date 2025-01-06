import cv2 
import numpy as np
from ultralytics import YOLO

class ObstacleDetector:
    def __init__(self, model_path=r"C:\Users\ishak\yolov8s.pt"):
        # Load YOLO model
        self.model = YOLO(model_path)

    def detect_obstacles(self, frame):
        """
        Runs YOLO detection on the frame.
        Returns:
            obstacle_boxes: list of (x1, y1, x2, y2, confidence, class_id)
            annotated_frame: the frame with bounding boxes drawn for visualization
        """
        
        results = self.model.predict(
            frame, 
            conf=0.3,       # Lower confidence threshold
            iou=0.45,       # Adjust as needed
            imgsz=1280,     # Increase resolution
            verbose=False
        )

        annotated_frame = frame.copy()
        obstacle_boxes = []

        if len(results) > 0:
            # Each 'results' element corresponds to a single image in a batch
            # Here we assume only one frame at a time, so we look at results[0]
            dets = results[0].boxes
            for det in dets:
                # YOLOv8 format gives us [x1, y1, x2, y2], conf, cls
                x1, y1, x2, y2 = det.xyxy[0]
                conf = float(det.conf[0])
                cls_id = int(det.cls[0])
                
                # Add the bounding boxes to our list
                obstacle_boxes.append((int(x1), int(y1), int(x2), int(y2), conf, cls_id))
                
                # Draw bounding box on the annotated frame
                cv2.rectangle(
                    annotated_frame, 
                    (int(x1), int(y1)), 
                    (int(x2), int(y2)), 
                    (0, 0, 255), 
                    2
                )
                # Display class/conf text above the bounding box
                cv2.putText(
                    annotated_frame,
                    f"Obj_{cls_id}: {conf:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

        return obstacle_boxes, annotated_frame
