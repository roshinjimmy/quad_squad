"""
Visualization Module
Handles drawing detection results on frames
"""

import cv2


class Visualizer:
    """Draw detection results on video frames"""
    
    def __init__(self, labels):
        """
        Initialize visualizer
        
        Args:
            labels (dict): Mapping of class IDs to label names
        """
        self.labels = labels
        self.color = (0, 255, 0)  # Green
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2
    
    def draw_detections(self, frame, boxes, classes, scores, threshold=0.5):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: OpenCV frame
            boxes: Detection boxes [ymin, xmin, ymax, xmax]
            classes: Class IDs
            scores: Confidence scores
            threshold (float): Minimum confidence threshold
            
        Returns:
            tuple: (annotated_frame, detected_objects)
        """
        h, w = frame.shape[:2]
        detected_objects = []
        
        for i, score in enumerate(scores):
            if score > threshold:
                # Get detection info
                class_id = int(classes[i])
                object_name = self.labels.get(class_id, f"Unknown-{class_id}")
                ymin, xmin, ymax, xmax = boxes[i]
                
                detected_objects.append((object_name, score))
                
                # Calculate pixel coordinates
                start_x, start_y = int(xmin * w), int(ymin * h)
                end_x, end_y = int(xmax * w), int(ymax * h)
                
                # Draw bounding box
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), 
                            self.color, self.thickness)
                
                # Prepare label
                label = f"{object_name}: {score:.2f}"
                label_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
                
                # Draw label background
                cv2.rectangle(frame, 
                            (start_x, start_y - label_size[1] - 10),
                            (start_x + label_size[0], start_y), 
                            self.color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (start_x, start_y - 5),
                          self.font, self.font_scale, (0, 0, 0), self.thickness)
        
        return frame, detected_objects
    
    def draw_info(self, frame, fps, detection_count, audio_enabled, threshold):
        """
        Draw information overlay on frame
        
        Args:
            frame: OpenCV frame
            fps (float): Current FPS
            detection_count (int): Number of detections
            audio_enabled (bool): Audio status
            threshold (float): Confidence threshold
        """
        h, w = frame.shape[:2]
        
        # Status info
        audio_status = "🔊 ON" if audio_enabled else "🔇 MUTED"
        info_text = f"FPS: {fps:.1f} | Detections: {detection_count} | Audio: {audio_status} | Threshold: {threshold:.2f}"
        
        cv2.putText(frame, info_text, (10, 30),
                   self.font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Controls
        controls = "Press 'q' to quit | 's' to toggle audio | '+/-' for threshold"
        cv2.putText(frame, controls, (10, h - 20),
                   self.font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return frame
