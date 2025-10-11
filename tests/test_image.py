"""
Test Object Detection on Single Image
Quick test script for model validation
"""

import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detector import ObjectDetector
from visualizer import Visualizer
from utils import load_labels


def test_image(image_path, model_path='models/model.tflite', label_path='data/labels.txt'):
    """Test detection on a single image"""
    
    print("Loading labels...")
    labels = load_labels(label_path)
    
    print("Loading detector...")
    detector = ObjectDetector(model_path)
    
    print("Loading image...")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    print("Running detection...")
    boxes, classes, scores, num_detections = detector.detect(frame)
    
    print("Visualizing results...")
    visualizer = Visualizer(labels)
    frame, detected_objects = visualizer.draw_detections(frame, boxes, classes, scores, threshold=0.5)
    
    print(f"\nDetections ({len(detected_objects)}):")
    for obj_name, score in detected_objects:
        print(f"  - {obj_name}: {score:.2f}")
    
    # Display
    cv2.imshow('Detection Result', frame)
    print("\nPress any key to close...")
    
    # Wait for key press or window close
    while True:
        key = cv2.waitKey(100)
        if key != -1:  # Any key pressed
            break
        if cv2.getWindowProperty('Detection Result', cv2.WND_PROP_VISIBLE) < 1:  # Window closed
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image(sys.argv[1])
    else:
        # Default test image
        test_image("images/car.jpg")
