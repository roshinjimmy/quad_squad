"""
Object Detection Module
Handles TensorFlow Lite model loading and inference
"""

import tensorflow as tf
import numpy as np


class ObjectDetector:
    """TensorFlow Lite object detection wrapper"""
    
    def __init__(self, model_path):
        """
        Initialize the detector with a TFLite model
        
        Args:
            model_path (str): Path to the .tflite model file
        """
        print(f"Loading model from: {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input dimensions
        input_shape = self.input_details[0]['shape']
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        
        print(f"✓ Model loaded | Input size: {self.input_width}x{self.input_height}")
    
    def preprocess(self, frame):
        """
        Preprocess frame for model input
        
        Args:
            frame: OpenCV BGR frame
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Convert BGR to RGB
        image_rgb = frame[:, :, ::-1]  # Faster than cv2.cvtColor
        
        # Resize to model input size
        import cv2
        image_resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
        
        # Add batch dimension
        input_tensor = np.expand_dims(image_resized, axis=0).astype(np.uint8)
        
        return input_tensor
    
    def detect(self, frame):
        """
        Run object detection on a frame
        
        Args:
            frame: OpenCV BGR frame
            
        Returns:
            tuple: (boxes, classes, scores, num_detections)
                - boxes: Bounding box coordinates [ymin, xmin, ymax, xmax]
                - classes: Class IDs for detected objects
                - scores: Confidence scores
                - num_detections: Number of valid detections
        """
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get outputs
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index']).item())
        
        return boxes[0], classes[0], scores[0], num_detections
    
    def get_input_size(self):
        """Return model input dimensions"""
        return (self.input_width, self.input_height)
