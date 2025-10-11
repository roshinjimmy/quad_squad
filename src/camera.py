"""
Camera Module
Handles webcam/video capture
"""

import cv2


class Camera:
    """Webcam/video capture wrapper"""
    
    def __init__(self, source=0):
        """
        Initialize camera
        
        Args:
            source: Camera index (0 for default) or video file path
        """
        print(f"Opening camera (source: {source})...")
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera: {source}")
        
        # Get camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✓ Camera opened | Resolution: {self.width}x{self.height} @ {self.fps}fps")
    
    def read(self):
        """
        Read a frame from camera
        
        Returns:
            tuple: (success, frame)
        """
        return self.cap.read()
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            print("Camera released")
    
    def get_resolution(self):
        """Get camera resolution"""
        return (self.width, self.height)
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()
