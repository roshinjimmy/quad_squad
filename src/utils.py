"""
Utility Functions
Helper functions for loading labels, FPS calculation, etc.
"""

import time


def load_labels(label_path):
    """
    Load label map from a text file
    
    Args:
        label_path (str): Path to label file
        
    Returns:
        dict: Mapping of class ID to label name
    """
    labels = {}
    with open(label_path, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split(':')
                class_id = int(parts[0])
                label_name = parts[1].strip().replace('"', '')
                labels[class_id] = label_name
            except (ValueError, IndexError):
                continue
    
    return labels


class FPSCounter:
    """Calculate frames per second"""
    
    def __init__(self, update_interval=30):
        """
        Initialize FPS counter
        
        Args:
            update_interval (int): Update FPS every N frames
        """
        self.update_interval = update_interval
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
    
    def update(self):
        """Update FPS counter"""
        self.frame_count += 1
        
        if self.frame_count % self.update_interval == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.update_interval / elapsed
            self.start_time = time.time()
        
        return self.fps
    
    def get(self):
        """Get current FPS"""
        return self.fps
