"""
Audio Feedback Module
Handles text-to-speech announcements for detected objects
"""

import pyttsx3
import threading
import time


class AudioFeedback:
    """Text-to-speech audio feedback system"""
    
    def __init__(self, rate=150, volume=0.9, cooldown=3):
        """
        Initialize audio feedback system
        
        Args:
            rate (int): Speech rate (words per minute)
            volume (float): Volume level (0.0 to 1.0)
            cooldown (int): Seconds between announcements for same object
        """
        print("Initializing text-to-speech engine...")
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        self.audio_queue = []
        self.audio_lock = threading.Lock()
        self.last_announcement_time = {}
        self.cooldown = cooldown
        self.enabled = True
        
        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()
        
        print("✓ Audio system initialized")
    
    def _audio_worker(self):
        """Background thread to handle text-to-speech"""
        while True:
            with self.audio_lock:
                if self.audio_queue:
                    message = self.audio_queue.pop(0)
                    if self.enabled:
                        print(f"🔊 Speaking: {message}")
                        self.engine.say(message)
                        self.engine.runAndWait()
            time.sleep(0.1)
    
    def announce(self, object_name, confidence=None):
        """
        Announce detected object
        
        Args:
            object_name (str): Name of detected object
            confidence (float, optional): Confidence score
        """
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # Check cooldown to avoid spam
        if object_name in self.last_announcement_time:
            if current_time - self.last_announcement_time[object_name] < self.cooldown:
                return
        
        self.last_announcement_time[object_name] = current_time
        
        # Create message
        message = f"{object_name} detected"
        
        with self.audio_lock:
            if message not in self.audio_queue:
                self.audio_queue.append(message)
    
    def toggle(self):
        """Toggle audio on/off"""
        self.enabled = not self.enabled
        return self.enabled
    
    def is_enabled(self):
        """Check if audio is enabled"""
        return self.enabled
    
    def set_cooldown(self, seconds):
        """Set cooldown time between announcements"""
        self.cooldown = seconds
