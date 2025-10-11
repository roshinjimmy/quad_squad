import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detector import ObjectDetector
from audio import AudioFeedback
from camera import Camera
from visualizer import Visualizer
from utils import load_labels, FPSCounter


# Configuration
CONFIG = {
    'model_path': 'models/model.tflite',
    'label_path': 'data/labels.txt',
    'camera_source': 0,
    'confidence_threshold': 0.5,
    'window_name': 'Echo Frame - Object Detection',
}


def main():
    """Main application loop"""
    
    print("="*70)
    print("ECHO FRAME - Real-Time Object Detection with Audio Feedback")
    print("="*70)
    
    try:
        # Initialize components
        print("\n[1/4] Loading labels...")
        labels = load_labels(CONFIG['label_path'])
        print(f"✓ Loaded {len(labels)} labels")
        
        print("\n[2/4] Initializing detector...")
        detector = ObjectDetector(CONFIG['model_path'])
        
        print("\n[3/4] Setting up audio feedback...")
        audio = AudioFeedback()
        
        print("\n[4/4] Opening camera...")
        camera = Camera(CONFIG['camera_source'])
        
        # Initialize visualizer and FPS counter
        visualizer = Visualizer(labels)
        fps_counter = FPSCounter()
        
        # Settings
        confidence_threshold = CONFIG['confidence_threshold']
        
        print("\n" + "="*70)
        print("CONTROLS:")
        print("  'q' - Quit")
        print("  's' - Toggle audio (mute/unmute)")
        print("  '+' - Increase confidence threshold")
        print("  '-' - Decrease confidence threshold")
        print("="*70)
        print("\nStarting detection...\n")
        
        # Main loop
        while True:
            # Read frame
            ret, frame = camera.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Run detection
            boxes, classes, scores, num_detections = detector.detect(frame)
            
            # Draw results
            frame, detected_objects = visualizer.draw_detections(
                frame, boxes, classes, scores, confidence_threshold
            )
            
            # Announce detections
            for obj_name, score in detected_objects:
                audio.announce(obj_name, score)
            
            # Update FPS
            fps = fps_counter.update()
            
            # Draw info overlay
            frame = visualizer.draw_info(
                frame, fps, len(detected_objects), 
                audio.is_enabled(), confidence_threshold
            )
            
            # Display
            cv2.imshow(CONFIG['window_name'], frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                audio_status = "enabled" if audio.toggle() else "disabled"
                print(f"Audio {audio_status}")
            elif key == ord('+') or key == ord('='):
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                print(f"Threshold: {confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                confidence_threshold = max(0.1, confidence_threshold - 0.05)
                print(f"Threshold: {confidence_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        if 'camera' in locals():
            camera.release()
        cv2.destroyAllWindows()
        print("Goodbye!")


if __name__ == "__main__":
    main()
