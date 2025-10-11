import tensorflow_hub as hub
import tensorflow as tf
import os
from pathlib import Path

# Get project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

print("="*70)
print("Object Detection Model Downloader")
print("="*70)

# Model URLs - Choose one:
MODELS = {
    'ssd_mobilenet_v2': "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
    'efficientdet_lite0': "https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1",
    'efficientdet_lite1': "https://tfhub.dev/tensorflow/efficientdet/lite1/detection/1",
}

# Select model (EfficientDet Lite0 is better than MobileNet V2)
MODEL_NAME = 'efficientdet_lite0'  # Best choice for ESP32
MODEL_URL = MODELS[MODEL_NAME]

print(f"\nSelected model: {MODEL_NAME}")
print(f"[1/2] Downloading model from TensorFlow Hub...")
print(f"URL: {MODEL_URL}")

try:
    # Load model
    model = hub.load(MODEL_URL)
    print("✓ Model downloaded successfully!")
    
    # Save path (using absolute path)
    MODELS_DIR.mkdir(exist_ok=True)
    save_path = str(MODELS_DIR / f"{MODEL_NAME}_savedmodel")
    
    print(f"\n[2/2] Saving model to: {save_path}")
    tf.saved_model.save(model, save_path)
    print("✓ Model saved successfully!")
    
    print("\n" + "="*70)
    print("SUCCESS!")
    print(f"Model: {MODEL_NAME}")
    print(f"Saved to: {save_path}")
    print("\nModel Comparison:")
    print("  • SSD MobileNet V2:    Good baseline")
    print("  • EfficientDet Lite0:  Better accuracy, similar speed ✓")
    print("  • EfficientDet Lite1:  Best accuracy, slower")
    print("\nNext steps:")
    print("  1. Run: python scripts/quantize_model.py")
    print("  2. Then run: python main.py")
    print("="*70)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
