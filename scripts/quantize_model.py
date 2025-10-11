"""
Model Quantization Script
Converts SavedModel to optimized INT8 TFLite model
"""

import tensorflow as tf
import numpy as np
import glob
import os
from PIL import Image
from pathlib import Path

# Get project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"


print("="*70)
print("MobileNet Model Quantization Tool")
print("="*70)


def representative_dataset_gen():
    """Generate calibration data for INT8 quantization"""
    dataset_size = 100
    image_paths = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg")) + list(IMAGES_DIR.glob("*.png"))
    image_paths = [str(p) for p in image_paths]
    
    if image_paths:
        print(f"  Using {len(image_paths)} images for calibration")
        for i in range(min(dataset_size, len(image_paths) * 10)):
            img_path = image_paths[i % len(image_paths)]
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((320, 320))
                img_array = np.array(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                yield [img_array]
            except:
                yield [np.random.uniform(0, 255, (1, 320, 320, 3)).astype(np.float32)]
    else:
        print("  Using random calibration data")
        for _ in range(dataset_size):
            yield [np.random.uniform(0, 255, (1, 320, 320, 3)).astype(np.float32)]


def main():
    # Auto-detect saved model (supports both old and new naming)
    model_dirs = [
        MODELS_DIR / "efficientdet_lite0_savedmodel",
        MODELS_DIR / "ssd_mobilenet_v2_savedmodel",
        MODELS_DIR / "mobilenet_savedmodel"
    ]
    
    saved_model_path = None
    for path in model_dirs:
        if path.exists():
            saved_model_path = str(path)
            break
    
    if not saved_model_path:
        print(f"✗ Error: No SavedModel found in {MODELS_DIR}")
        print(f"Checked for:")
        for path in model_dirs:
            print(f"  - {path}")
        print("\nPlease run: python scripts/download_model.py first")
        return
    
    output_path = str(MODELS_DIR / "model.tflite")
    
    print(f"\n[1/3] Loading SavedModel from: {saved_model_path}")
    
    if not os.path.exists(saved_model_path):
        print(f"✗ Error: SavedModel not found at {saved_model_path}")
        print("Please run: python scripts/download_model.py first")
        return
    
    try:
        # Try loading the model to check signatures
        print("  Inspecting model signatures...")
        loaded_model = tf.saved_model.load(saved_model_path)
        
        # Check available signatures
        if hasattr(loaded_model, 'signatures'):
            signatures = list(loaded_model.signatures.keys())
            print(f"  Available signatures: {signatures}")
            
            if signatures:
                # Use the first available signature
                signature_key = signatures[0]
                print(f"  Using signature: {signature_key}")
                converter = tf.lite.TFLiteConverter.from_saved_model(
                    saved_model_path,
                    signature_keys=[signature_key]
                )
            else:
                print("  No signatures found, using default serving signature...")
                converter = tf.lite.TFLiteConverter.from_saved_model(
                    saved_model_path,
                    signature_keys=['serving_default']
                )
        else:
            print("  Using default conversion method...")
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        # Create converter
        # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        print("\n[2/3] Applying INT8 quantization...")
        
        # Enable quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = representative_dataset_gen
        
        # Convert
        print("  Converting (this may take a few minutes)...")
        tflite_model = converter.convert()
        
        print("\n[3/3] Saving quantized model...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Model saved: {output_path}")
        print(f"  Size: {model_size:.2f} MB")
        
        print("\n" + "="*70)
        print("SUCCESS!")
        print(f"Quantized model ready: {output_path}")
        print("\nRun the application:")
        print("  python main.py")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Full quantization failed: {e}")
        print("\nTrying alternative conversion method...")
        
        try:
            # Alternative: Download pre-quantized TFLite model directly
            print("Attempting to download pre-converted TFLite model...")
            import urllib.request
            
            # EfficientDet-Lite0 pre-converted TFLite model URL
            tflite_url = "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/default/1?lite-format=tflite"
            
            print(f"Downloading from: {tflite_url}")
            urllib.request.urlretrieve(tflite_url, output_path)
            
            model_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n✓ Pre-converted model downloaded: {output_path}")
            print(f"  Size: {model_size:.2f} MB")
            
            print("\n" + "="*70)
            print("SUCCESS!")
            print(f"EfficientDet-Lite0 TFLite model ready: {output_path}")
            print("\nRun the application:")
            print("  python main.py")
            print("="*70)
            return  # Exit successfully
            
        except Exception as e2:
            print(f"\n✗ Alternative method also failed: {e2}")
            print("\nPlease check your internet connection or try a different model.")
            return


if __name__ == "__main__":
    main()
