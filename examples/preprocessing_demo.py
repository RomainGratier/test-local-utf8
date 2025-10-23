#!/usr/bin/env python3
"""
Image Preprocessing Pipeline Demo

This script demonstrates the image preprocessing capabilities including
resize, normalization, format conversion, and batch processing.
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.transforms import (
    ImagePreprocessor,
    preprocess_single,
    preprocess_batch,
    validate_image,
    get_image_info
)


def create_sample_images(output_dir: str = "sample_images"):
    """Create sample images for demonstration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create RGB image
    rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(output_dir, "sample_rgb.jpg"), 
                cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    
    # Create grayscale image
    gray_image = np.random.randint(0, 255, (150, 150), dtype=np.uint8)
    cv2.imwrite(os.path.join(output_dir, "sample_gray.jpg"), gray_image)
    
    # Create RGBA image
    rgba_image = np.random.randint(0, 255, (80, 80, 4), dtype=np.uint8)
    cv2.imwrite(os.path.join(output_dir, "sample_rgba.png"), 
                cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA))
    
    print(f"✅ Created sample images in {output_dir}/")
    return [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
            if f.endswith(('.jpg', '.png'))]


def demo_single_image_preprocessing():
    """Demonstrate single image preprocessing."""
    print("\n🔍 Single Image Preprocessing Demo")
    print("=" * 50)
    
    # Create a sample image
    sample_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    
    print(f"Original image shape: {sample_image.shape}")
    print(f"Original image dtype: {sample_image.dtype}")
    print(f"Original image range: [{sample_image.min()}, {sample_image.max()}]")
    
    # Preprocess the image
    processed = preprocess_single(
        sample_image,
        target_size=(224, 224),
        normalize=True,
        target_format='RGB'
    )
    
    print(f"\nProcessed image shape: {processed.shape}")
    print(f"Processed image dtype: {processed.dtype}")
    print(f"Processed image range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    return processed


def demo_batch_preprocessing():
    """Demonstrate batch preprocessing."""
    print("\n📦 Batch Preprocessing Demo")
    print("=" * 50)
    
    # Create sample images
    sample_images = create_sample_images()
    
    if not sample_images:
        print("❌ No sample images found")
        return
    
    print(f"Processing {len(sample_images)} images...")
    
    # Process batch
    try:
        processed_images = preprocess_batch(
            sample_images,
            target_size=(224, 224),
            normalize=True,
            target_format='RGB'
        )
        
        print(f"✅ Successfully processed {len(processed_images)} images")
        
        for i, img in enumerate(processed_images):
            print(f"  Image {i+1}: shape={img.shape}, dtype={img.dtype}, "
                  f"range=[{img.min():.3f}, {img.max():.3f}]")
                  
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")


def demo_image_validation():
    """Demonstrate image validation."""
    print("\n✅ Image Validation Demo")
    print("=" * 50)
    
    # Create sample images
    sample_images = create_sample_images()
    
    for image_path in sample_images:
        print(f"\nValidating: {os.path.basename(image_path)}")
        validation_result = validate_image(image_path)
        
        print(f"  Valid: {validation_result['valid']}")
        print(f"  Format: {validation_result['format']}")
        print(f"  Size: {validation_result['size_bytes']} bytes")
        print(f"  Dimensions: {validation_result['dimensions']}")
        print(f"  Channels: {validation_result['channels']}")
        
        if validation_result['error']:
            print(f"  Error: {validation_result['error']}")


def demo_image_info():
    """Demonstrate image information extraction."""
    print("\n📊 Image Information Demo")
    print("=" * 50)
    
    # Create different types of images
    images = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),  # RGB array
        np.random.randint(0, 255, (50, 50), dtype=np.uint8),       # Grayscale array
        Image.fromarray(np.random.randint(0, 255, (75, 75, 3), dtype=np.uint8))  # PIL Image
    ]
    
    for i, img in enumerate(images):
        print(f"\nImage {i+1}:")
        info = get_image_info(img)
        
        print(f"  Type: {info['type']}")
        print(f"  Shape: {info['shape']}")
        print(f"  Dtype: {info['dtype']}")
        print(f"  Channels: {info['channels']}")
        if info.get('format'):
            print(f"  Format: {info['format']}")
        if info.get('size_bytes'):
            print(f"  Size: {info['size_bytes']} bytes")


def demo_preprocessor_class():
    """Demonstrate ImagePreprocessor class usage."""
    print("\n🔧 ImagePreprocessor Class Demo")
    print("=" * 50)
    
    # Create preprocessor with custom settings
    preprocessor = ImagePreprocessor(
        target_size=(128, 128),
        normalize=True,
        target_format='RGB',
        interpolation=cv2.INTER_CUBIC
    )
    
    print(f"Target size: {preprocessor.target_size}")
    print(f"Normalize: {preprocessor.normalize}")
    print(f"Target format: {preprocessor.target_format}")
    print(f"Max size: {preprocessor.max_size_bytes / 1024 / 1024:.1f} MB")
    
    # Process different image types
    test_images = [
        np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8),  # RGB
        np.random.randint(0, 255, (150, 150), dtype=np.uint8),     # Grayscale
        np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)   # RGBA
    ]
    
    for i, img in enumerate(test_images):
        print(f"\nProcessing image {i+1} (shape: {img.shape}):")
        try:
            result = preprocessor.preprocess(img)
            print(f"  Result shape: {result.shape}")
            print(f"  Result dtype: {result.dtype}")
            print(f"  Result range: [{result.min():.3f}, {result.max():.3f}]")
        except Exception as e:
            print(f"  Error: {e}")


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n⚠️  Error Handling Demo")
    print("=" * 50)
    
    # Test various error conditions
    error_cases = [
        ("Nonexistent file", "nonexistent.jpg"),
        ("Invalid format", "test.txt"),
        ("Invalid array", np.array([1, 2, 3])),  # 1D array
        ("Empty array", np.array([])),
        ("Invalid type", "not_an_image")
    ]
    
    for case_name, test_input in error_cases:
        print(f"\nTesting: {case_name}")
        try:
            if isinstance(test_input, str) and test_input.endswith(('.jpg', '.txt')):
                # File validation
                result = validate_image(test_input)
                print(f"  Valid: {result['valid']}")
                if result['error']:
                    print(f"  Error: {result['error']}")
            else:
                # Image processing
                result = preprocess_single(test_input)
                print(f"  Success: shape={result.shape}")
        except Exception as e:
            print(f"  Caught error: {type(e).__name__}: {e}")


def main():
    """Run all demonstrations."""
    print("🚀 Image Preprocessing Pipeline Demo")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_single_image_preprocessing()
        demo_batch_preprocessing()
        demo_image_validation()
        demo_image_info()
        demo_preprocessor_class()
        demo_error_handling()
        
        print("\n✅ All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Run tests: pytest tests/test_preprocessing.py -v")
        print("2. Check the sample_images/ directory for generated test images")
        print("3. Integrate preprocessing into your training pipeline")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())