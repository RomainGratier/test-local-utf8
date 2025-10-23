"""
Unit tests for image preprocessing pipeline.

This module tests all preprocessing functionality including resize,
normalization, format conversion, and batch processing.
"""

import os
import tempfile
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import cv2

from src.preprocessing.transforms import (
    ImagePreprocessor,
    resize_image,
    normalize_image,
    convert_format,
    preprocess_single,
    preprocess_batch,
    validate_image,
    get_image_info,
    SUPPORTED_FORMATS,
    MAX_IMAGE_SIZE
)


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            normalize=True,
            target_format='RGB'
        )
        
        # Create test images
        self.test_image_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_image_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.test_image_rgba = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    
    def test_init_default_params(self):
        """Test ImagePreprocessor initialization with default parameters."""
        preprocessor = ImagePreprocessor()
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.normalize is True
        assert preprocessor.target_format == 'RGB'
        assert preprocessor.interpolation == cv2.INTER_LINEAR
        assert preprocessor.max_size_bytes == 10 * 1024 * 1024
    
    def test_init_custom_params(self):
        """Test ImagePreprocessor initialization with custom parameters."""
        preprocessor = ImagePreprocessor(
            target_size=(128, 128),
            normalize=False,
            target_format='GRAY',
            interpolation=cv2.INTER_CUBIC,
            max_size_mb=5
        )
        assert preprocessor.target_size == (128, 128)
        assert preprocessor.normalize is False
        assert preprocessor.target_format == 'GRAY'
        assert preprocessor.interpolation == cv2.INTER_CUBIC
        assert preprocessor.max_size_bytes == 5 * 1024 * 1024
    
    def test_preprocess_rgb_image(self):
        """Test preprocessing RGB image."""
        result = self.preprocessor.preprocess(self.test_image_rgb)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0
    
    def test_preprocess_grayscale_image(self):
        """Test preprocessing grayscale image."""
        result = self.preprocessor.preprocess(self.test_image_gray)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)  # Converted to RGB
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0
    
    def test_preprocess_rgba_image(self):
        """Test preprocessing RGBA image."""
        result = self.preprocessor.preprocess(self.test_image_rgba)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)  # Converted to RGB
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0
    
    def test_preprocess_without_normalization(self):
        """Test preprocessing without normalization."""
        preprocessor = ImagePreprocessor(normalize=False)
        result = preprocessor.preprocess(self.test_image_rgb)
        
        assert result.dtype == np.uint8
        assert 0 <= result.min() <= result.max() <= 255
    
    def test_preprocess_different_formats(self):
        """Test preprocessing with different target formats."""
        # Test RGB format
        preprocessor_rgb = ImagePreprocessor(target_format='RGB')
        result_rgb = preprocessor_rgb.preprocess(self.test_image_rgb)
        assert result_rgb.shape[2] == 3
        
        # Test BGR format
        preprocessor_bgr = ImagePreprocessor(target_format='BGR')
        result_bgr = preprocessor_bgr.preprocess(self.test_image_rgb)
        assert result_bgr.shape[2] == 3
        
        # Test GRAY format
        preprocessor_gray = ImagePreprocessor(target_format='GRAY')
        result_gray = preprocessor_gray.preprocess(self.test_image_rgb)
        assert len(result_gray.shape) == 2 or result_gray.shape[2] == 1
    
    def test_preprocess_invalid_input(self):
        """Test preprocessing with invalid input."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            self.preprocessor.preprocess("invalid_input")
    
    def test_validate_image_valid(self):
        """Test image validation with valid image."""
        self.preprocessor._validate_image(self.test_image_rgb)
        # Should not raise any exception
    
    def test_validate_image_invalid(self):
        """Test image validation with invalid image."""
        with pytest.raises(ValueError, match="Image must be a numpy array"):
            self.preprocessor._validate_image("not_an_array")
        
        with pytest.raises(ValueError, match="Image is empty"):
            self.preprocessor._validate_image(np.array([]))
        
        with pytest.raises(ValueError, match="Invalid image shape"):
            self.preprocessor._validate_image(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError, match="Invalid number of channels"):
            invalid_image = np.random.randint(0, 255, (100, 100, 5), dtype=np.uint8)
            self.preprocessor._validate_image(invalid_image)


class TestImageTransforms:
    """Test cases for individual transform functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_resize_image(self):
        """Test resize_image function."""
        result = resize_image(self.test_image, target_size=(50, 50))
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50, 3)
        assert result.dtype == np.uint8
    
    def test_normalize_image(self):
        """Test normalize_image function."""
        result = normalize_image(self.test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0
    
    def test_convert_format_rgb(self):
        """Test convert_format function with RGB target."""
        result = convert_format(self.test_image, target_format='RGB')
        
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3
    
    def test_convert_format_bgr(self):
        """Test convert_format function with BGR target."""
        result = convert_format(self.test_image, target_format='BGR')
        
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3
    
    def test_convert_format_gray(self):
        """Test convert_format function with GRAY target."""
        result = convert_format(self.test_image, target_format='GRAY')
        
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2 or result.shape[2] == 1
    
    def test_preprocess_single(self):
        """Test preprocess_single function."""
        result = preprocess_single(
            self.test_image,
            target_size=(64, 64),
            normalize=True,
            target_format='RGB'
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0


class TestBatchProcessing:
    """Test cases for batch processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_images = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            for _ in range(3)
        ]
    
    def test_preprocess_batch_success(self):
        """Test successful batch processing."""
        results = preprocess_batch(
            self.test_images,
            target_size=(32, 32),
            normalize=True
        )
        
        assert isinstance(results, list)
        assert len(results) == 3
        
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == (32, 32, 3)
            assert result.dtype == np.float32
    
    def test_preprocess_batch_with_errors(self):
        """Test batch processing with error handling."""
        # Mix valid and invalid images
        mixed_images = [
            self.test_images[0],  # Valid
            "invalid_image",      # Invalid
            self.test_images[1]   # Valid
        ]
        
        with pytest.raises(ValueError):
            preprocess_batch(mixed_images)
    
    def test_preprocess_batch_return_errors(self):
        """Test batch processing with error return."""
        mixed_images = [
            self.test_images[0],  # Valid
            "invalid_image",      # Invalid
            self.test_images[1]   # Valid
        ]
        
        results, errors = preprocess_batch(mixed_images, return_errors=True)
        
        assert isinstance(results, list)
        assert isinstance(errors, list)
        assert len(results) == 2  # Only valid images
        assert len(errors) == 1   # One error


class TestImageValidation:
    """Test cases for image validation functionality."""
    
    def test_validate_image_file_not_found(self):
        """Test validation with non-existent file."""
        result = validate_image("nonexistent.jpg")
        
        assert result['valid'] is False
        assert result['exists'] is False
        assert "File does not exist" in result['error']
    
    def test_validate_image_unsupported_format(self):
        """Test validation with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"not an image")
            tmp_path = tmp.name
        
        try:
            result = validate_image(tmp_path)
            
            assert result['valid'] is False
            assert result['exists'] is True
            assert "Unsupported format" in result['error']
        finally:
            os.unlink(tmp_path)
    
    def test_validate_image_too_large(self):
        """Test validation with oversized file."""
        # Create a large dummy file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Write more than MAX_IMAGE_SIZE bytes
            tmp.write(b"x" * (MAX_IMAGE_SIZE + 1))
            tmp_path = tmp.name
        
        try:
            result = validate_image(tmp_path)
            
            assert result['valid'] is False
            assert result['exists'] is True
            assert "File too large" in result['error']
        finally:
            os.unlink(tmp_path)
    
    def test_validate_image_valid(self):
        """Test validation with valid image file."""
        # Create a valid test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
            tmp_path = tmp.name
        
        try:
            result = validate_image(tmp_path)
            
            assert result['valid'] is True
            assert result['exists'] is True
            assert result['error'] is None
            assert result['format'] == '.jpg'
            assert result['size_bytes'] > 0
            assert result['dimensions'] is not None
            assert result['channels'] == 3
        finally:
            os.unlink(tmp_path)
    
    def test_get_image_info_numpy_array(self):
        """Test get_image_info with numpy array."""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        info = get_image_info(test_image)
        
        assert info['type'] == 'ndarray'
        assert info['shape'] == (100, 100, 3)
        assert info['dtype'] == 'uint8'
        assert info['channels'] == 3
        assert info['format'] is None
        assert info['size_bytes'] is None
    
    def test_get_image_info_pil_image(self):
        """Test get_image_info with PIL Image."""
        test_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        info = get_image_info(test_image)
        
        assert info['type'] == 'Image'
        assert info['shape'] == (100, 100, 3)
        assert info['channels'] == 3
    
    def test_get_image_info_file_path(self):
        """Test get_image_info with file path."""
        # Create a test image file
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
            tmp_path = tmp.name
        
        try:
            info = get_image_info(tmp_path)
            
            assert info['type'] == 'str'
            assert info['format'] == '.png'
            assert info['size_bytes'] > 0
            assert info['shape'] == (50, 50, 3)
            assert info['channels'] == 3
        finally:
            os.unlink(tmp_path)
    
    def test_get_image_info_invalid_type(self):
        """Test get_image_info with invalid type."""
        info = get_image_info(123)  # Invalid type
        
        assert 'error' in info
        assert "Unsupported image type" in info['error']


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""
    
    def test_empty_image_array(self):
        """Test handling of empty image array."""
        preprocessor = ImagePreprocessor()
        
        with pytest.raises(ValueError, match="Image is empty"):
            preprocessor._validate_image(np.array([]))
    
    def test_single_pixel_image(self):
        """Test processing single pixel image."""
        single_pixel = np.array([[[255, 0, 0]]], dtype=np.uint8)
        result = preprocess_single(single_pixel, target_size=(224, 224))
        
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32
    
    def test_very_large_image(self):
        """Test processing very large image."""
        large_image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        result = preprocess_single(large_image, target_size=(224, 224))
        
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32
    
    def test_different_dtypes(self):
        """Test processing images with different data types."""
        # Test uint8
        image_uint8 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result_uint8 = preprocess_single(image_uint8, normalize=True)
        assert result_uint8.dtype == np.float32
        assert 0.0 <= result_uint8.min() <= result_uint8.max() <= 1.0
        
        # Test uint16
        image_uint16 = np.random.randint(0, 65535, (50, 50, 3), dtype=np.uint16)
        result_uint16 = preprocess_single(image_uint16, normalize=True)
        assert result_uint16.dtype == np.float32
        assert 0.0 <= result_uint16.min() <= result_uint16.max() <= 1.0
        
        # Test float32
        image_float32 = np.random.rand(50, 50, 3).astype(np.float32)
        result_float32 = preprocess_single(image_float32, normalize=True)
        assert result_float32.dtype == np.float32
        assert 0.0 <= result_float32.min() <= result_float32.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])