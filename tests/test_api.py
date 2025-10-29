"""
API endpoint tests for computer vision image classification.

This module tests all API endpoints including image upload,
classification, batch processing, and model management.
"""

import os
import sys
import pytest
import tempfile
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image
import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from src.api.routes import router
from src.models.cnn import CNNModel
from src.preprocessing.transforms import ImagePreprocessor


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes for testing."""
        # Create a simple test image
        image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Convert to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()
    
    @pytest.fixture
    def sample_image_file(self):
        """Create a temporary image file for testing."""
        # Create a simple test image
        image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            return tmp.name
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
    
    def test_model_info_without_model(self, client):
        """Test model info endpoint when no model is loaded."""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 503
        
        data = response.json()
        assert "detail" in data
        assert "Model not loaded" in data["detail"]
    
    def test_classify_without_model(self, client, sample_image_bytes):
        """Test classification endpoint when no model is loaded."""
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        data = {"return_confidence": True}
        
        response = client.post("/api/v1/classify", files=files, data=data)
        assert response.status_code == 503
        
        data = response.json()
        assert "detail" in data
        assert "Model not loaded" in data["detail"]
    
    def test_batch_classify_without_model(self, client, sample_image_bytes):
        """Test batch classification endpoint when no model is loaded."""
        files = [
            ("images", ("test1.jpg", sample_image_bytes, "image/jpeg")),
            ("images", ("test2.jpg", sample_image_bytes, "image/jpeg"))
        ]
        data = {"return_confidence": True}
        
        response = client.post("/api/v1/batch_classify", files=files, data=data)
        assert response.status_code == 503
        
        data = response.json()
        assert "detail" in data
        assert "Model not loaded" in data["detail"]
    
    def test_classify_invalid_image(self, client):
        """Test classification with invalid image data."""
        # Create invalid image data
        invalid_data = b"not an image"
        files = {"image": ("test.txt", invalid_data, "text/plain")}
        data = {"return_confidence": True}
        
        response = client.post("/api/v1/classify", files=files, data=data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Invalid image" in data["detail"]
    
    def test_batch_classify_too_many_images(self, client, sample_image_bytes):
        """Test batch classification with too many images."""
        # Create 11 images (limit is 10)
        files = []
        for i in range(11):
            files.append(("images", (f"test{i}.jpg", sample_image_bytes, "image/jpeg")))
        
        data = {"return_confidence": True}
        
        response = client.post("/api/v1/batch_classify", files=files, data=data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Maximum 10 images allowed per batch" in data["detail"]
    
    def test_load_model_endpoint(self, client):
        """Test model loading endpoint."""
        # This will fail because the model file doesn't exist
        response = client.post("/api/v1/model/load", json={"model_path": "nonexistent_model.h5"})
        assert response.status_code == 500
        
        data = response.json()
        assert "detail" in data


class TestImageValidation:
    """Test cases for image validation functionality."""
    
    def test_validate_image_from_bytes_valid(self):
        """Test validation of valid image bytes."""
        from src.preprocessing.transforms import validate_image_from_bytes
        
        # Create valid image bytes
        image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
        
        result = validate_image_from_bytes(image_bytes)
        
        assert result['valid'] is True
        assert result['shape'] == (100, 100, 3)
        assert result['channels'] == 3
        assert 'error' not in result
    
    def test_validate_image_from_bytes_invalid(self):
        """Test validation of invalid image bytes."""
        from src.preprocessing.transforms import validate_image_from_bytes
        
        # Test with invalid data
        invalid_bytes = b"not an image"
        result = validate_image_from_bytes(invalid_bytes)
        
        assert result['valid'] is False
        assert 'error' in result
        assert "Failed to load image" in result['error']
    
    def test_validate_image_from_bytes_empty(self):
        """Test validation of empty image bytes."""
        from src.preprocessing.transforms import validate_image_from_bytes
        
        # Test with empty bytes
        empty_bytes = b""
        result = validate_image_from_bytes(empty_bytes)
        
        assert result['valid'] is False
        assert 'error' in result
        assert "Failed to load image" in result['error']


class TestImagePreprocessing:
    """Test cases for image preprocessing functionality."""
    
    def test_preprocess_from_bytes(self):
        """Test preprocessing image from bytes."""
        from src.preprocessing.transforms import preprocess_from_bytes
        
        # Create test image
        image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
        
        # Preprocess image
        result = preprocess_from_bytes(image_bytes)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0
    
    def test_preprocess_from_bytes_invalid(self):
        """Test preprocessing invalid image bytes."""
        from src.preprocessing.transforms import preprocess_from_bytes
        
        # Test with invalid data
        invalid_bytes = b"not an image"
        
        with pytest.raises(ValueError, match="Failed to preprocess image from bytes"):
            preprocess_from_bytes(invalid_bytes)
    
    def test_image_preprocessor_from_bytes(self):
        """Test ImagePreprocessor.preprocess_from_bytes method."""
        preprocessor = ImagePreprocessor(target_size=(128, 128), normalize=True)
        
        # Create test image
        image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
        
        # Preprocess image
        result = preprocessor.preprocess_from_bytes(image_bytes)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_global_exception_handler(self, client):
        """Test global exception handler."""
        # This should trigger a 500 error
        response = client.get("/api/v1/nonexistent_endpoint")
        assert response.status_code == 404
    
    def test_http_exception_handler(self, client):
        """Test HTTP exception handler."""
        # Test with invalid model path
        response = client.post("/api/v1/model/load", json={"model_path": "invalid_path"})
        assert response.status_code == 500
        
        data = response.json()
        assert "error" in data
        assert "error_code" in data
        assert "timestamp" in data


class TestModelIntegration:
    """Test cases for model integration."""
    
    def test_cnn_model_creation(self):
        """Test CNN model creation."""
        model = CNNModel(
            input_shape=(224, 224, 3),
            num_classes=3,
            architecture="standard"
        )
        
        assert model.input_shape == (224, 224, 3)
        assert model.num_classes == 3
        assert model.architecture == "standard"
        assert model.model is None  # Not built yet
    
    def test_cnn_model_build(self):
        """Test CNN model building."""
        model = CNNModel(
            input_shape=(224, 224, 3),
            num_classes=3,
            architecture="standard"
        )
        
        # Build model
        built_model = model.build_model()
        
        assert built_model is not None
        assert model.model is not None
        assert model.model.input_shape == (None, 224, 224, 3)
        assert model.model.output_shape == (None, 3)
    
    def test_cnn_model_compile(self):
        """Test CNN model compilation."""
        model = CNNModel(
            input_shape=(224, 224, 3),
            num_classes=3,
            architecture="standard"
        )
        
        # Build and compile model
        model.build_model()
        model.compile_model()
        
        assert model.is_compiled is True
        assert model.model.optimizer is not None
        assert model.model.loss is not None
        assert len(model.model.metrics) > 0


class TestDataValidation:
    """Test cases for data validation."""
    
    def test_supported_image_formats(self):
        """Test supported image format validation."""
        from src.preprocessing.transforms import SUPPORTED_FORMATS
        
        expected_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        assert SUPPORTED_FORMATS == expected_formats
    
    def test_max_image_size(self):
        """Test maximum image size constant."""
        from src.preprocessing.transforms import MAX_IMAGE_SIZE
        
        expected_size = 10 * 1024 * 1024  # 10MB
        assert MAX_IMAGE_SIZE == expected_size
    
    def test_image_preprocessor_validation(self):
        """Test ImagePreprocessor validation."""
        # Test valid parameters
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            normalize=True,
            target_format='RGB'
        )
        
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.normalize is True
        assert preprocessor.target_format == 'RGB'
        
        # Test invalid target size
        with pytest.raises(ValueError):
            ImagePreprocessor(target_size=(10, 10))  # Too small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])