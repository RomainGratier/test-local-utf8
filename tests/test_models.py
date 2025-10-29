"""
Model tests for computer vision image classification.

This module tests the model implementations including CNN model,
base model functionality, and model training/inference capabilities.
"""

import os
import sys
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.base import BaseModel
from src.models.cnn import CNNModel


class TestBaseModel:
    """Test cases for BaseModel abstract class."""
    
    def test_base_model_initialization(self):
        """Test BaseModel initialization."""
        # This should raise an error since BaseModel is abstract
        with pytest.raises(TypeError):
            BaseModel()
    
    def test_base_model_abstract_methods(self):
        """Test that BaseModel has required abstract methods."""
        abstract_methods = ['build_model', 'compile_model']
        
        for method_name in abstract_methods:
            assert hasattr(BaseModel, method_name)
            method = getattr(BaseModel, method_name)
            assert hasattr(method, '__isabstractmethod__')
            assert method.__isabstractmethod__


class TestCNNModel:
    """Test cases for CNNModel implementation."""
    
    def test_cnn_model_initialization_default(self):
        """Test CNNModel initialization with default parameters."""
        model = CNNModel()
        
        assert model.input_shape == (224, 224, 3)
        assert model.num_classes == 3
        assert model.model_name == "cnn_model"
        assert model.architecture == "standard"
        assert model.dropout_rate == 0.5
        assert model.l2_reg == 1e-4
        assert model.use_batch_norm is True
        assert model.use_data_augmentation is False
        assert model.model is None
        assert model.is_compiled is False
    
    def test_cnn_model_initialization_custom(self):
        """Test CNNModel initialization with custom parameters."""
        model = CNNModel(
            input_shape=(128, 128, 1),
            num_classes=5,
            model_name="custom_model",
            architecture="deep",
            dropout_rate=0.3,
            l2_reg=1e-3,
            use_batch_norm=False,
            use_data_augmentation=True
        )
        
        assert model.input_shape == (128, 128, 1)
        assert model.num_classes == 5
        assert model.model_name == "custom_model"
        assert model.architecture == "deep"
        assert model.dropout_rate == 0.3
        assert model.l2_reg == 1e-3
        assert model.use_batch_norm is False
        assert model.use_data_augmentation is True
    
    def test_cnn_model_build_standard(self):
        """Test building standard CNN architecture."""
        model = CNNModel(architecture="standard")
        built_model = model.build_model()
        
        assert built_model is not None
        assert model.model is not None
        assert built_model.input_shape == (None, 224, 224, 3)
        assert built_model.output_shape == (None, 3)
        
        # Check that model has expected layers
        layer_names = [layer.name for layer in built_model.layers]
        assert 'input_image' in layer_names
        assert 'conv1_1' in layer_names
        assert 'conv2_1' in layer_names
        assert 'conv3_1' in layer_names
        assert 'global_avg_pool' in layer_names
        assert 'dense_1' in layer_names
        assert 'dense_2' in layer_names
        assert 'predictions' in layer_names
    
    def test_cnn_model_build_deep(self):
        """Test building deep CNN architecture."""
        model = CNNModel(architecture="deep")
        built_model = model.build_model()
        
        assert built_model is not None
        assert model.model is not None
        
        # Check that deep architecture has more layers
        layer_names = [layer.name for layer in built_model.layers]
        assert 'conv4_1' in layer_names  # Deep architecture has 4th conv block
    
    def test_cnn_model_build_light(self):
        """Test building light CNN architecture."""
        model = CNNModel(architecture="light")
        built_model = model.build_model()
        
        assert built_model is not None
        assert model.model is not None
        
        # Check that light architecture has fewer layers
        layer_names = [layer.name for layer in built_model.layers]
        assert 'conv1' in layer_names  # Light architecture uses single conv layers
        assert 'conv2' in layer_names
        assert 'conv3' in layer_names
    
    def test_cnn_model_build_invalid_architecture(self):
        """Test building with invalid architecture."""
        model = CNNModel(architecture="invalid")
        
        with pytest.raises(ValueError, match="Unknown architecture"):
            model.build_model()
    
    def test_cnn_model_compile_default(self):
        """Test model compilation with default parameters."""
        model = CNNModel()
        model.build_model()
        model.compile_model()
        
        assert model.is_compiled is True
        assert model.model.optimizer is not None
        assert model.model.loss is not None
        assert len(model.model.metrics) > 0
    
    def test_cnn_model_compile_custom(self):
        """Test model compilation with custom parameters."""
        model = CNNModel()
        model.build_model()
        model.compile_model(
            optimizer='sgd',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        assert model.is_compiled is True
        assert isinstance(model.model.optimizer, tf.keras.optimizers.SGD)
        assert model.model.loss == 'sparse_categorical_crossentropy'
        assert len(model.model.metrics) == 1
    
    def test_cnn_model_compile_without_build(self):
        """Test compilation without building model first."""
        model = CNNModel()
        
        with pytest.raises(ValueError, match="Model not built"):
            model.compile_model()
    
    def test_cnn_model_predict_single(self):
        """Test single image prediction."""
        model = CNNModel()
        model.build_model()
        model.compile_model()
        
        # Create test image
        test_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Make prediction
        result = model.predict_single(test_image)
        
        assert isinstance(result, dict)
        assert 'predicted_class' in result
        assert 'confidence' in result
        assert isinstance(result['predicted_class'], int)
        assert isinstance(result['confidence'], float)
        assert 0 <= result['predicted_class'] < 3
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_cnn_model_predict_single_with_confidence(self):
        """Test single image prediction with confidence scores."""
        model = CNNModel()
        model.build_model()
        model.compile_model()
        
        # Create test image
        test_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Make prediction with confidence
        result = model.predict_single(test_image, return_confidence=True)
        
        assert 'class_probabilities' in result
        assert isinstance(result['class_probabilities'], list)
        assert len(result['class_probabilities']) == 3
        assert all(0.0 <= prob <= 1.0 for prob in result['class_probabilities'])
        assert abs(sum(result['class_probabilities']) - 1.0) < 1e-6  # Should sum to 1
    
    def test_cnn_model_predict_batch(self):
        """Test batch prediction."""
        model = CNNModel()
        model.build_model()
        model.compile_model()
        
        # Create test images
        test_images = np.random.rand(5, 224, 224, 3).astype(np.float32)
        
        # Make predictions
        predictions = model.predict(test_images)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (5, 3)
        assert predictions.dtype == np.float32
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-6)  # Probabilities should sum to 1
    
    def test_cnn_model_predict_without_build(self):
        """Test prediction without building model."""
        model = CNNModel()
        test_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="Model not built"):
            model.predict_single(test_image)
    
    def test_cnn_model_save_load(self, tmp_path):
        """Test model saving and loading."""
        model = CNNModel()
        model.build_model()
        model.compile_model()
        
        # Save model
        model_path = tmp_path / "test_model.h5"
        model.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Create new model and load
        new_model = CNNModel()
        new_model.load_model(str(model_path))
        
        assert new_model.model is not None
        assert new_model.is_compiled is True
        assert new_model.model.input_shape == model.model.input_shape
        assert new_model.model.output_shape == model.model.output_shape
    
    def test_cnn_model_save_without_build(self, tmp_path):
        """Test saving model without building."""
        model = CNNModel()
        model_path = tmp_path / "test_model.h5"
        
        with pytest.raises(ValueError, match="Model not built"):
            model.save_model(str(model_path))
    
    def test_cnn_model_load_nonexistent(self):
        """Test loading nonexistent model."""
        model = CNNModel()
        
        with pytest.raises(FileNotFoundError):
            model.load_model("nonexistent_model.h5")
    
    def test_cnn_model_get_model_info(self):
        """Test getting model information."""
        model = CNNModel()
        
        # Test before building
        info = model.get_model_info()
        assert info['model_name'] == "cnn_model"
        assert info['input_shape'] == (224, 224, 3)
        assert info['num_classes'] == 3
        assert info['is_built'] is False
        assert info['is_compiled'] is False
        assert info['total_params'] == 0
        
        # Test after building
        model.build_model()
        info = model.get_model_info()
        assert info['is_built'] is True
        assert info['total_params'] > 0
        assert 'trainable_params' in info
        assert 'non_trainable_params' in info
    
    def test_cnn_model_get_architecture_info(self):
        """Test getting architecture information."""
        model = CNNModel(
            architecture="deep",
            dropout_rate=0.3,
            l2_reg=1e-3,
            use_batch_norm=False,
            use_data_augmentation=True
        )
        
        info = model.get_architecture_info()
        
        assert info['architecture_type'] == "deep"
        assert info['input_shape'] == (224, 224, 3)
        assert info['num_classes'] == 3
        assert info['dropout_rate'] == 0.3
        assert info['l2_regularization'] == 1e-3
        assert info['use_batch_norm'] is False
        assert info['use_data_augmentation'] is True
        assert info['model_name'] == "cnn_model"
    
    def test_cnn_model_data_generator(self, tmp_path):
        """Test data generator creation."""
        model = CNNModel()
        
        # Create test data directory structure
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        
        # Create class directories
        for i in range(3):
            class_dir = data_dir / f"class_{i}"
            class_dir.mkdir()
            
            # Create dummy image files
            for j in range(5):
                img_path = class_dir / f"image_{j}.jpg"
                # Create a simple test image
                test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                import cv2
                cv2.imwrite(str(img_path), cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        
        # Create data generators
        train_gen, val_gen = model.create_data_generator(
            data_dir=str(data_dir),
            batch_size=2,
            validation_split=0.2
        )
        
        assert train_gen is not None
        assert val_gen is not None
        
        # Test generator
        batch_x, batch_y = next(iter(train_gen))
        assert batch_x.shape[0] == 2  # batch_size
        assert batch_x.shape[1:] == (224, 224, 3)  # input shape
        assert batch_y.shape[0] == 2  # batch_size
        assert batch_y.shape[1] == 3  # num_classes
    
    def test_cnn_model_string_representation(self):
        """Test string representation of model."""
        model = CNNModel(input_shape=(128, 128, 1), num_classes=5)
        
        str_repr = str(model)
        assert "CNNModel" in str_repr
        assert "(128, 128, 1)" in str_repr
        assert "5" in str_repr
        
        repr_str = repr(model)
        assert "CNNModel" in repr_str
        assert "input_shape=(128, 128, 1)" in repr_str
        assert "num_classes=5" in repr_str
        assert "model_name='cnn_model'" in repr_str


class TestModelTraining:
    """Test cases for model training functionality."""
    
    def test_model_training_without_build(self):
        """Test training without building model."""
        model = CNNModel()
        
        # Create dummy data
        train_data = tf.data.Dataset.from_tensor_slices((
            np.random.rand(10, 224, 224, 3).astype(np.float32),
            np.random.randint(0, 3, 10)
        )).batch(2)
        
        with pytest.raises(ValueError, match="Model not built"):
            model.train(train_data, epochs=1)
    
    def test_model_training_without_compile(self):
        """Test training without compiling model."""
        model = CNNModel()
        model.build_model()
        
        # Create dummy data
        train_data = tf.data.Dataset.from_tensor_slices((
            np.random.rand(10, 224, 224, 3).astype(np.float32),
            tf.keras.utils.to_categorical(np.random.randint(0, 3, 10), 3)
        )).batch(2)
        
        with pytest.raises(ValueError, match="Model not compiled"):
            model.train(train_data, epochs=1)
    
    def test_model_evaluation_without_build(self):
        """Test evaluation without building model."""
        model = CNNModel()
        
        # Create dummy data
        test_data = tf.data.Dataset.from_tensor_slices((
            np.random.rand(10, 224, 224, 3).astype(np.float32),
            tf.keras.utils.to_categorical(np.random.randint(0, 3, 10), 3)
        )).batch(2)
        
        with pytest.raises(ValueError, match="Model not built"):
            model.evaluate(test_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])