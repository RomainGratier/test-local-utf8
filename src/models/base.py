"""
Base model class for image classification.

This module provides the abstract base class that all model implementations
must inherit from, ensuring consistent interface across different model types.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import tensorflow as tf
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all image classification models.
    
    This class defines the interface that all model implementations must follow,
    ensuring consistency across different model types and architectures.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 3,
        model_name: str = "base_model"
    ):
        """
        Initialize the base model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            model_name: Name identifier for the model
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.is_compiled = False
        self.training_history = None
        
    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        """
        Build the model architecture.
        
        Returns:
            Compiled TensorFlow model
        """
        pass
    
    @abstractmethod
    def compile_model(
        self,
        optimizer: str = 'adam',
        loss: str = 'categorical_crossentropy',
        metrics: List[str] = None
    ) -> None:
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            optimizer: Optimizer name or instance
            loss: Loss function name or instance
            metrics: List of metric names or instances
        """
        pass
    
    def train(
        self,
        train_data: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset] = None,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: List[tf.keras.callbacks.Callback] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train the model on the provided dataset.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Training history object
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if not self.is_compiled:
            raise ValueError("Model not compiled. Call compile_model() first.")
        
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        # Set up callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        # Train the model
        self.training_history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("Training completed successfully")
        return self.training_history
    
    def evaluate(
        self,
        test_data: tf.data.Dataset,
        verbose: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test dataset
            verbose: Verbosity level
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        logger.info("Evaluating model on test data")
        
        results = self.model.evaluate(test_data, verbose=verbose)
        
        # Get metric names
        metric_names = self.model.metrics_names
        evaluation_results = dict(zip(metric_names, results))
        
        logger.info(f"Evaluation results: {evaluation_results}")
        return evaluation_results
    
    def predict(
        self,
        data: np.ndarray,
        batch_size: int = 32,
        verbose: int = 0
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input data array
            batch_size: Batch size for prediction
            verbose: Verbosity level
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.predict(data, batch_size=batch_size, verbose=verbose)
    
    def predict_single(
        self,
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction on a single image.
        
        Args:
            image: Single image array
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.predict(image)
        
        # Get predicted class
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        result = {
            'predicted_class': int(predicted_class),
            'confidence': confidence
        }
        
        if return_confidence:
            result['class_probabilities'] = predictions[0].tolist()
        
        return result
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = tf.keras.models.load_model(filepath)
        self.is_compiled = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """
        Get a string summary of the model architecture.
        
        Returns:
            Model summary as string
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Capture model summary
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            self.model.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return summary
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {
                'model_name': self.model_name,
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'is_built': False,
                'is_compiled': False,
                'total_params': 0
            }
        
        return {
            'model_name': self.model_name,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'is_built': True,
            'is_compiled': self.is_compiled,
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        }
    
    def _get_default_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Get default training callbacks.
        
        Returns:
            List of default callbacks
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(input_shape={self.input_shape}, num_classes={self.num_classes})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}(input_shape={self.input_shape}, num_classes={self.num_classes}, model_name='{self.model_name}')"
