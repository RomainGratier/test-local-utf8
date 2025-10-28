"""
CNN model implementation for image classification.

This module provides a convolutional neural network implementation
specifically designed for image classification tasks.
"""

import logging
from typing import List, Dict, Any, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from .base import BaseModel

logger = logging.getLogger(__name__)

class CNNModel(BaseModel):
    """
    Convolutional Neural Network for image classification.
    
    This class implements a CNN architecture optimized for image classification
    with configurable depth, filters, and regularization options.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 3,
        model_name: str = "cnn_model",
        architecture: str = "standard",
        dropout_rate: float = 0.5,
        l2_reg: float = 1e-4,
        use_batch_norm: bool = True,
        use_data_augmentation: bool = False
    ):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            model_name: Name identifier for the model
            architecture: Architecture type ('standard', 'deep', 'light')
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
            use_batch_norm: Whether to use batch normalization
            use_data_augmentation: Whether to include data augmentation layers
        """
        super().__init__(input_shape, num_classes, model_name)
        
        self.architecture = architecture
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        self.use_data_augmentation = use_data_augmentation
        
    def build_model(self) -> tf.keras.Model:
        """
        Build the CNN model architecture.
        
        Returns:
            Compiled TensorFlow model
        """
        logger.info(f"Building CNN model with architecture: {self.architecture}")
        
        # Input layer
        inputs = tf.keras.Input(shape=self.input_shape, name='input_image')
        x = inputs
        
        # Data augmentation (optional)
        if self.use_data_augmentation:
            x = self._add_data_augmentation_layers(x)
        
        # Build architecture based on type
        if self.architecture == "standard":
            x = self._build_standard_architecture(x)
        elif self.architecture == "deep":
            x = self._build_deep_architecture(x)
        elif self.architecture == "light":
            x = self._build_light_architecture(x)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Dense layers
        x = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='dense_1'
        )(x)
        
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_1')(x)
        
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        
        x = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='dense_2'
        )(x)
        
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_2')(x)
        
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name=self.model_name)
        self.model = model
        
        logger.info("CNN model built successfully")
        return model
    
    def _add_data_augmentation_layers(self, x):
        """Add data augmentation layers."""
        x = layers.RandomFlip("horizontal", name='random_flip')(x)
        x = layers.RandomRotation(0.1, name='random_rotation')(x)
        x = layers.RandomZoom(0.1, name='random_zoom')(x)
        x = layers.RandomContrast(0.1, name='random_contrast')(x)
        return x
    
    def _build_standard_architecture(self, x):
        """Build standard CNN architecture."""
        # First block
        x = layers.Conv2D(32, (3, 3), activation='relu', name='conv1_1')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv1')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', name='conv1_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(0.25, name='dropout_conv1')(x)
        
        # Second block
        x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2_1')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv2')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(0.25, name='dropout_conv2')(x)
        
        # Third block
        x = layers.Conv2D(128, (3, 3), activation='relu', name='conv3_1')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv3')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', name='conv3_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = layers.Dropout(0.25, name='dropout_conv3')(x)
        
        return x
    
    def _build_deep_architecture(self, x):
        """Build deep CNN architecture."""
        # First block
        x = layers.Conv2D(32, (3, 3), activation='relu', name='conv1_1')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv1_1')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', name='conv1_2')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv1_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(0.25, name='dropout_conv1')(x)
        
        # Second block
        x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2_1')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv2_1')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2_2')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv2_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(0.25, name='dropout_conv2')(x)
        
        # Third block
        x = layers.Conv2D(128, (3, 3), activation='relu', name='conv3_1')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv3_1')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', name='conv3_2')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv3_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = layers.Dropout(0.25, name='dropout_conv3')(x)
        
        # Fourth block
        x = layers.Conv2D(256, (3, 3), activation='relu', name='conv4_1')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv4_1')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', name='conv4_2')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv4_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool4')(x)
        x = layers.Dropout(0.25, name='dropout_conv4')(x)
        
        return x
    
    def _build_light_architecture(self, x):
        """Build lightweight CNN architecture."""
        # First block
        x = layers.Conv2D(16, (3, 3), activation='relu', name='conv1')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv1')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(0.25, name='dropout_conv1')(x)
        
        # Second block
        x = layers.Conv2D(32, (3, 3), activation='relu', name='conv2')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(0.25, name='dropout_conv2')(x)
        
        # Third block
        x = layers.Conv2D(64, (3, 3), activation='relu', name='conv3')(x)
        if self.use_batch_norm:
            x = layers.BatchNormalization(name='batch_norm_conv3')(x)
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = layers.Dropout(0.25, name='dropout_conv3')(x)
        
        return x
    
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
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall']
        
        # Configure optimizer
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=0.001)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=0.001)
        else:
            opt = optimizer
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        self.is_compiled = True
        logger.info(f"Model compiled with optimizer: {optimizer}, loss: {loss}")
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """
        Get information about the CNN architecture.
        
        Returns:
            Dictionary with architecture information
        """
        return {
            'architecture_type': self.architecture,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'l2_regularization': self.l2_reg,
            'use_batch_norm': self.use_batch_norm,
            'use_data_augmentation': self.use_data_augmentation,
            'model_name': self.model_name
        }
    
    def create_data_generator(
        self,
        data_dir: str,
        batch_size: int = 32,
        validation_split: float = 0.2,
        image_size: Tuple[int, int] = (224, 224),
        shuffle: bool = True
    ) -> Tuple[tf.keras.preprocessing.image.ImageDataGenerator, tf.keras.preprocessing.image.ImageDataGenerator]:
        """
        Create data generators for training and validation.
        
        Args:
            data_dir: Directory containing training data
            batch_size: Batch size for data generation
            validation_split: Fraction of data to use for validation
            image_size: Target image size
            shuffle: Whether to shuffle the data
            
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=validation_split
        )
        
        # No augmentation for validation
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=shuffle
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=shuffle
        )
        
        return train_generator, validation_generator
