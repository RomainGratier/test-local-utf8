"""
Configuration settings for image preprocessing pipeline.

This module contains default configuration values and settings for
the image preprocessing functionality.
"""

from typing import Tuple, Set
import cv2

# Image processing defaults
DEFAULT_TARGET_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_NORMALIZE: bool = True
DEFAULT_TARGET_FORMAT: str = 'RGB'
DEFAULT_INTERPOLATION: int = cv2.INTER_LINEAR

# File size limits
MAX_IMAGE_SIZE_MB: int = 10
MAX_IMAGE_SIZE_BYTES: int = MAX_IMAGE_SIZE_MB * 1024 * 1024

# Supported image formats
SUPPORTED_FORMATS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Supported color formats
SUPPORTED_COLOR_FORMATS: Set[str] = {'RGB', 'BGR', 'GRAY'}

# Interpolation methods
INTERPOLATION_METHODS = {
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC,
    'nearest': cv2.INTER_NEAREST,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

# Preprocessing presets
PREPROCESSING_PRESETS = {
    'imagenet': {
        'target_size': (224, 224),
        'normalize': True,
        'target_format': 'RGB',
        'interpolation': cv2.INTER_LINEAR
    },
    'mobilenet': {
        'target_size': (224, 224),
        'normalize': True,
        'target_format': 'RGB',
        'interpolation': cv2.INTER_LINEAR
    },
    'efficientnet': {
        'target_size': (224, 224),
        'normalize': True,
        'target_format': 'RGB',
        'interpolation': cv2.INTER_LINEAR
    },
    'resnet': {
        'target_size': (224, 224),
        'normalize': True,
        'target_format': 'RGB',
        'interpolation': cv2.INTER_LINEAR
    },
    'custom_small': {
        'target_size': (128, 128),
        'normalize': True,
        'target_format': 'RGB',
        'interpolation': cv2.INTER_CUBIC
    },
    'custom_large': {
        'target_size': (512, 512),
        'normalize': True,
        'target_format': 'RGB',
        'interpolation': cv2.INTER_LINEAR
    },
    'grayscale': {
        'target_size': (224, 224),
        'normalize': True,
        'target_format': 'GRAY',
        'interpolation': cv2.INTER_LINEAR
    }
}

# Validation settings
VALIDATION_SETTINGS = {
    'min_width': 1,
    'min_height': 1,
    'max_width': 10000,
    'max_height': 10000,
    'min_channels': 1,
    'max_channels': 4,
    'allowed_dtypes': ['uint8', 'uint16', 'float32', 'float64']
}

# Error messages
ERROR_MESSAGES = {
    'file_not_found': "Image file not found: {path}",
    'file_too_large': "Image too large: {size:.1f}MB > {max_size}MB",
    'unsupported_format': "Unsupported image format: {format}. Supported: {supported}",
    'load_failed': "Failed to load image {path}: {error}",
    'invalid_type': "Unsupported image type: {type}",
    'empty_image': "Image is empty",
    'invalid_shape': "Invalid image shape: {shape}. Expected 2D or 3D array",
    'invalid_channels': "Invalid number of channels: {channels}. Expected 1, 3, or 4",
    'resize_failed': "Failed to resize image: {error}",
    'format_conversion_failed': "Failed to convert image format: {error}",
    'validation_failed': "Image validation failed: {error}"
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': ['console', 'file']
}

# Performance settings
PERFORMANCE_SETTINGS = {
    'batch_size': 32,
    'num_workers': 4,
    'prefetch_factor': 2,
    'pin_memory': True
}