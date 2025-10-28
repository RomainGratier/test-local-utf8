"""
Model definitions for computer vision image classification.

This module provides base model classes and CNN implementations
for the image classification system.
"""

from .base import BaseModel
from .cnn import CNNModel

__all__ = ['BaseModel', 'CNNModel']
