"""
API module for computer vision image classification.

This module provides FastAPI routes, schemas, and endpoints
for the image classification system.
"""

from .schemas import (
    ClassificationRequest,
    ClassificationResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse
)
from .routes import router

__all__ = [
    'ClassificationRequest',
    'ClassificationResponse', 
    'BatchClassificationRequest',
    'BatchClassificationResponse',
    'ModelInfo',
    'HealthResponse',
    'ErrorResponse',
    'router'
]
