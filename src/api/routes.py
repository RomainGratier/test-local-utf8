"""
FastAPI routes for image classification API.

This module defines all the API endpoints for the image classification system,
including image upload, classification, batch processing, and model management.
"""

import os
import time
import logging
from typing import List, Optional
from datetime import datetime
from io import BytesIO

import numpy as np
import tensorflow as tf
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from PIL import Image

from .schemas import (
    ClassificationResponse,
    BatchClassificationResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
    TrainingRequest,
    TrainingResponse,
    EvaluationRequest,
    EvaluationResponse
)
from ..models.cnn import CNNModel
from ..preprocessing.transforms import ImagePreprocessor, validate_image

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global model instance
model_instance: Optional[CNNModel] = None
model_loaded = False
start_time = time.time()

# Class names mapping (can be loaded from config)
CLASS_NAMES = {
    0: "class_0",
    1: "class_1", 
    2: "class_2"
}

def get_model() -> CNNModel:
    """Get the global model instance."""
    global model_instance
    if model_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please load a model first."
        )
    return model_instance

def load_model(model_path: str) -> None:
    """Load a model from file."""
    global model_instance, model_loaded
    
    try:
        model_instance = CNNModel()
        model_instance.load_model(model_path)
        model_loaded = True
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        model_loaded=model_loaded,
        uptime_seconds=uptime
    )

@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    model = get_model()
    model_info = model.get_model_info()
    
    # Add architecture info if available
    if hasattr(model, 'get_architecture_info'):
        model_info['architecture_info'] = model.get_architecture_info()
    
    return ModelInfo(**model_info)

@router.post("/model/load")
async def load_model_endpoint(model_path: str):
    """Load a model from file."""
    try:
        load_model(model_path)
        return {"message": f"Model loaded successfully from {model_path}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}"
        )

@router.post("/classify", response_model=ClassificationResponse)
async def classify_image(
    image: UploadFile = File(...),
    return_confidence: bool = Form(True)
):
    """Classify a single image."""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    start_time = time.time()
    
    try:
        # Read image file
        image_bytes = await image.read()
        
        # Validate image
        image_info = validate_image_from_bytes(image_bytes)
        if not image_info['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image: {image_info.get('error', 'Unknown error')}"
            )
        
        # Preprocess image
        preprocessor = ImagePreprocessor(target_size=(224, 224), normalize=True)
        processed_image = preprocessor.preprocess_from_bytes(image_bytes)
        
        # Make prediction
        model = get_model()
        prediction = model.predict_single(processed_image, return_confidence=return_confidence)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Get class name
        predicted_class_name = CLASS_NAMES.get(prediction['predicted_class'], f"class_{prediction['predicted_class']}")
        
        return ClassificationResponse(
            predicted_class=prediction['predicted_class'],
            predicted_class_name=predicted_class_name,
            confidence=prediction['confidence'],
            class_probabilities=prediction.get('class_probabilities') if return_confidence else None,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error classifying image: {str(e)}"
        )

@router.post("/batch_classify", response_model=BatchClassificationResponse)
async def batch_classify_images(
    images: List[UploadFile] = File(...),
    return_confidence: bool = Form(True)
):
    """Classify multiple images in batch."""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    if len(images) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )
    
    start_time = time.time()
    results = []
    errors = []
    
    try:
        model = get_model()
        preprocessor = ImagePreprocessor(target_size=(224, 224), normalize=True)
        
        for i, image in enumerate(images):
            try:
                # Read image file
                image_bytes = await image.read()
                
                # Validate image
                image_info = validate_image_from_bytes(image_bytes)
                if not image_info['valid']:
                    errors.append(f"Image {i+1}: {image_info.get('error', 'Invalid image')}")
                    continue
                
                # Preprocess image
                processed_image = preprocessor.preprocess_from_bytes(image_bytes)
                
                # Make prediction
                prediction = model.predict_single(processed_image, return_confidence=return_confidence)
                
                # Get class name
                predicted_class_name = CLASS_NAMES.get(prediction['predicted_class'], f"class_{prediction['predicted_class']}")
                
                results.append(ClassificationResponse(
                    predicted_class=prediction['predicted_class'],
                    predicted_class_name=predicted_class_name,
                    confidence=prediction['confidence'],
                    class_probabilities=prediction.get('class_probabilities') if return_confidence else None,
                    processing_time_ms=0  # Individual processing time not calculated for batch
                ))
                
            except Exception as e:
                error_msg = f"Image {i+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        total_processing_time = (time.time() - start_time) * 1000
        
        return BatchClassificationResponse(
            results=results,
            total_processing_time_ms=total_processing_time,
            success_count=len(results),
            error_count=len(errors),
            errors=errors if errors else None
        )
        
    except Exception as e:
        logger.error(f"Error in batch classification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch classification: {str(e)}"
        )

def validate_image_from_bytes(image_bytes: bytes) -> dict:
    """Validate image from bytes."""
    try:
        # Try to open with PIL
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Basic validation
        if image_array.size == 0:
            return {'valid': False, 'error': 'Empty image'}
        
        if len(image_array.shape) not in [2, 3]:
            return {'valid': False, 'error': f'Invalid image shape: {image_array.shape}'}
        
        if len(image_array.shape) == 3 and image_array.shape[2] not in [1, 3, 4]:
            return {'valid': False, 'error': f'Invalid number of channels: {image_array.shape[2]}'}
        
        return {
            'valid': True,
            'shape': image_array.shape,
            'dtype': str(image_array.dtype),
            'channels': image_array.shape[2] if len(image_array.shape) == 3 else 1
        }
        
    except Exception as e:
        return {'valid': False, 'error': f'Failed to load image: {str(e)}'}

# Error handlers
@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"exception": str(exc)},
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )
