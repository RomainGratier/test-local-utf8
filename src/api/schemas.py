"""
Pydantic schemas for API request and response models.

This module defines the data models used for API communication,
including request/response schemas and validation rules.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
import numpy as np

class ClassificationRequest(BaseModel):
    """Request schema for single image classification."""
    
    image: bytes = Field(..., description="Image file as bytes")
    return_confidence: bool = Field(True, description="Whether to return confidence scores")
    
    class Config:
        arbitrary_types_allowed = True

class ClassificationResponse(BaseModel):
    """Response schema for single image classification."""
    
    predicted_class: int = Field(..., description="Predicted class index")
    predicted_class_name: Optional[str] = Field(None, description="Predicted class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    class_probabilities: Optional[List[float]] = Field(None, description="All class probabilities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist()
        }

class BatchClassificationRequest(BaseModel):
    """Request schema for batch image classification."""
    
    images: List[bytes] = Field(..., min_items=1, max_items=10, description="List of image files as bytes")
    return_confidence: bool = Field(True, description="Whether to return confidence scores")
    
    class Config:
        arbitrary_types_allowed = True

class BatchClassificationResponse(BaseModel):
    """Response schema for batch image classification."""
    
    results: List[ClassificationResponse] = Field(..., description="List of classification results")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    success_count: int = Field(..., description="Number of successfully processed images")
    error_count: int = Field(..., description="Number of failed images")
    errors: Optional[List[str]] = Field(None, description="List of error messages for failed images")

class ModelInfo(BaseModel):
    """Schema for model information."""
    
    model_name: str = Field(..., description="Model name")
    input_shape: List[int] = Field(..., description="Input image shape")
    num_classes: int = Field(..., description="Number of output classes")
    is_built: bool = Field(..., description="Whether model is built")
    is_compiled: bool = Field(..., description="Whether model is compiled")
    total_params: int = Field(..., description="Total number of parameters")
    trainable_params: Optional[int] = Field(None, description="Number of trainable parameters")
    non_trainable_params: Optional[int] = Field(None, description="Number of non-trainable parameters")
    architecture_info: Optional[Dict[str, Any]] = Field(None, description="Architecture-specific information")

class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")

class TrainingRequest(BaseModel):
    """Request schema for model training."""
    
    data_dir: str = Field(..., description="Path to training data directory")
    epochs: int = Field(50, ge=1, le=1000, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, le=256, description="Batch size for training")
    learning_rate: float = Field(0.001, ge=1e-6, le=1.0, description="Learning rate")
    validation_split: float = Field(0.2, ge=0.0, le=0.5, description="Validation split ratio")
    architecture: str = Field("standard", description="Model architecture type")
    use_data_augmentation: bool = Field(True, description="Whether to use data augmentation")
    
    @validator('architecture')
    def validate_architecture(cls, v):
        allowed_architectures = ['standard', 'deep', 'light']
        if v not in allowed_architectures:
            raise ValueError(f'Architecture must be one of: {allowed_architectures}')
        return v

class TrainingResponse(BaseModel):
    """Response schema for training results."""
    
    training_id: str = Field(..., description="Unique training session ID")
    status: str = Field(..., description="Training status")
    model_path: Optional[str] = Field(None, description="Path to saved model")
    training_time_seconds: float = Field(..., description="Total training time")
    final_accuracy: Optional[float] = Field(None, description="Final training accuracy")
    final_val_accuracy: Optional[float] = Field(None, description="Final validation accuracy")
    training_history: Optional[Dict[str, List[float]]] = Field(None, description="Training metrics history")

class EvaluationRequest(BaseModel):
    """Request schema for model evaluation."""
    
    model_path: str = Field(..., description="Path to model file")
    test_data_dir: str = Field(..., description="Path to test data directory")
    batch_size: int = Field(32, ge=1, le=256, description="Batch size for evaluation")

class EvaluationResponse(BaseModel):
    """Response schema for evaluation results."""
    
    accuracy: float = Field(..., description="Test accuracy")
    precision: float = Field(..., description="Test precision")
    recall: float = Field(..., description="Test recall")
    f1_score: float = Field(..., description="Test F1 score")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    class_metrics: Dict[str, Dict[str, float]] = Field(..., description="Per-class metrics")
    evaluation_time_seconds: float = Field(..., description="Evaluation time")

class ClassMapping(BaseModel):
    """Schema for class name mapping."""
    
    class_index: int = Field(..., ge=0, description="Class index")
    class_name: str = Field(..., description="Class name")
    description: Optional[str] = Field(None, description="Class description")

class ModelConfig(BaseModel):
    """Schema for model configuration."""
    
    input_shape: List[int] = Field([224, 224, 3], description="Input image shape")
    num_classes: int = Field(3, ge=2, description="Number of classes")
    architecture: str = Field("standard", description="Model architecture")
    dropout_rate: float = Field(0.5, ge=0.0, le=1.0, description="Dropout rate")
    l2_regularization: float = Field(1e-4, ge=0.0, description="L2 regularization factor")
    use_batch_norm: bool = Field(True, description="Use batch normalization")
    use_data_augmentation: bool = Field(False, description="Use data augmentation")
    
    @validator('input_shape')
    def validate_input_shape(cls, v):
        if len(v) != 3:
            raise ValueError('Input shape must have 3 dimensions: height, width, channels')
        if v[0] < 32 or v[1] < 32:
            raise ValueError('Input dimensions must be at least 32x32')
        if v[2] not in [1, 3, 4]:
            raise ValueError('Number of channels must be 1, 3, or 4')
        return v

class PredictionConfidence(BaseModel):
    """Schema for prediction confidence information."""
    
    predicted_class: int = Field(..., description="Predicted class index")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    uncertainty: float = Field(..., ge=0.0, le=1.0, description="Prediction uncertainty")
    entropy: float = Field(..., ge=0.0, description="Prediction entropy")
    top_k_predictions: Optional[List[Dict[str, Union[int, float]]]] = Field(
        None, 
        description="Top-k predictions with confidence scores"
    )

class ImageMetadata(BaseModel):
    """Schema for image metadata."""
    
    filename: Optional[str] = Field(None, description="Original filename")
    file_size_bytes: int = Field(..., description="File size in bytes")
    image_shape: List[int] = Field(..., description="Image dimensions")
    format: str = Field(..., description="Image format")
    channels: int = Field(..., description="Number of color channels")
    dtype: str = Field(..., description="Data type")

class ProcessingStats(BaseModel):
    """Schema for processing statistics."""
    
    total_images_processed: int = Field(..., description="Total images processed")
    successful_predictions: int = Field(..., description="Successful predictions")
    failed_predictions: int = Field(..., description="Failed predictions")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    throughput_images_per_second: float = Field(..., description="Processing throughput")
