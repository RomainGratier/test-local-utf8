"""
Logging configuration for the image classification system.

This module provides centralized logging setup with configurable
output destinations, formats, and levels.
"""

import os
import logging
import logging.handlers
from typing import Optional, Dict, Any
from pathlib import Path

from .config import get_config

def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    file_path: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_file_size_mb: int = 10,
    backup_count: int = 5
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Log message format string
        file_path: Path to log file (optional)
        console_output: Whether to output to console
        file_output: Whether to output to file
        max_file_size_mb: Maximum log file size in MB
        backup_count: Number of backup files to keep
    """
    # Get config if not provided
    config = get_config()
    
    # Use config values if not provided
    if level is None:
        level = config.logging.level
    if format_string is None:
        format_string = config.logging.format
    if file_path is None:
        file_path = config.logging.file_path
    if console_output is None:
        console_output = config.logging.console_output
    if file_output is None:
        file_output = config.logging.file_output
    if max_file_size_mb is None:
        max_file_size_mb = config.logging.max_file_size_mb
    if backup_count is None:
        backup_count = config.logging.backup_count
    
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if file_output and file_path:
        # Create log directory if it doesn't exist
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set up specific loggers
    _setup_module_loggers(numeric_level, formatter)

def _setup_module_loggers(level: int, formatter: logging.Formatter):
    """Set up loggers for specific modules."""
    
    # TensorFlow logging
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.setLevel(logging.WARNING)  # Reduce TensorFlow verbosity
    
    # PIL logging
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.WARNING)
    
    # OpenCV logging
    cv_logger = logging.getLogger('cv2')
    cv_logger.setLevel(logging.WARNING)
    
    # FastAPI logging
    fastapi_logger = logging.getLogger('fastapi')
    fastapi_logger.setLevel(level)
    
    # Uvicorn logging
    uvicorn_logger = logging.getLogger('uvicorn')
    uvicorn_logger.setLevel(level)
    
    # Application loggers
    app_loggers = [
        'src',
        'src.models',
        'src.preprocessing',
        'src.api',
        'src.utils'
    ]
    
    for logger_name in app_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """
    Log a function call with parameters.
    
    Args:
        logger: Logger instance
        func_name: Function name
        **kwargs: Function parameters
    """
    params = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Calling {func_name}({params})")

def log_execution_time(logger: logging.Logger, func_name: str, execution_time: float):
    """
    Log function execution time.
    
    Args:
        logger: Logger instance
        func_name: Function name
        execution_time: Execution time in seconds
    """
    logger.info(f"{func_name} executed in {execution_time:.3f} seconds")

def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """
    Log an error with context.
    
    Args:
        logger: Logger instance
        error: Exception instance
        context: Additional context string
    """
    context_str = f" in {context}" if context else ""
    logger.error(f"Error{context_str}: {type(error).__name__}: {str(error)}")

def log_performance(logger: logging.Logger, operation: str, **metrics):
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        operation: Operation name
        **metrics: Performance metrics
    """
    metrics_str = ', '.join([f"{k}={v}" for k, v in metrics.items()])
    logger.info(f"Performance - {operation}: {metrics_str}")

def log_model_info(logger: logging.Logger, model_info: Dict[str, Any]):
    """
    Log model information.
    
    Args:
        logger: Logger instance
        model_info: Model information dictionary
    """
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")

def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None
):
    """
    Log training progress.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        total_epochs: Total number of epochs
        train_loss: Training loss
        train_acc: Training accuracy
        val_loss: Validation loss (optional)
        val_acc: Validation accuracy (optional)
    """
    progress = (epoch / total_epochs) * 100
    val_str = f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}" if val_loss is not None else ""
    logger.info(
        f"Epoch {epoch}/{total_epochs} ({progress:.1f}%) - "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}{val_str}"
    )

def log_api_request(
    logger: logging.Logger,
    method: str,
    endpoint: str,
    status_code: int,
    response_time_ms: float,
    **kwargs
):
    """
    Log API request information.
    
    Args:
        logger: Logger instance
        method: HTTP method
        endpoint: API endpoint
        status_code: HTTP status code
        response_time_ms: Response time in milliseconds
        **kwargs: Additional information
    """
    extra_info = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    extra_str = f" ({extra_info})" if extra_info else ""
    logger.info(
        f"API {method} {endpoint} - {status_code} - {response_time_ms:.1f}ms{extra_str}"
    )

def log_classification_result(
    logger: logging.Logger,
    predicted_class: int,
    confidence: float,
    processing_time_ms: float,
    image_info: Optional[Dict[str, Any]] = None
):
    """
    Log image classification result.
    
    Args:
        logger: Logger instance
        predicted_class: Predicted class index
        confidence: Prediction confidence
        processing_time_ms: Processing time in milliseconds
        image_info: Image information (optional)
    """
    image_str = f", image_info={image_info}" if image_info else ""
    logger.info(
        f"Classification - class={predicted_class}, confidence={confidence:.3f}, "
        f"time={processing_time_ms:.1f}ms{image_str}"
    )

def log_batch_classification_result(
    logger: logging.Logger,
    total_images: int,
    successful: int,
    failed: int,
    total_time_ms: float,
    avg_time_ms: float
):
    """
    Log batch classification result.
    
    Args:
        logger: Logger instance
        total_images: Total number of images
        successful: Number of successful classifications
        failed: Number of failed classifications
        total_time_ms: Total processing time in milliseconds
        avg_time_ms: Average processing time per image in milliseconds
    """
    logger.info(
        f"Batch Classification - total={total_images}, successful={successful}, "
        f"failed={failed}, total_time={total_time_ms:.1f}ms, avg_time={avg_time_ms:.1f}ms"
    )

def log_data_loading(
    logger: logging.Logger,
    data_type: str,
    count: int,
    loading_time_seconds: float,
    **kwargs
):
    """
    Log data loading information.
    
    Args:
        logger: Logger instance
        data_type: Type of data (train, test, validation)
        count: Number of samples loaded
        loading_time_seconds: Loading time in seconds
        **kwargs: Additional information
    """
    extra_info = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    extra_str = f" ({extra_info})" if extra_info else ""
    logger.info(
        f"Data Loading - {data_type}: {count} samples in {loading_time_seconds:.3f}s{extra_str}"
    )

def log_model_evaluation(
    logger: logging.Logger,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    evaluation_time_seconds: float
):
    """
    Log model evaluation results.
    
    Args:
        logger: Logger instance
        accuracy: Test accuracy
        precision: Test precision
        recall: Test recall
        f1_score: Test F1 score
        evaluation_time_seconds: Evaluation time in seconds
    """
    logger.info(
        f"Model Evaluation - accuracy={accuracy:.4f}, precision={precision:.4f}, "
        f"recall={recall:.4f}, f1_score={f1_score:.4f}, time={evaluation_time_seconds:.3f}s"
    )

# Initialize logging when module is imported
setup_logging()
