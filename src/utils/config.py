"""
Configuration management for the image classification system.

This module provides centralized configuration management using environment
variables and default values for all system components.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration settings."""
    
    input_shape: tuple = (224, 224, 3)
    num_classes: int = 3
    architecture: str = "standard"
    dropout_rate: float = 0.5
    l2_regularization: float = 1e-4
    use_batch_norm: bool = True
    use_data_augmentation: bool = False
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall"])

@dataclass
class TrainingConfig:
    """Training configuration settings."""
    
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    min_lr: float = 1e-7
    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: str = "min"

@dataclass
class DataConfig:
    """Data configuration settings."""
    
    train_dir: str = "data/train"
    test_dir: str = "data/test"
    validation_dir: str = "data/validation"
    supported_formats: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"])
    max_image_size_mb: int = 10
    target_size: tuple = (224, 224)
    normalize: bool = True
    target_format: str = "RGB"

@dataclass
class APIConfig:
    """API configuration settings."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    max_file_size_mb: int = 10
    max_batch_size: int = 10
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "logs/app.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    console_output: bool = True
    file_output: bool = True

@dataclass
class Config:
    """Main configuration class."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    environment: str = "development"
    version: str = "1.0.0"
    model_path: Optional[str] = None
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "class_0",
        1: "class_1",
        2: "class_2"
    })
    
    def __post_init__(self):
        """Post-initialization processing."""
        self._load_from_env()
        self._validate_config()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Model settings
        if os.getenv("MODEL_INPUT_SHAPE"):
            shape_str = os.getenv("MODEL_INPUT_SHAPE")
            try:
                # Parse "224,224,3" format
                shape = tuple(map(int, shape_str.split(",")))
                self.model.input_shape = shape
            except ValueError:
                logger.warning(f"Invalid MODEL_INPUT_SHAPE: {shape_str}")
        
        if os.getenv("MODEL_NUM_CLASSES"):
            self.model.num_classes = int(os.getenv("MODEL_NUM_CLASSES"))
        
        if os.getenv("MODEL_ARCHITECTURE"):
            self.model.architecture = os.getenv("MODEL_ARCHITECTURE")
        
        if os.getenv("MODEL_DROPOUT_RATE"):
            self.model.dropout_rate = float(os.getenv("MODEL_DROPOUT_RATE"))
        
        if os.getenv("MODEL_L2_REGULARIZATION"):
            self.model.l2_regularization = float(os.getenv("MODEL_L2_REGULARIZATION"))
        
        if os.getenv("MODEL_USE_BATCH_NORM"):
            self.model.use_batch_norm = os.getenv("MODEL_USE_BATCH_NORM").lower() == "true"
        
        if os.getenv("MODEL_USE_DATA_AUGMENTATION"):
            self.model.use_data_augmentation = os.getenv("MODEL_USE_DATA_AUGMENTATION").lower() == "true"
        
        if os.getenv("MODEL_LEARNING_RATE"):
            self.model.learning_rate = float(os.getenv("MODEL_LEARNING_RATE"))
        
        if os.getenv("MODEL_OPTIMIZER"):
            self.model.optimizer = os.getenv("MODEL_OPTIMIZER")
        
        # Training settings
        if os.getenv("TRAINING_EPOCHS"):
            self.training.epochs = int(os.getenv("TRAINING_EPOCHS"))
        
        if os.getenv("TRAINING_BATCH_SIZE"):
            self.training.batch_size = int(os.getenv("TRAINING_BATCH_SIZE"))
        
        if os.getenv("TRAINING_VALIDATION_SPLIT"):
            self.training.validation_split = float(os.getenv("TRAINING_VALIDATION_SPLIT"))
        
        if os.getenv("TRAINING_EARLY_STOPPING_PATIENCE"):
            self.training.early_stopping_patience = int(os.getenv("TRAINING_EARLY_STOPPING_PATIENCE"))
        
        # Data settings
        if os.getenv("DATA_TRAIN_DIR"):
            self.data.train_dir = os.getenv("DATA_TRAIN_DIR")
        
        if os.getenv("DATA_TEST_DIR"):
            self.data.test_dir = os.getenv("DATA_TEST_DIR")
        
        if os.getenv("DATA_VALIDATION_DIR"):
            self.data.validation_dir = os.getenv("DATA_VALIDATION_DIR")
        
        if os.getenv("DATA_MAX_IMAGE_SIZE_MB"):
            self.data.max_image_size_mb = int(os.getenv("DATA_MAX_IMAGE_SIZE_MB"))
        
        if os.getenv("DATA_TARGET_SIZE"):
            size_str = os.getenv("DATA_TARGET_SIZE")
            try:
                size = tuple(map(int, size_str.split(",")))
                self.data.target_size = size
            except ValueError:
                logger.warning(f"Invalid DATA_TARGET_SIZE: {size_str}")
        
        if os.getenv("DATA_NORMALIZE"):
            self.data.normalize = os.getenv("DATA_NORMALIZE").lower() == "true"
        
        if os.getenv("DATA_TARGET_FORMAT"):
            self.data.target_format = os.getenv("DATA_TARGET_FORMAT")
        
        # API settings
        if os.getenv("API_HOST"):
            self.api.host = os.getenv("API_HOST")
        
        if os.getenv("API_PORT"):
            self.api.port = int(os.getenv("API_PORT"))
        
        if os.getenv("API_DEBUG"):
            self.api.debug = os.getenv("API_DEBUG").lower() == "true"
        
        if os.getenv("API_RELOAD"):
            self.api.reload = os.getenv("API_RELOAD").lower() == "true"
        
        if os.getenv("API_WORKERS"):
            self.api.workers = int(os.getenv("API_WORKERS"))
        
        if os.getenv("API_MAX_FILE_SIZE_MB"):
            self.api.max_file_size_mb = int(os.getenv("API_MAX_FILE_SIZE_MB"))
        
        if os.getenv("API_MAX_BATCH_SIZE"):
            self.api.max_batch_size = int(os.getenv("API_MAX_BATCH_SIZE"))
        
        # Logging settings
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL")
        
        if os.getenv("LOG_FILE"):
            self.logging.file_path = os.getenv("LOG_FILE")
        
        if os.getenv("LOG_CONSOLE_OUTPUT"):
            self.logging.console_output = os.getenv("LOG_CONSOLE_OUTPUT").lower() == "true"
        
        if os.getenv("LOG_FILE_OUTPUT"):
            self.logging.file_output = os.getenv("LOG_FILE_OUTPUT").lower() == "true"
        
        # Global settings
        if os.getenv("ENVIRONMENT"):
            self.environment = os.getenv("ENVIRONMENT")
        
        if os.getenv("VERSION"):
            self.version = os.getenv("VERSION")
        
        if os.getenv("MODEL_PATH"):
            self.model_path = os.getenv("MODEL_PATH")
        
        # Class names from environment
        if os.getenv("CLASS_NAMES"):
            try:
                import json
                class_names_str = os.getenv("CLASS_NAMES")
                self.class_names = json.loads(class_names_str)
                # Convert string keys to integers
                self.class_names = {int(k): v for k, v in self.class_names.items()}
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Invalid CLASS_NAMES format: {e}")
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate model config
        if self.model.num_classes < 2:
            raise ValueError("Number of classes must be at least 2")
        
        if not (0.0 <= self.model.dropout_rate <= 1.0):
            raise ValueError("Dropout rate must be between 0.0 and 1.0")
        
        if self.model.l2_regularization < 0:
            raise ValueError("L2 regularization must be non-negative")
        
        if self.model.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate training config
        if self.training.epochs < 1:
            raise ValueError("Number of epochs must be at least 1")
        
        if self.training.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        if not (0.0 <= self.training.validation_split <= 1.0):
            raise ValueError("Validation split must be between 0.0 and 1.0")
        
        # Validate data config
        if self.data.max_image_size_mb <= 0:
            raise ValueError("Max image size must be positive")
        
        if len(self.data.target_size) != 2:
            raise ValueError("Target size must have exactly 2 dimensions")
        
        if any(size < 32 for size in self.data.target_size):
            raise ValueError("Target size dimensions must be at least 32")
        
        # Validate API config
        if self.api.port < 1 or self.api.port > 65535:
            raise ValueError("Port must be between 1 and 65535")
        
        if self.api.max_file_size_mb <= 0:
            raise ValueError("Max file size must be positive")
        
        if self.api.max_batch_size < 1:
            raise ValueError("Max batch size must be at least 1")
        
        # Validate logging config
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level.upper() not in valid_log_levels:
            raise ValueError(f"Log level must be one of: {valid_log_levels}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": {
                "input_shape": self.model.input_shape,
                "num_classes": self.model.num_classes,
                "architecture": self.model.architecture,
                "dropout_rate": self.model.dropout_rate,
                "l2_regularization": self.model.l2_regularization,
                "use_batch_norm": self.model.use_batch_norm,
                "use_data_augmentation": self.model.use_data_augmentation,
                "learning_rate": self.model.learning_rate,
                "optimizer": self.model.optimizer,
                "loss_function": self.model.loss_function,
                "metrics": self.model.metrics
            },
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "validation_split": self.training.validation_split,
                "early_stopping_patience": self.training.early_stopping_patience,
                "reduce_lr_patience": self.training.reduce_lr_patience,
                "min_lr": self.training.min_lr,
                "save_best_only": self.training.save_best_only,
                "monitor": self.training.monitor,
                "mode": self.training.mode
            },
            "data": {
                "train_dir": self.data.train_dir,
                "test_dir": self.data.test_dir,
                "validation_dir": self.data.validation_dir,
                "supported_formats": self.data.supported_formats,
                "max_image_size_mb": self.data.max_image_size_mb,
                "target_size": self.data.target_size,
                "normalize": self.data.normalize,
                "target_format": self.data.target_format
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug,
                "reload": self.api.reload,
                "workers": self.api.workers,
                "max_file_size_mb": self.api.max_file_size_mb,
                "max_batch_size": self.api.max_batch_size,
                "cors_origins": self.api.cors_origins,
                "cors_methods": self.api.cors_methods,
                "cors_headers": self.api.cors_headers
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file_path": self.logging.file_path,
                "max_file_size_mb": self.logging.max_file_size_mb,
                "backup_count": self.logging.backup_count,
                "console_output": self.logging.console_output,
                "file_output": self.logging.file_output
            },
            "environment": self.environment,
            "version": self.version,
            "model_path": self.model_path,
            "class_names": self.class_names
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to file."""
        import json
        
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        """Load configuration from file."""
        import json
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create config instance
        config = cls()
        
        # Update with loaded values
        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        if "api" in config_dict:
            for key, value in config_dict["api"].items():
                if hasattr(config.api, key):
                    setattr(config.api, key, value)
        
        if "logging" in config_dict:
            for key, value in config_dict["logging"].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        # Global settings
        if "environment" in config_dict:
            config.environment = config_dict["environment"]
        
        if "version" in config_dict:
            config.version = config_dict["version"]
        
        if "model_path" in config_dict:
            config.model_path = config_dict["model_path"]
        
        if "class_names" in config_dict:
            config.class_names = config_dict["class_names"]
        
        # Validate loaded config
        config._validate_config()
        
        logger.info(f"Configuration loaded from {filepath}")
        return config

# Global config instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

def set_config(config: Config):
    """Set the global configuration instance."""
    global _config
    _config = config
