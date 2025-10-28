"""
Training script for computer vision image classification.

This script provides a command-line interface for training models
with configurable hyperparameters and data augmentation.
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import get_config, set_config
from src.utils.logging import setup_logging, get_logger, log_training_progress
from src.models.cnn import CNNModel
from src.preprocessing.transforms import ImagePreprocessor

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a computer vision image classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to training data directory"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        help="Path to test data directory (optional)"
    )
    parser.add_argument(
        "--validation_dir",
        type=str,
        help="Path to validation data directory (optional)"
    )
    
    # Model arguments
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["standard", "deep", "light"],
        default="standard",
        help="Model architecture type"
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="224,224,3",
        help="Input image shape (height,width,channels)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        help="Number of output classes (auto-detected if not specified)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    
    # Regularization arguments
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate"
    )
    parser.add_argument(
        "--l2_reg",
        type=float,
        default=1e-4,
        help="L2 regularization factor"
    )
    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
        default=True,
        help="Use batch normalization"
    )
    parser.add_argument(
        "--no_batch_norm",
        action="store_false",
        dest="use_batch_norm",
        help="Disable batch normalization"
    )
    
    # Data augmentation arguments
    parser.add_argument(
        "--use_data_augmentation",
        action="store_true",
        default=True,
        help="Use data augmentation"
    )
    parser.add_argument(
        "--no_data_augmentation",
        action="store_false",
        dest="use_data_augmentation",
        help="Disable data augmentation"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="trained_model",
        help="Name for the trained model"
    )
    parser.add_argument(
        "--save_best_only",
        action="store_true",
        default=True,
        help="Save only the best model"
    )
    parser.add_argument(
        "--save_all_epochs",
        action="store_false",
        dest="save_best_only",
        help="Save model after each epoch"
    )
    
    # Training control arguments
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--reduce_lr_patience",
        type=int,
        default=5,
        help="Reduce learning rate patience"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-7,
        help="Minimum learning rate"
    )
    
    # Other arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to configuration file"
    )
    
    return parser.parse_args()

def setup_environment(seed: Optional[int] = None):
    """Set up environment for reproducible training."""
    import numpy as np
    import tensorflow as tf
    import random
    
    if seed is not None:
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Configure TensorFlow for reproducibility
        tf.config.experimental.enable_op_determinism()
        
        logger.info(f"Random seed set to {seed}")

def detect_num_classes(data_dir: str) -> int:
    """Detect number of classes from data directory structure."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Look for class directories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {data_dir}")
    
    logger.info(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    return len(class_dirs)

def count_samples(data_dir: str) -> Dict[str, int]:
    """Count samples in each class."""
    data_path = Path(data_dir)
    class_counts = {}
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            # Count image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                image_files.extend(class_dir.glob(f'*{ext}'))
                image_files.extend(class_dir.glob(f'*{ext.upper()}'))
            
            class_counts[class_dir.name] = len(image_files)
    
    return class_counts

def create_model(
    input_shape: tuple,
    num_classes: int,
    architecture: str,
    dropout_rate: float,
    l2_reg: float,
    use_batch_norm: bool,
    use_data_augmentation: bool
) -> CNNModel:
    """Create and configure the model."""
    logger.info("Creating model...")
    logger.info(f"Architecture: {architecture}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Dropout rate: {dropout_rate}")
    logger.info(f"L2 regularization: {l2_reg}")
    logger.info(f"Batch normalization: {use_batch_norm}")
    logger.info(f"Data augmentation: {use_data_augmentation}")
    
    model = CNNModel(
        input_shape=input_shape,
        num_classes=num_classes,
        architecture=architecture,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        use_batch_norm=use_batch_norm,
        use_data_augmentation=use_data_augmentation
    )
    
    # Build and compile model
    model.build_model()
    model.compile_model(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Print model summary
    if args.verbose:
        print("\nModel Architecture:")
        print(model.get_model_summary())
    
    return model

def create_callbacks(
    output_dir: str,
    model_name: str,
    save_best_only: bool,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    min_lr: float
):
    """Create training callbacks."""
    import tensorflow as tf
    
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_path = os.path.join(output_dir, f"{model_name}_checkpoint.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=save_best_only,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping_callback)
    
    # Reduce learning rate callback
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1
    )
    callbacks.append(reduce_lr_callback)
    
    # CSV logger callback
    csv_path = os.path.join(output_dir, f"{model_name}_training_log.csv")
    csv_callback = tf.keras.callbacks.CSVLogger(csv_path)
    callbacks.append(csv_callback)
    
    return callbacks

def main():
    """Main training function."""
    global args, logger
    
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    logger = get_logger(__name__)
    
    # Load configuration if provided
    if args.config_file:
        config = get_config().load_from_file(args.config_file)
        set_config(config)
        logger.info(f"Configuration loaded from {args.config_file}")
    
    # Set up environment
    setup_environment(args.seed)
    
    # Parse input shape
    try:
        input_shape = tuple(map(int, args.input_shape.split(',')))
        if len(input_shape) != 3:
            raise ValueError("Input shape must have 3 dimensions")
    except ValueError as e:
        logger.error(f"Invalid input shape: {e}")
        sys.exit(1)
    
    # Detect number of classes if not specified
    if args.num_classes is None:
        args.num_classes = detect_num_classes(args.data_dir)
    
    # Count samples
    class_counts = count_samples(args.data_dir)
    total_samples = sum(class_counts.values())
    logger.info(f"Total training samples: {total_samples}")
    for class_name, count in class_counts.items():
        logger.info(f"  {class_name}: {count} samples")
    
    # Check minimum samples per class
    min_samples = min(class_counts.values())
    if min_samples < 10:
        logger.warning(f"Some classes have very few samples (minimum: {min_samples})")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    model = create_model(
        input_shape=input_shape,
        num_classes=args.num_classes,
        architecture=args.architecture,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg,
        use_batch_norm=args.use_batch_norm,
        use_data_augmentation=args.use_data_augmentation
    )
    
    # Create data generators
    logger.info("Creating data generators...")
    train_gen, val_gen = model.create_data_generator(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        image_size=input_shape[:2],
        shuffle=True
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        output_dir=args.output_dir,
        model_name=args.model_name,
        save_best_only=args.save_best_only,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        min_lr=args.min_lr
    )
    
    # Train model
    logger.info("Starting training...")
    start_time = time.time()
    
    try:
        history = model.train(
            train_data=train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1 if args.verbose else 2
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, f"{args.model_name}_final.h5")
        model.save_model(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Print training summary
        final_epoch = len(history.history['loss'])
        final_train_loss = history.history['loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        final_val_loss = history.history.get('val_loss', [0])[-1]
        final_val_acc = history.history.get('val_accuracy', [0])[-1]
        
        logger.info("Training Summary:")
        logger.info(f"  Final epoch: {final_epoch}")
        logger.info(f"  Training loss: {final_train_loss:.4f}")
        logger.info(f"  Training accuracy: {final_train_acc:.4f}")
        logger.info(f"  Validation loss: {final_val_loss:.4f}")
        logger.info(f"  Validation accuracy: {final_val_acc:.4f}")
        
        # Evaluate on test data if provided
        if args.test_dir and os.path.exists(args.test_dir):
            logger.info("Evaluating on test data...")
            test_gen, _ = model.create_data_generator(
                data_dir=args.test_dir,
                batch_size=args.batch_size,
                validation_split=0.0,
                image_size=input_shape[:2],
                shuffle=False
            )
            
            test_results = model.evaluate(test_gen, verbose=1)
            logger.info("Test Results:")
            for metric, value in test_results.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
