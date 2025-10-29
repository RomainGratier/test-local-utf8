#!/usr/bin/env python3
"""
Model evaluation script for computer vision image classification.

This script provides comprehensive evaluation of trained models including
accuracy, precision, recall, F1-score, and confusion matrix analysis.
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.utils.config import get_config
from src.utils.logging import setup_logging, get_logger
from src.models.cnn import CNNModel
from src.preprocessing.transforms import ImagePreprocessor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a computer vision image classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Path to test data directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_report",
        type=str,
        help="Path to save evaluation report (JSON format)"
    )
    parser.add_argument(
        "--output_plots",
        type=str,
        help="Directory to save evaluation plots"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--class_names",
        type=str,
        help="Comma-separated list of class names"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def load_test_data(test_dir: str, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load test data from directory.
    
    Args:
        test_dir: Path to test data directory
        batch_size: Batch size for data loading
        
    Returns:
        Tuple of (images, labels, class_names)
    """
    logger = get_logger(__name__)
    logger.info(f"Loading test data from {test_dir}")
    
    test_path = Path(test_dir)
    if not test_path.exists():
        raise ValueError(f"Test directory does not exist: {test_dir}")
    
    # Get class directories
    class_dirs = [d for d in test_path.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class directories found in {test_dir}")
    
    class_names = sorted([d.name for d in class_dirs])
    logger.info(f"Found {len(class_names)} classes: {class_names}")
    
    # Load images and labels
    images = []
    labels = []
    
    preprocessor = ImagePreprocessor(target_size=(224, 224), normalize=True)
    
    for class_idx, class_dir in enumerate(sorted(class_dirs)):
        logger.info(f"Loading images from {class_dir.name}...")
        
        # Get image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            image_files.extend(class_dir.glob(f'*{ext}'))
            image_files.extend(class_dir.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images in {class_dir.name}")
        
        # Process images
        for image_file in image_files:
            try:
                # Load and preprocess image
                processed_image = preprocessor.preprocess(str(image_file))
                images.append(processed_image)
                labels.append(class_idx)
            except Exception as e:
                logger.warning(f"Failed to load {image_file}: {e}")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    logger.info(f"Loaded {len(images)} test images")
    logger.info(f"Image shape: {images.shape}")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    
    return images, labels, class_names


def evaluate_model(
    model: CNNModel,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    class_names: List[str],
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        test_images: Test images
        test_labels: Test labels
        class_names: List of class names
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    logger = get_logger(__name__)
    logger.info("Starting model evaluation...")
    
    start_time = time.time()
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(test_images, batch_size=batch_size, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predicted_classes)
    precision = precision_score(test_labels, predicted_classes, average='weighted')
    recall = recall_score(test_labels, predicted_classes, average='weighted')
    f1 = f1_score(test_labels, predicted_classes, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predicted_classes)
    
    # Per-class metrics
    precision_per_class = precision_score(test_labels, predicted_classes, average=None)
    recall_per_class = recall_score(test_labels, predicted_classes, average=None)
    f1_per_class = f1_score(test_labels, predicted_classes, average=None)
    
    # Create per-class metrics dictionary
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(np.sum(test_labels == i))
        }
    
    evaluation_time = time.time() - start_time
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_metrics': class_metrics,
        'evaluation_time_seconds': evaluation_time,
        'total_samples': len(test_images),
        'class_names': class_names
    }
    
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    
    return results


def create_evaluation_plots(
    results: Dict[str, Any],
    output_dir: str,
    class_names: List[str]
) -> None:
    """
    Create evaluation plots.
    
    Args:
        results: Evaluation results
        output_dir: Directory to save plots
        class_names: List of class names
    """
    logger = get_logger(__name__)
    logger.info(f"Creating evaluation plots in {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class metrics
    metrics = ['precision', 'recall', 'f1_score']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results['class_metrics'][class_name][metric] for class_name in class_names]
        bars = axes[i].bar(class_names, values, alpha=0.7)
        axes[i].set_title(f'Per-class {metric.title()}')
        axes[i].set_ylabel(metric.title())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Overall metrics comparison
    overall_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    values = [results[metric] for metric in overall_metrics]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(overall_metrics, values, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Overall Model Performance')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Evaluation plots saved successfully")


def save_evaluation_report(
    results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results
        output_path: Path to save the report
    """
    logger = get_logger(__name__)
    logger.info(f"Saving evaluation report to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation report saved successfully")


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Print evaluation summary to console."""
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    
    print(f"Total samples: {results['total_samples']}")
    print(f"Evaluation time: {results['evaluation_time_seconds']:.2f} seconds")
    print()
    
    print("Overall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-score:  {results['f1_score']:.4f}")
    print()
    
    print("Per-class Metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<8}")
    print("-" * 60)
    
    for class_name in results['class_names']:
        metrics = results['class_metrics'][class_name]
        print(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f} {metrics['support']:<8}")
    
    print("\n" + "="*60)


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    logger = get_logger(__name__)
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model_path}")
        model = CNNModel()
        model.load_model(args.model_path)
        logger.info("Model loaded successfully")
        
        # Load test data
        test_images, test_labels, class_names = load_test_data(args.test_dir, args.batch_size)
        
        # Override class names if provided
        if args.class_names:
            class_names = [name.strip() for name in args.class_names.split(',')]
            logger.info(f"Using provided class names: {class_names}")
        
        # Evaluate model
        results = evaluate_model(
            model=model,
            test_images=test_images,
            test_labels=test_labels,
            class_names=class_names,
            batch_size=args.batch_size
        )
        
        # Print summary
        print_evaluation_summary(results)
        
        # Save report if requested
        if args.output_report:
            save_evaluation_report(results, args.output_report)
        
        # Create plots if requested
        if args.output_plots:
            create_evaluation_plots(results, args.output_plots, class_names)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()