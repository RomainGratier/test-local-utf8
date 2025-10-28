"""
Image transformation and preprocessing functions.

This module provides core image preprocessing capabilities including resize,
normalization, format conversion, and batch processing for the computer vision
classification system.
"""

import os
import logging
from typing import Union, List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import cv2
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB in bytes

class ImagePreprocessor:
    """
    Main image preprocessing class that handles resize, normalization,
    and format conversion operations.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        target_format: str = 'RGB',
        interpolation: int = cv2.INTER_LINEAR,
        max_size_mb: int = 10
    ):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target size for resizing (width, height)
            normalize: Whether to normalize pixel values to [0, 1]
            target_format: Target color format ('RGB', 'BGR', 'GRAY')
            interpolation: OpenCV interpolation method
            max_size_mb: Maximum image size in MB
        """
        self.target_size = target_size
        self.normalize = normalize
        self.target_format = target_format
        self.interpolation = interpolation
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
    def preprocess(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess a single image with resize, normalize, and format conversion.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed image as numpy array
            
        Raises:
            ValueError: If image format is not supported
            FileNotFoundError: If image file doesn't exist
            OSError: If image cannot be loaded
        """
        # Load image if it's a file path
        if isinstance(image, (str, Path)):
            image = self._load_image(str(image))
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif not isinstance(image, np.ndarray):
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        # Validate image
        self._validate_image(image)
        
        # Apply preprocessing steps
        processed = self._resize_image(image)
        processed = self._convert_format(processed)
        
        if self.normalize:
            processed = self._normalize_image(processed)
            
        return processed
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Check file size
        file_size = os.path.getsize(image_path)
        if file_size > self.max_size_bytes:
            raise ValueError(f"Image too large: {file_size / 1024 / 1024:.1f}MB > {self.max_size_bytes / 1024 / 1024}MB")
            
        # Check file extension
        ext = Path(image_path).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}. Supported: {SUPPORTED_FORMATS}")
            
        try:
            # Try loading with OpenCV first (better for various formats)
            image = cv2.imread(image_path)
            if image is None:
                # Fallback to PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
            else:
                # Convert BGR to RGB for consistency
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            return image
        except Exception as e:
            raise OSError(f"Failed to load image {image_path}: {str(e)}")
    
    def _validate_image(self, image: np.ndarray) -> None:
        """Validate image array."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
            
        if image.size == 0:
            raise ValueError("Image is empty")
            
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D or 3D array")
            
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            raise ValueError(f"Invalid number of channels: {image.shape[2]}. Expected 1, 3, or 4")
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        if image.shape[:2] == self.target_size[::-1]:  # (height, width) vs (width, height)
            return image
            
        try:
            resized = cv2.resize(
                image, 
                self.target_size, 
                interpolation=self.interpolation
            )
            return resized
        except Exception as e:
            raise ValueError(f"Failed to resize image: {str(e)}")
    
    def _convert_format(self, image: np.ndarray) -> np.ndarray:
        """Convert image to target format."""
        if self.target_format == 'RGB':
            if len(image.shape) == 2:  # Grayscale
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # Already RGB
                return image
            else:
                raise ValueError(f"Cannot convert {image.shape[2]} channels to RGB")
                
        elif self.target_format == 'BGR':
            if len(image.shape) == 2:  # Grayscale
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:  # RGB to BGR
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"Cannot convert {image.shape[2]} channels to BGR")
                
        elif self.target_format == 'GRAY':
            if len(image.shape) == 2:  # Already grayscale
                return image
            elif image.shape[2] == 3:  # RGB to grayscale
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:  # RGBA to grayscale
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                raise ValueError(f"Cannot convert {image.shape[2]} channels to GRAY")
        else:
            raise ValueError(f"Unsupported target format: {self.target_format}")
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to [0, 1] range."""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            # Assume already in [0, 1] range or normalize to [0, 1]
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / 255.0
            return image


def resize_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize an image to target size.
    
    Args:
        image: Input image (file path, numpy array, or PIL Image)
        target_size: Target size (width, height)
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized image as numpy array
    """
    preprocessor = ImagePreprocessor(target_size=target_size, normalize=False)
    return preprocessor._resize_image(preprocessor.preprocess(image))


def normalize_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    target_format: str = 'RGB'
) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image: Input image (file path, numpy array, or PIL Image)
        target_format: Target color format
        
    Returns:
        Normalized image as numpy array
    """
    preprocessor = ImagePreprocessor(normalize=True, target_format=target_format)
    return preprocessor.preprocess(image)


def convert_format(
    image: Union[str, Path, np.ndarray, Image.Image],
    target_format: str = 'RGB'
) -> np.ndarray:
    """
    Convert image to target format.
    
    Args:
        image: Input image (file path, numpy array, or PIL Image)
        target_format: Target color format ('RGB', 'BGR', 'GRAY')
        
    Returns:
        Converted image as numpy array
    """
    preprocessor = ImagePreprocessor(normalize=False, target_format=target_format)
    return preprocessor.preprocess(image)


def preprocess_single(
    image: Union[str, Path, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    target_format: str = 'RGB'
) -> np.ndarray:
    """
    Preprocess a single image with all transformations.
    
    Args:
        image: Input image (file path, numpy array, or PIL Image)
        target_size: Target size (width, height)
        normalize: Whether to normalize pixel values
        target_format: Target color format
        
    Returns:
        Preprocessed image as numpy array
    """
    preprocessor = ImagePreprocessor(
        target_size=target_size,
        normalize=normalize,
        target_format=target_format
    )
    return preprocessor.preprocess(image)


def preprocess_batch(
    images: List[Union[str, Path, np.ndarray, Image.Image]],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    target_format: str = 'RGB',
    return_errors: bool = False
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[str]]]:
    """
    Preprocess a batch of images.
    
    Args:
        images: List of input images
        target_size: Target size (width, height)
        normalize: Whether to normalize pixel values
        target_format: Target color format
        return_errors: Whether to return error messages for failed images
        
    Returns:
        List of preprocessed images, optionally with error messages
    """
    preprocessor = ImagePreprocessor(
        target_size=target_size,
        normalize=normalize,
        target_format=target_format
    )
    
    processed_images = []
    errors = []
    
    for i, image in enumerate(images):
        try:
            processed = preprocessor.preprocess(image)
            processed_images.append(processed)
        except Exception as e:
            error_msg = f"Failed to process image {i}: {str(e)}"
            logger.error(error_msg)
            if return_errors:
                errors.append(error_msg)
            else:
                raise e
    
    if return_errors:
        return processed_images, errors
    return processed_images


def validate_image(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate an image file and return information about it.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information and validation results
    """
    info = {
        'valid': False,
        'path': str(image_path),
        'exists': False,
        'size_bytes': 0,
        'format': None,
        'dimensions': None,
        'channels': None,
        'dtype': None,
        'error': None
    }
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            info['error'] = "File does not exist"
            return info
        info['exists'] = True
        
        # Check file size
        file_size = os.path.getsize(image_path)
        info['size_bytes'] = file_size
        if file_size > MAX_IMAGE_SIZE:
            info['error'] = f"File too large: {file_size / 1024 / 1024:.1f}MB"
            return info
        
        # Check file extension
        ext = Path(image_path).suffix.lower()
        info['format'] = ext
        if ext not in SUPPORTED_FORMATS:
            info['error'] = f"Unsupported format: {ext}"
            return info
        
        # Try to load image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                # Fallback to PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            info['error'] = f"Failed to load image: {str(e)}"
            return info
        
        # Get image properties
        info['dimensions'] = image.shape[:2]  # (height, width)
        info['channels'] = image.shape[2] if len(image.shape) == 3 else 1
        info['dtype'] = str(image.dtype)
        info['valid'] = True
        
    except Exception as e:
        info['error'] = f"Validation error: {str(e)}"
    
    return info


def get_image_info(image: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
    """
    Get detailed information about an image.
    
    Args:
        image: Input image (file path, numpy array, or PIL Image)
        
    Returns:
        Dictionary with image information
    """
    info = {
        'type': type(image).__name__,
        'shape': None,
        'dtype': None,
        'channels': None,
        'size_bytes': None,
        'format': None
    }
    
    try:
        if isinstance(image, (str, Path)):
            # File path
            info['format'] = Path(image).suffix.lower()
            info['size_bytes'] = os.path.getsize(image)
            image_array = cv2.imread(str(image))
            if image_array is not None:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image_array = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            # PIL Image
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            # Numpy array
            image_array = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        info['shape'] = image_array.shape
        info['dtype'] = str(image_array.dtype)
        info['channels'] = image_array.shape[2] if len(image_array.shape) == 3 else 1
        
    except Exception as e:
        info['error'] = str(e)
    
    return info
    def preprocess_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess an image from bytes.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image from bytes
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes))
            image_array = np.array(image)
            
            # Validate image
            self._validate_image(image_array)
            
            # Apply preprocessing steps
            processed = self._resize_image(image_array)
            processed = self._convert_format(processed)
            
            if self.normalize:
                processed = self._normalize_image(processed)
                
            return processed
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess image from bytes: {str(e)}")
