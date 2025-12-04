"""
Security utilities for Agrivision API
Handles file validation, size checks, and image integrity verification
"""

from PIL import Image
import io
import imghdr
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum file size: 5MB
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes

# Allowed MIME types
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg", 
    "image/png"
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Minimum image dimensions (to filter out very small/invalid images)
MIN_IMAGE_SIZE = 32

# Green color range for plant detection (HSV)
# Plants typically have green hues
GREEN_HUE_RANGE = (35, 85)  # Hue values for green in HSV (0-180 scale)
GREEN_SATURATION_MIN = 20   # Minimum saturation for green detection


class SecurityError(Exception):
    """Custom exception for security validation errors"""
    def __init__(self, message: str, error_code: str, status_code: int = 400):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)


def validate_file_type(filename: Optional[str], content_type: Optional[str]) -> bool:
    """
    Validate file type based on filename extension and MIME type
    
    Args:
        filename: Original filename
        content_type: MIME type from upload
    
    Returns:
        True if valid
    
    Raises:
        SecurityError: If file type is invalid
    """
    # Check MIME type
    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
        logger.warning(f"Invalid MIME type: {content_type}")
        raise SecurityError(
            message=f"Invalid file type. Allowed types: JPEG, PNG. Received: {content_type}",
            error_code="INVALID_MIME_TYPE",
            status_code=415
        )
    
    # Check file extension
    if filename:
        ext = "." + filename.lower().split(".")[-1] if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"Invalid file extension: {ext}")
            raise SecurityError(
                message=f"Invalid file extension. Allowed extensions: .jpg, .jpeg, .png",
                error_code="INVALID_EXTENSION",
                status_code=415
            )
    
    return True


def validate_file_size(content: bytes) -> bool:
    """
    Validate file size is within limits
    
    Args:
        content: File content as bytes
    
    Returns:
        True if valid
    
    Raises:
        SecurityError: If file is too large
    """
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE:
        size_mb = file_size / (1024 * 1024)
        logger.warning(f"File too large: {size_mb:.2f}MB")
        raise SecurityError(
            message=f"File too large. Maximum size: 5MB. Received: {size_mb:.2f}MB",
            error_code="FILE_TOO_LARGE",
            status_code=413
        )
    
    if file_size == 0:
        raise SecurityError(
            message="Empty file received",
            error_code="EMPTY_FILE",
            status_code=400
        )
    
    return True


def validate_image_integrity(content: bytes) -> Image.Image:
    """
    Validate that the content is a valid, non-corrupt image
    
    Args:
        content: File content as bytes
    
    Returns:
        PIL Image object if valid
    
    Raises:
        SecurityError: If image is corrupt or invalid
    """
    # Check magic bytes using imghdr
    detected_type = imghdr.what(None, h=content)
    if detected_type not in ["jpeg", "png"]:
        logger.warning(f"Invalid image magic bytes. Detected: {detected_type}")
        raise SecurityError(
            message="Invalid image format. File does not appear to be a valid JPEG or PNG image",
            error_code="INVALID_IMAGE_FORMAT",
            status_code=400
        )
    
    try:
        # Try to open and verify the image
        image = Image.open(io.BytesIO(content))
        image.verify()  # Verify image integrity
        
        # Need to reopen after verify (verify() closes the file)
        image = Image.open(io.BytesIO(content))
        
        # Check image dimensions
        width, height = image.size
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            raise SecurityError(
                message=f"Image too small. Minimum size: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE} pixels",
                error_code="IMAGE_TOO_SMALL",
                status_code=400
            )
        
        # Force load the image data to check for corruption
        image.load()
        
        return image
        
    except SecurityError:
        raise
    except Exception as e:
        logger.warning(f"Image validation failed: {e}")
        raise SecurityError(
            message="Corrupt or invalid image file",
            error_code="CORRUPT_IMAGE",
            status_code=400
        )


def validate_plant_image(image: Image.Image) -> bool:
    """
    Validate that the image likely contains a plant
    Uses color analysis to detect presence of green (plant) colors
    
    Args:
        image: PIL Image object
    
    Returns:
        True if image appears to contain plant material
    
    Raises:
        SecurityError: If image does not appear to be a plant image
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for faster analysis
        analysis_size = (100, 100)
        resized = image.resize(analysis_size, Image.Resampling.LANCZOS)
        
        # Convert to HSV for color analysis
        # We'll use a simple approach: check for presence of green-ish colors
        import numpy as np
        
        rgb_array = np.array(resized)
        
        # Convert RGB to HSV manually (simplified)
        # Normalize RGB to 0-1
        rgb_normalized = rgb_array.astype(np.float32) / 255.0
        
        r, g, b = rgb_normalized[:, :, 0], rgb_normalized[:, :, 1], rgb_normalized[:, :, 2]
        
        # Calculate value (brightness)
        v = np.maximum(np.maximum(r, g), b)
        
        # Calculate saturation
        min_rgb = np.minimum(np.minimum(r, g), b)
        s = np.where(v > 0, (v - min_rgb) / v, 0)
        
        # Calculate hue
        diff = v - min_rgb
        h = np.zeros_like(v)
        
        # When max is r
        mask = (v == r) & (diff > 0)
        h[mask] = 60 * ((g[mask] - b[mask]) / diff[mask] % 6)
        
        # When max is g
        mask = (v == g) & (diff > 0)
        h[mask] = 60 * ((b[mask] - r[mask]) / diff[mask] + 2)
        
        # When max is b
        mask = (v == b) & (diff > 0)
        h[mask] = 60 * ((r[mask] - g[mask]) / diff[mask] + 4)
        
        # Normalize hue to 0-180 (OpenCV convention)
        h = h / 2
        
        # Convert saturation to 0-255 scale
        s = s * 255
        
        # Count pixels that are in the green range
        green_mask = (
            (h >= GREEN_HUE_RANGE[0]) & 
            (h <= GREEN_HUE_RANGE[1]) & 
            (s >= GREEN_SATURATION_MIN)
        )
        
        green_pixel_ratio = np.sum(green_mask) / (analysis_size[0] * analysis_size[1])
        
        # Also check for yellow-green and brown tones (common in diseased plants)
        # Extended range for plant material (includes browns and yellows)
        extended_plant_mask = (
            ((h >= 20) & (h <= 90) & (s >= 15)) |  # Green to yellow range
            ((h >= 0) & (h <= 30) & (s >= 20) & (s <= 150))  # Brown/orange range (diseased leaves)
        )
        
        plant_pixel_ratio = np.sum(extended_plant_mask) / (analysis_size[0] * analysis_size[1])
        
        # Require at least 5% green or 10% plant-like colors
        if green_pixel_ratio < 0.05 and plant_pixel_ratio < 0.10:
            logger.warning(f"Image does not appear to contain plant material. Green ratio: {green_pixel_ratio:.2%}, Plant ratio: {plant_pixel_ratio:.2%}")
            raise SecurityError(
                message="Image does not appear to contain a plant. Please upload an image of a plant leaf or plant part.",
                error_code="NOT_PLANT_IMAGE",
                status_code=400
            )
        
        logger.debug(f"Plant validation passed. Green ratio: {green_pixel_ratio:.2%}, Plant ratio: {plant_pixel_ratio:.2%}")
        return True
        
    except SecurityError:
        raise
    except Exception as e:
        # If analysis fails, we allow the image through (fail open for this check)
        logger.warning(f"Plant validation analysis failed: {e}. Allowing image.")
        return True
