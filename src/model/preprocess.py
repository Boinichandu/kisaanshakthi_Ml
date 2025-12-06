"""
Image preprocessing and prediction for Agrivision
Handles image transformation and model inference
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Input image size for MobileNetV2
INPUT_SIZE = 224


def get_transform() -> transforms.Compose:
    """
    Get the image transformation pipeline

    Returns:
        Composed transforms for preprocessing
    """
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for model input

    Args:
        image: PIL Image object

    Returns:
        Preprocessed tensor of shape (1, 3, 224, 224)
    """
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply transformations
    transform = get_transform()
    tensor = transform(image)

    # Cast to Tensor (transform returns Tensor but type checker doesn't know)
    assert isinstance(tensor, torch.Tensor)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    logger.debug(f"Preprocessed image to tensor of shape {tensor.shape}")
    return tensor


def predict(
    model: torch.nn.Module, tensor: torch.Tensor, class_names: list[str]
) -> tuple[str, float]:
    """
    Run prediction on preprocessed image tensor

    Args:
        model: Loaded PyTorch model in eval mode
        tensor: Preprocessed image tensor of shape (1, 3, 224, 224)
        class_names: List of class names

    Returns:
        Tuple of (predicted_class_name, confidence_score)
    """
    # Ensure model is in eval mode
    model.eval()

    # Disable gradient computation for inference
    with torch.no_grad():
        # Forward pass
        outputs = model(tensor)

        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Get the class with highest probability
        confidence, predicted_idx = torch.max(probabilities, dim=1)

        # Get class name and confidence value
        predicted_class = class_names[int(predicted_idx.item())]
        confidence_score = float(confidence.item())

    logger.debug(
        f"Prediction: {predicted_class} with confidence {confidence_score:.4f}"
    )
    return predicted_class, confidence_score


def predict_top_k(
    model: torch.nn.Module, tensor: torch.Tensor, class_names: list[str], k: int = 5
) -> list[tuple[str, float]]:
    """
    Get top-k predictions for an image

    Args:
        model: Loaded PyTorch model in eval mode
        tensor: Preprocessed image tensor
        class_names: List of class names
        k: Number of top predictions to return

    Returns:
        List of tuples (class_name, confidence) sorted by confidence
    """
    model.eval()

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k, dim=1)

        results: list[tuple[str, float]] = []
        for i in range(k):
            class_name = class_names[int(top_indices[0][i].item())]
            confidence = float(top_probs[0][i].item())
            results.append((class_name, confidence))

    return results
