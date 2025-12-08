"""
Model loader for Agrivision
Loads MobileNetV2 model and provides class names for 38 plant disease categories
"""

import os
import torch
import torch.nn as nn
from torchvision import models
import logging

logger = logging.getLogger(__name__)

# Path to the model file
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "mobilenetv2_agrivision.pt"
)

# 38 Plant Disease Classes (PlantVillage Dataset)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def get_class_names() -> list:
    """
    Get the list of 38 plant disease class names

    Returns:
        List of class name strings
    """
    return CLASS_NAMES


def create_model(num_classes: int = 38) -> nn.Module:
    """
    Create a MobileNetV2 model architecture for plant disease classification

    The model uses transfer learning with the following custom top layers:
    - Global Average Pooling layer to reduce dimensionality
    - Dense layer (128 neurons, ReLU activation) for feature extraction
    - Softmax Output layer (38 neurons) for disease classification

    Args:
        num_classes: Number of output classes (default: 38)

    Returns:
        MobileNetV2 model with modified classifier
    """
    model = models.mobilenet_v2(weights=None)

    # Replace the classifier with custom dense layers
    # MobileNetV2 already has AdaptiveAvgPool2d before classifier (acts as Global Average Pooling)
    # Input features from the convolutional base: 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 128),  # Dense layer with 128 neurons
        nn.ReLU(),  # ReLU activation
        nn.Linear(
            128, num_classes
        ),  # Output layer (Softmax applied during loss computation)
    )

    return model


def load_model(model_path: str | None = None) -> nn.Module:
    """
    Load the trained MobileNetV2 model for plant disease classification

    Args:
        model_path: Optional custom path to model file

    Returns:
        Loaded PyTorch model in eval mode (CPU only)

    Raises:
        FileNotFoundError: If model file does not exist
        RuntimeError: If model loading fails
    """
    if model_path is None:
        model_path = MODEL_PATH

    # Resolve the path
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        # Create model architecture
        model = create_model(num_classes=len(CLASS_NAMES))

        # Load weights (CPU only)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

        # Handle different state dict formats
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        elif "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        else:
            model.load_state_dict(state_dict)

        # Set to evaluation mode
        model.eval()

        logger.info(f"Model loaded successfully from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
