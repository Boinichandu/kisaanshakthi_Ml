"""
Agrivision - Plant Disease Detection API
FastAPI application with /health and /predict endpoints
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging

from src.model.loader import load_model, get_class_names
from src.model.preprocess import preprocess_image, predict
from src.utils.security import (
    validate_file_type,
    validate_file_size,
    validate_image_integrity,
    validate_plant_image,
    SecurityError,
    MAX_FILE_SIZE,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agrivision API",
    description="Plant Disease Detection API using MobileNetV2",
    version="1.0.0",
)

# Global model variable
model = None
class_names = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, class_names
    try:
        model = load_model()
        class_names = get_class_names()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Model will be None, endpoints will handle this


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Returns: status of the API and model loading state
    """
    model_status = "loaded" if model is not None else "not_loaded"
    return {"status": "healthy", "model_status": model_status, "version": "1.0.0"}


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image

    Args:
        file: Uploaded image file (jpg/jpeg/png, max 5MB)

    Returns:
        JSON with predicted class and confidence score
    """
    # Check if model is loaded
    if model is None or class_names is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "MODEL_NOT_LOADED",
                "message": "Model is not loaded. Please ensure the model file exists at models/mobilenetv2_agrivision.pt",
            },
        )

    try:
        # Read file content
        content = await file.read()

        # Security validations
        try:
            # Validate file type (MIME type)
            validate_file_type(file.filename, file.content_type)

            # Validate file size
            validate_file_size(content)

            # Validate image integrity (not corrupt)
            image = validate_image_integrity(content)

            # Validate it's a plant image
            validate_plant_image(image)

        except SecurityError as e:
            raise HTTPException(
                status_code=e.status_code,
                detail={"error": e.error_code, "message": e.message},
            )

        # Preprocess image
        tensor = preprocess_image(image)

        # Run prediction
        predicted_class, confidence = predict(model, tensor, class_names)

        logger.info(f"Prediction: {predicted_class} with confidence {confidence:.4f}")

        return JSONResponse(
            status_code=200,
            content={"class": predicted_class, "confidence": round(confidence, 4)},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PREDICTION_ERROR",
                "message": f"An error occurred during prediction: {str(e)}",
            },
        )
    finally:
        await file.close()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
    )
