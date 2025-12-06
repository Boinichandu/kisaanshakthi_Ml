# Agrivision

Plant disease detection API using deep learning. Built with FastAPI and MobileNetV2, trained on the PlantVillage dataset to classify 38 different plant disease categories.

## Table of Contents

- [Overview](#overview)
- [Supported Classes](#supported-classes)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Project Structure](#project-structure)

## Overview

Agrivision is a computer vision model that identifies plant diseases from leaf images. The model uses MobileNetV2 architecture optimized for inference speed while maintaining high accuracy.

Key features:
- Classifies 38 plant disease categories across 14 crop types
- RESTful API with health check and prediction endpoints
- Input validation and security checks
- Docker support for development and production

## Supported Classes

The model can identify diseases in the following crops:

| Crop | Conditions |
|------|------------|
| Apple | Apple scab, Black rot, Cedar apple rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery mildew, Healthy |
| Corn | Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy |
| Grape | Black rot, Esca (Black Measles), Leaf blight, Healthy |
| Orange | Huanglongbing (Citrus greening) |
| Peach | Bacterial spot, Healthy |
| Pepper | Bacterial spot, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch, Healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy |

## Requirements

- Python 3.11+
- PyTorch 2.1+
- Docker (optional, for containerized deployment)

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/KevinTheDuck/agrivision-cv-model.git
cd agrivision-cv-model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure the model file exists at `models/mobilenetv2_agrivision.pt`.

5. Start the server:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Docker Setup

Build and run using Docker Compose:
```bash
cd docker
docker compose up agrivision-api
```

The API will be available at `http://localhost:8000`.

## Usage

### Basic Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@leaf_image.jpg"
```

### Python Example

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("leaf_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Response Format

```json
{
  "class": "Tomato___Late_blight",
  "confidence": 0.9542
}
```

## API Reference

### Health Check

Check API status and model loading state.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "version": "1.0.0"
}
```

| Field | Description |
|-------|-------------|
| status | API health status |
| model_status | `loaded` or `not_loaded` |
| version | API version |

### Predict

Upload an image to get disease prediction.

**Endpoint:** `POST /predict`

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` - Image file (JPG, JPEG, or PNG)

**Constraints:**
- Maximum file size: 5MB
- Supported formats: JPG, JPEG, PNG
- Minimum image dimensions: 32x32 pixels

**Success Response (200):**
```json
{
  "class": "Apple___Black_rot",
  "confidence": 0.8765
}
```

**Error Responses:**

| Code | Error | Description |
|------|-------|-------------|
| 400 | INVALID_FILE_TYPE | Unsupported file format |
| 400 | FILE_TOO_LARGE | File exceeds 5MB limit |
| 400 | INVALID_IMAGE | Corrupt or invalid image |
| 400 | NOT_PLANT_IMAGE | Image does not appear to contain a plant |
| 503 | MODEL_NOT_LOADED | Model file not found or failed to load |
| 500 | PREDICTION_ERROR | Internal prediction error |

## Deployment

### Development

Run with hot reload enabled:
```bash
docker compose up agrivision-api
```

Or locally:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Production

Build the production image:
```bash
docker build -f docker/Dockerfile.prod -t agrivision:latest .
```

Run the container:
```bash
docker run -d -p 8000:8000 --name agrivision agrivision:latest
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| PYTHONPATH | Python module path | /app |
| PYTHONUNBUFFERED | Disable output buffering | 1 |

### Training

To train or fine-tune the model:
```bash
docker compose up agrivision-training
```

This container includes GPU support and TensorBoard on port 6006.

## Project Structure

```
agrivision-cv-model/
├── data/
│   ├── processed/          # Preprocessed datasets
│   └── raw/                 # Raw PlantVillage dataset
├── docker/
│   ├── docker-compose.yml   # Docker services configuration
│   ├── Dockerfile.dev       # Development image
│   ├── Dockerfile.prod      # Production image
│   └── Dockerfile.training  # Training environment
├── models/
│   └── mobilenetv2_agrivision.pt  # Trained model weights
├── notebooks/
│   └── 01_train_model.ipynb # Training notebook
├── src/
│   ├── api/
│   │   └── app.py           # FastAPI application
│   ├── model/
│   │   ├── loader.py        # Model loading utilities
│   │   └── preprocess.py    # Image preprocessing
│   └── utils/
│       └── security.py      # Input validation
├── requirements.txt         # Production dependencies
└── requirements-training.txt # Training dependencies
```

## License

See LICENSE file for details.
