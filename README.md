# 🌿 Agrivision - Plant Disease Detection API

A FastAPI-based REST API for detecting plant diseases using a MobileNetV2 deep learning model. Upload an image of a plant leaf, and the API will predict one of 38 plant-disease classes with a confidence score.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [Production Deployment](#production-deployment)
- [API Usage](#api-usage)
- [Model Information](#model-information)
- [Security Features](#security-features)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Plant Disease Classification**: Detects 38 different plant disease classes
- **Fast Inference**: Uses MobileNetV2 for efficient CPU-based inference
- **Secure**: File validation, size limits, and image integrity checks
- **Docker Ready**: Development and production Docker configurations
- **RESTful API**: Clean FastAPI endpoints with automatic documentation

## 📁 Project Structure

```
agrivision/
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI application
│   ├── model/
│   │   ├── loader.py           # Model loading utilities
│   │   └── preprocess.py       # Image preprocessing
│   └── utils/
│       └── security.py         # Security validation
├── docker/
│   ├── Dockerfile.dev          # Development Dockerfile
│   ├── Dockerfile.prod         # Production Dockerfile
│   └── docker-compose.yml      # Docker Compose config
├── models/                     # Model files (add .pt file here)
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 🔧 Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher (included with Docker Desktop)
- **Model File**: `mobilenetv2_agrivision.pt` (see [Adding the Model](#adding-the-model))

### Installing Docker

#### Ubuntu/Debian
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER

# Log out and log back in for group changes to take effect
```

#### Windows/macOS
Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd agrivision
```

### 2. Add the Model File
Place your trained MobileNetV2 model file in the `models/` directory:
```bash
# Copy your model file
cp /path/to/your/mobilenetv2_agrivision.pt models/
```

### 3. Start Development Server
```bash
cd docker
docker compose up --build
```

The API will be available at `http://localhost:8000`

### 4. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Predict (replace with your image path)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/plant_image.jpg"
```

## 💻 Development Setup

### Using Docker Compose (Recommended)

1. **Navigate to docker directory**:
   ```bash
   cd docker
   ```

2. **Build and start the development container**:
   ```bash
   docker compose up --build
   ```

3. **Access the API**:
   - API: http://localhost:8000
   - Interactive Docs (Swagger): http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

4. **Hot Reload**: The development server automatically reloads when you modify source files.

5. **Stop the container**:
   ```bash
   docker compose down
   ```

### Running Commands in Container

```bash
# Execute bash in running container
docker compose exec agrivision-api bash

# Run Python commands
docker compose exec agrivision-api python -c "print('Hello')"
```

### Development Tips

- Source code is mounted as a volume, so changes reflect immediately
- Check logs: `docker compose logs -f`
- Rebuild after changing requirements.txt: `docker compose up --build`

## 🏭 Production Deployment

### Building Production Image

1. **Build the production image**:
   ```bash
   docker build -f docker/Dockerfile.prod -t agrivision:latest .
   ```

2. **Run the production container**:
   ```bash
   docker run -d \
     --name agrivision-prod \
     -p 8000:8000 \
     -v $(pwd)/models:/app/models:ro \
     agrivision:latest
   ```

### Production Docker Compose

Create a `docker-compose.prod.yml`:
```yaml
services:
  agrivision-api:
    build:
      context: .
      dockerfile: docker/Dockerfile.prod
    container_name: agrivision-prod
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    restart: always
    deploy:
      resources:
        limits:
          memory: 2G
```

Run with:
```bash
docker compose -f docker-compose.prod.yml up -d
```

### Production Checklist

- [ ] Model file is in place (`models/mobilenetv2_agrivision.pt`)
- [ ] Resource limits are configured
- [ ] Health checks are passing
- [ ] Logs are being collected
- [ ] Reverse proxy (nginx) configured for HTTPS
- [ ] Rate limiting enabled

## 📡 API Usage

### Endpoints

#### Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "version": "1.0.0"
}
```

#### Predict Disease
```http
POST /predict
Content-Type: multipart/form-data
```

**Request**:
- `file`: Image file (JPEG or PNG, max 5MB)

**Success Response** (200):
```json
{
  "class": "Tomato___Late_blight",
  "confidence": 0.9523
}
```

### Example Usage

#### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tomato_leaf.jpg"
```

#### Python
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("tomato_leaf.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

#### JavaScript (fetch)
```javascript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});
const result = await response.json();
console.log(result);
```

### Error Responses

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | EMPTY_FILE | Empty file received |
| 400 | CORRUPT_IMAGE | Image file is corrupt |
| 400 | IMAGE_TOO_SMALL | Image dimensions too small |
| 400 | NOT_PLANT_IMAGE | Image doesn't appear to contain a plant |
| 413 | FILE_TOO_LARGE | File exceeds 5MB limit |
| 415 | INVALID_MIME_TYPE | Invalid file type (not JPEG/PNG) |
| 415 | INVALID_EXTENSION | Invalid file extension |
| 503 | MODEL_NOT_LOADED | Model file not found |

## 🌱 Model Information

### Supported Classes (38)

The model can classify the following plant-disease combinations:

| Plant | Conditions |
|-------|------------|
| Apple | Apple scab, Black rot, Cedar apple rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery mildew, Healthy |
| Corn (Maize) | Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy |
| Grape | Black rot, Esca (Black Measles), Leaf blight, Healthy |
| Orange | Haunglongbing (Citrus greening) |
| Peach | Bacterial spot, Healthy |
| Pepper (Bell) | Bacterial spot, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch, Healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy |

### Adding the Model

1. Train your MobileNetV2 model using the PlantVillage dataset
2. Save the model weights:
   ```python
   torch.save(model.state_dict(), 'mobilenetv2_agrivision.pt')
   # Or with additional metadata:
   torch.save({
       'model_state_dict': model.state_dict(),
       'class_names': class_names,
   }, 'mobilenetv2_agrivision.pt')
   ```
3. Place the file in the `models/` directory

## 🔒 Security Features

- **MIME Type Validation**: Only accepts image/jpeg and image/png
- **File Extension Check**: Validates .jpg, .jpeg, .png extensions
- **File Size Limit**: Maximum 5MB per upload
- **Image Integrity**: Verifies image is not corrupt
- **Plant Detection**: Basic color analysis to filter non-plant images
- **Non-root User**: Docker containers run as non-root user

## 🔍 Troubleshooting

### Model Not Loading
```
Model file not found at models/mobilenetv2_agrivision.pt
```
**Solution**: Ensure the model file is placed in the `models/` directory.

### Permission Denied (Docker)
```
Permission denied while trying to connect to Docker daemon
```
**Solution**: Add your user to the docker group:
```bash
sudo usermod -aG docker $USER
# Log out and log back in
```

### Port Already in Use
```
Error: Port 8000 is already in use
```
**Solution**: Stop the other service or use a different port:
```bash
docker compose down
# Or change the port in docker-compose.yml
```

### Out of Memory
**Solution**: Increase Docker memory limit in Docker Desktop settings, or adjust resource limits in docker-compose.yml.

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Support

For issues and questions, please create a GitHub issue.
