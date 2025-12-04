# 🌿 Agrivision - Plant Disease Detection API

A FastAPI-based REST API for detecting plant diseases using a MobileNetV2 deep learning model. Upload an image of a plant leaf, and the API will predict one of 38 plant-disease classes with a confidence score.

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Windows (WSL2)](#windows-wsl2)
  - [Ubuntu/Debian](#ubuntudebian)
  - [Arch Linux](#arch-linux)
- [Quick Start](#-quick-start)
- [Development Workflow](#-development-workflow)
  - [Model Training](#1-model-training-vs-code-dev-container)
  - [API Development](#2-api-development)
  - [Testing Your Model](#3-testing-your-model)
- [Production Deployment](#-production-deployment)
- [API Reference](#-api-reference)
- [Model Information](#-model-information)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- **Plant Disease Classification**: Detects 38 different plant disease classes
- **Fast Inference**: Uses MobileNetV2 for efficient CPU-based inference
- **Secure**: File validation, size limits, and image integrity checks
- **Docker Ready**: Development, training, and production Docker configurations
- **VS Code Dev Container**: Integrated development environment for model training
- **RESTful API**: Clean FastAPI endpoints with automatic documentation

---

## 📁 Project Structure

```
agrivision_cv_model/
├── .devcontainer/
│   └── devcontainer.json       # VS Code Dev Container config
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI application
│   ├── model/
│   │   ├── loader.py           # Model loading utilities
│   │   └── preprocess.py       # Image preprocessing
│   └── utils/
│       └── security.py         # Security validation
├── docker/
│   ├── Dockerfile.dev          # API development
│   ├── Dockerfile.prod         # Production deployment
│   ├── Dockerfile.training     # Model training environment
│   └── docker-compose.yml      # All services configuration
├── notebooks/
│   └── 01_train_model.ipynb    # Model training notebook
├── data/
│   ├── raw/                    # Raw dataset (PlantVillage)
│   └── processed/              # Preprocessed data
├── models/                     # Trained model files (.pt)
├── requirements.txt            # API dependencies
├── requirements-training.txt   # Training dependencies
├── .gitignore
├── .env.example
└── README.md
```

---

## 🔧 Installation

### Prerequisites

- **Git**: For cloning the repository
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **VS Code** (optional): For Dev Container support
  - Extension: [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

---

### Windows (WSL2)

#### Step 1: Install WSL2

```powershell
# Run in PowerShell as Administrator
wsl --install

# Restart your computer, then set WSL2 as default
wsl --set-default-version 2
```

#### Step 2: Install Docker Desktop

1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Run the installer
3. During setup, ensure **"Use WSL 2 based engine"** is checked
4. After installation, open Docker Desktop
5. Go to **Settings → Resources → WSL Integration**
6. Enable integration with your WSL distro (Ubuntu recommended)

#### Step 3: Install VS Code + Extensions

1. Download [VS Code](https://code.visualstudio.com/)
2. Install extensions:
   - **WSL** (ms-vscode-remote.remote-wsl)
   - **Dev Containers** (ms-vscode-remote.remote-containers)

#### Step 4: Clone and Setup

```bash
# Open WSL terminal (Ubuntu)
cd ~
git clone <repository-url> agrivision_cv_model
cd agrivision_cv_model

# Verify Docker works in WSL
docker --version
docker compose version
```

---

### Ubuntu/Debian

#### Step 1: Install Docker Engine

```bash
# Update package index
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to docker group (run without sudo)
sudo usermod -aG docker $USER

# Apply group changes (or log out and back in)
newgrp docker

# Verify installation
docker --version
docker compose version
```

#### Step 2: Install VS Code (Optional)

```bash
# Download and install
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list
sudo apt update
sudo apt install -y code

# Install Dev Containers extension
code --install-extension ms-vscode-remote.remote-containers
```

#### Step 3: Clone Repository

```bash
git clone <repository-url> agrivision_cv_model
cd agrivision_cv_model
```

---

### Arch Linux

#### Step 1: Install Docker

```bash
# Install Docker and Docker Compose
sudo pacman -S docker docker-compose

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group
sudo usermod -aG docker $USER

# Apply group changes (or log out and back in)
newgrp docker

# Verify installation
docker --version
docker compose version
```

#### Step 2: Install VS Code (Optional)

```bash
# From AUR (using yay or paru)
yay -S visual-studio-code-bin

# Or use the open-source version
sudo pacman -S code

# Install Dev Containers extension
code --install-extension ms-vscode-remote.remote-containers
```

#### Step 3: Clone Repository

```bash
git clone <repository-url> agrivision_cv_model
cd agrivision_cv_model
```

---

## 🚀 Quick Start

### Option A: Just Run the API (if you have a trained model)

```bash
# 1. Place your model file
cp /path/to/mobilenetv2_agrivision.pt models/

# 2. Start the API
cd docker
docker-compose up agrivision-api --build

# 3. Test it
curl http://localhost:8000/health
```

### Option B: Train a Model First (recommended for new users)

```bash
# 1. Build the training container
cd docker
docker-compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g) agrivision-training

# 2. Start the training container
docker-compose up agrivision-training -d

# 3. Open VS Code and attach to container
code ..
# Then: F1 → "Dev Containers: Reopen in Container"

# 4. Train your model in the notebook
# Open notebooks/01_train_model.ipynb
```

---

## 💻 Development Workflow

### 1. Model Training (VS Code Dev Container)

This is where you train your MobileNetV2 model using Jupyter notebooks.

#### Start Training Environment

```bash
cd docker

# Build with your user permissions (important for file editing)
docker-compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g) agrivision-training

# Start the container
docker-compose up agrivision-training -d
```

#### Open in VS Code Dev Container

1. Open VS Code in the project folder: `code .`
2. Press `F1` → **"Dev Containers: Reopen in Container"**
3. Wait for the container to build and attach
4. Open `notebooks/01_train_model.ipynb`
5. Select kernel: **"Python 3.11"** or **"Agrivision (Python 3.11)"**

#### Download Dataset

Download the PlantVillage dataset:
- **Kaggle**: https://www.kaggle.com/datasets/emmarex/plantdisease

Extract to `data/raw/plantvillage/`:
```
data/raw/plantvillage/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
└── ... (38 class folders)
```

#### Train and Export Model

After training in the notebook, save your model:

```python
# Save model weights
torch.save(model.state_dict(), "../models/mobilenetv2_agrivision.pt")
print("Model saved!")
```

#### Install Additional Packages

Inside the Dev Container terminal or notebook:

```bash
pip install <package-name>
```

Or add to `requirements-training.txt` and rebuild:

```bash
docker-compose build agrivision-training
```

---

### 2. API Development

Run the FastAPI server with hot-reload for development.

#### Start API Server

```bash
cd docker
docker-compose up agrivision-api --build
```

#### Access Points

| URL | Description |
|-----|-------------|
| http://localhost:8000 | API base URL |
| http://localhost:8000/docs | Swagger UI (interactive docs) |
| http://localhost:8000/redoc | ReDoc documentation |
| http://localhost:8000/health | Health check endpoint |

#### Development Features

- **Hot Reload**: Code changes apply automatically
- **Volume Mount**: Edit files locally, changes reflect in container
- **Logs**: `docker-compose logs -f agrivision-api`

---

### 3. Testing Your Model

#### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict disease
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/plant_leaf.jpg"
```

#### Using Python

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("plant_leaf.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
# {"class": "Tomato___Late_blight", "confidence": 0.9523}
```

#### Using the Swagger UI

1. Open http://localhost:8000/docs
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Upload an image file
5. Click "Execute"

---

## 🏭 Production Deployment

### Build Production Image

```bash
# From project root
docker build -f docker/Dockerfile.prod -t agrivision:latest .
```

### Run Production Container

```bash
docker run -d \
  --name agrivision-prod \
  -p 8000:8000 \
  --restart unless-stopped \
  agrivision:latest
```

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
services:
  agrivision:
    build:
      context: .
      dockerfile: docker/Dockerfile.prod
    container_name: agrivision-prod
    ports:
      - "8000:8000"
    restart: always
    deploy:
      resources:
        limits:
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Production Checklist

- [ ] Model file exists in `models/mobilenetv2_agrivision.pt`
- [ ] Production image built successfully
- [ ] Health check passing
- [ ] Resource limits configured
- [ ] Reverse proxy (nginx/traefik) for HTTPS
- [ ] Rate limiting enabled
- [ ] Logging configured

---

## 📡 API Reference

### Endpoints

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "version": "1.0.0"
}
```

#### POST /predict

Predict plant disease from image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file, JPEG/PNG, max 5MB)

**Success Response (200):**
```json
{
  "class": "Tomato___Late_blight",
  "confidence": 0.9523
}
```

### Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `EMPTY_FILE` | Empty file received |
| 400 | `CORRUPT_IMAGE` | Image file is corrupt |
| 400 | `IMAGE_TOO_SMALL` | Image dimensions too small (< 32x32) |
| 400 | `NOT_PLANT_IMAGE` | Image doesn't appear to contain a plant |
| 413 | `FILE_TOO_LARGE` | File exceeds 5MB limit |
| 415 | `INVALID_MIME_TYPE` | Invalid file type (not JPEG/PNG) |
| 415 | `INVALID_EXTENSION` | Invalid file extension |
| 503 | `MODEL_NOT_LOADED` | Model file not found |

**Error Response Format:**
```json
{
  "error": "FILE_TOO_LARGE",
  "message": "File too large. Maximum size: 5MB. Received: 7.23MB"
}
```

---

## 🌱 Model Information

### Supported Classes (38)

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

### Model Architecture

- **Base**: MobileNetV2 (pretrained on ImageNet)
- **Classifier**: Custom head with 38 output classes
- **Input Size**: 224x224 RGB
- **Normalization**: ImageNet mean/std

---

## 🔒 Security Features

- **MIME Type Validation**: Only accepts `image/jpeg` and `image/png`
- **File Extension Check**: Validates `.jpg`, `.jpeg`, `.png` extensions
- **File Size Limit**: Maximum 5MB per upload
- **Image Integrity**: Verifies image is not corrupt using PIL
- **Plant Detection**: Basic color analysis to filter non-plant images
- **Non-root User**: Docker containers run as non-root user

---

## 🔍 Troubleshooting

### Docker Permission Denied

```
Permission denied while trying to connect to Docker daemon
```

**Solution:**
```bash
sudo usermod -aG docker $USER
newgrp docker  # or log out and back in
```

### Cannot Edit Files in Dev Container

**Solution:** Rebuild with your UID/GID:
```bash
cd docker
docker-compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g) agrivision-training
docker-compose up agrivision-training -d
```

### Model Not Loading

```
Model file not found at models/mobilenetv2_agrivision.pt
```

**Solution:** Ensure the model file is in the `models/` directory.

### Port Already in Use

```
Error: Port 8000 is already in use
```

**Solution:**
```bash
# Find what's using the port
sudo lsof -i :8000

# Stop it or change port in docker-compose.yml
docker-compose down
```

### WSL2: Docker Not Working

**Solution:**
1. Open Docker Desktop
2. Go to Settings → Resources → WSL Integration
3. Enable integration with your distro
4. Restart Docker Desktop

### Out of Memory During Training

**Solution:** Reduce batch size in notebook:
```python
CONFIG["batch_size"] = 16  # or lower
```

Or increase Docker memory limit in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 12G
```

---

## 📊 Useful Commands

| Task | Command |
|------|---------|
| Start training env | `docker-compose up agrivision-training -d` |
| Start API | `docker-compose up agrivision-api -d` |
| Start both | `docker-compose up -d` |
| View logs | `docker-compose logs -f` |
| Stop all | `docker-compose down` |
| Rebuild training | `docker-compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g) agrivision-training` |
| Rebuild API | `docker-compose build agrivision-api` |
| Shell into container | `docker-compose exec agrivision-training bash` |
| Run TensorBoard | `tensorboard --logdir=runs --host=0.0.0.0` (inside container, port 6006) |

---

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Support

For issues and questions, please create a GitHub issue.
