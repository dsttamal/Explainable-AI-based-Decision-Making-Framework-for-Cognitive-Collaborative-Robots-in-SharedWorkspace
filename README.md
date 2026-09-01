# 🎬 Action Recognition API with LIME

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00.svg)](https://www.tensorflow.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-ready FastAPI web service that provides **action recognition predictions** from images with **explainable AI** using LIME (Local Interpretable Model-agnostic Explanations).

<div align="center">

### 🎯 Predict Actions | 🔍 Explain Decisions | 🚀 Deploy Easily

</div>

---

## ✨ Features

- 🎯 **Action Recognition**: Identify human actions from images across 10 distinct classes
- 🔍 **Explainable AI**: Generate LIME visualizations showing which image regions influenced predictions
- 🚀 **FastAPI Backend**: High-performance RESTful API with automatic interactive documentation
- 📱 **Web Interface**: Ready-to-use HTML client for instant testing
- 📊 **Batch Processing**: Process multiple images simultaneously
- 🎨 **Visual Explanations**: Interactive heatmap overlays highlighting decision-making regions
- 🔌 **Easy Integration**: Simple API endpoints for seamless integration into your applications

---

## 📋 Table of Contents

- [Action Classes](#-action-classes)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Endpoints](#-api-endpoints)
- [Usage Examples](#-usage-examples)
- [Response Format](#-response-format)
- [Configuration](#-configuration)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏃 Action Classes

The model can recognize these 10 human actions:

| Action | Description |
|--------|-------------|
| 🤸 jumping | Person jumping or leaping |
| 📱 phoning | Using a phone or mobile device |
| 🎸 playinginstrument | Playing musical instruments |
| 📖 reading | Reading books, papers, or screens |
| 🚴 ridingbike | Riding a bicycle |
| 🏇 ridinghorse | Riding a horse |
| 🏃 running | Running or jogging |
| 📷 takingphoto | Taking photographs |
| 💻 usingcomputer | Working on a computer |
| 🚶 walking | Walking or strolling |

---

## 📦 Prerequisites

### Required Files

1. **Trained Model Files** (required in the project root directory):
   - `trained_action_recognition_model.keras` - The trained TensorFlow/Keras model
   - `model_metadata.json` - Model metadata and configuration

2. **Dataset** (optional, only needed for training):
   - Download the PASCAL VOC 2012 dataset from Kaggle:
   - 🔗 [PASCAL VOC 2012 Dataset](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset)
   - Extract to the `VOCdevkit/` directory in the project root

3. **System Requirements**:
   - Python 3.8 or higher
   - 4GB+ RAM recommended
   - 2GB+ disk space for dependencies

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd LIME
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- FastAPI 0.104.1
- TensorFlow 2.15.0
- LIME 0.2.0.1
- NumPy, Pillow, Matplotlib, and more

---

## ⚡ Quick Start

### Method 1: Using the Launcher Script (Recommended)

```bash
python run_api.py
```

### Method 2: Using Uvicorn Directly

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Method 3: Using the Batch Script (Windows)

```bash
start_api.bat
```

### Access the Application

Once started, the API will be available at:

- 🌐 **Main API**: http://localhost:8000
- 📚 **Interactive Docs** (Swagger UI): http://localhost:8000/docs
- 📖 **ReDoc Documentation**: http://localhost:8000/redoc
- 🎨 **Web Client**: Open `test_client.html` in your browser

---

## 🔌 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint with API information |
| GET | `/health` | Health check and server status |
| GET | `/model-info` | Model metrics and configuration |
| POST | `/predict` | Predict action from single image |
| POST | `/explain` | Generate LIME explanation with visualization |
| POST | `/batch-predict` | Predict actions for multiple images (max 10) |

### Detailed Endpoint Information

#### `GET /`
- **Description**: Root endpoint with API information
- **Response**: JSON with API status and available endpoints

#### `GET /health`
- **Description**: Health check endpoint
- **Response**: Server and model loading status

#### `GET /model-info`
- **Description**: Get detailed model information
- **Response**: Model accuracy, action classes, training metadata

#### `POST /predict`
- **Description**: Predict action from an uploaded image
- **Input**: 
  - `file` (form-data): Image file (JPG, PNG, GIF, WebP)
- **Response**: Predicted action with confidence scores

#### `POST /explain`
- **Description**: Generate LIME explanation with visual heatmap
- **Input**: 
  - `file` (form-data): Image file
  - `num_samples` (query, optional): Number of samples for LIME (default: 300)
- **Response**: Prediction with base64-encoded LIME visualization

#### `POST /batch-predict`
- **Description**: Process multiple images at once
- **Input**: 
  - `files` (form-data): Multiple image files (max 10)
- **Response**: Array of predictions for each image

---

## 💡 Usage Examples

### 🖥️ Using cURL

#### Basic Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

#### Get LIME Explanation

```bash
curl -X POST "http://localhost:8000/explain?num_samples=300" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

#### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch-predict" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg" \
     -F "files=@image3.jpg"
```

---

### 🐍 Using Python

#### Simple Prediction

```python
import requests

# Predict action
with open('your_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    result = response.json()
    print(f"Predicted action: {result['predicted_action']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

#### Get Explanation with LIME

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Get explanation
with open('your_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/explain',
        params={'num_samples': 300},
        files={'file': f}
    )
    result = response.json()
    
    # Decode and save LIME visualization
    lime_image_data = base64.b64decode(result['explanation']['lime_visualization'])
    lime_image = Image.open(BytesIO(lime_image_data))
    lime_image.save('lime_explanation.png')
    print("LIME explanation saved to lime_explanation.png")
```

#### Batch Processing

```python
import requests

# Process multiple images
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]

response = requests.post(
    'http://localhost:8000/batch-predict',
    files=files
)

for img_result in response.json()['results']:
    print(f"{img_result['filename']}: {img_result['predicted_action']}")
```

---

### 🌐 Using JavaScript (Fetch API)

#### Basic Prediction

```javascript
// Predict action from file input
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(`Predicted: ${result.predicted_action} (${(result.confidence * 100).toFixed(1)}%)`);
```

#### Display LIME Explanation

```javascript
// Get and display LIME explanation
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/explain?num_samples=300', {
    method: 'POST',
    body: formData
});

const result = await response.json();

// Display LIME image
const limeImage = document.createElement('img');
limeImage.src = `data:image/png;base64,${result.explanation.lime_visualization}`;
document.body.appendChild(limeImage);
```

---

## 📊 Response Format

### Prediction Response (`/predict`)

```json
{
    "filename": "running_person.jpg",
    "predicted_action": "running",
    "confidence": 0.8542,
    "top_predictions": [
        {
            "action": "running",
            "confidence": 0.8542
        },
        {
            "action": "walking",
            "confidence": 0.1203
        },
        {
            "action": "jumping",
            "confidence": 0.0255
        }
    ],
    "model_accuracy": "20.54%"
}
```

### Explanation Response (`/explain`)

```json
{
    "filename": "running_person.jpg",
    "explanation": {
        "top_predictions": [
            {
                "action": "running",
                "confidence": 0.8542
            },
            {
                "action": "walking",
                "confidence": 0.1203
            }
        ],
        "lime_visualization": "iVBORw0KGgoAAAANSUhEUgAA...base64_encoded_image...=="
    },
    "num_samples_used": 300,
    "model_accuracy": "20.54%"
}
```

### Batch Prediction Response (`/batch-predict`)

```json
{
    "results": [
        {
            "filename": "image1.jpg",
            "predicted_action": "running",
            "confidence": 0.8542,
            "top_predictions": [...]
        },
        {
            "filename": "image2.jpg",
            "predicted_action": "reading",
            "confidence": 0.7123,
            "top_predictions": [...]
        }
    ],
    "total_images": 2,
    "model_accuracy": "20.54%"
}
```

---

## ⚙️ Configuration

### Modifying Server Settings

Edit the following in [`app.py`](app.py):

```python
# Server configuration
HOST = "0.0.0.0"  # Change to "127.0.0.1" to restrict to localhost
PORT = 8000       # Change port number

# LIME configuration
DEFAULT_NUM_SAMPLES = 300  # Adjust LIME sample size (100-1000)

# Batch processing
MAX_BATCH_SIZE = 10  # Maximum images per batch request
```

### CORS Configuration

To allow cross-origin requests from specific domains, modify the CORS middleware in [`app.py`](app.py):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## ⚡ Performance

### Benchmarks

| Operation | Average Time | Notes |
|-----------|-------------|-------|
| Single Prediction | 1-2 seconds | Depends on image size |
| LIME Explanation | 30-60 seconds | Depends on `num_samples` parameter |
| Batch Prediction (10 images) | 10-20 seconds | Processed sequentially |

### Optimization Tips

- **Memory Usage**: ~2-4 GB RAM for loaded model
- **Supported Formats**: JPG, PNG, GIF, WebP
- **Image Processing**: All images auto-resized to 224x224 pixels
- **LIME Samples**: 
  - Lower values (100-200): Faster but less accurate explanations
  - Higher values (500-1000): Slower but more accurate explanations
  - Default (300): Good balance

### Development Mode

Run with auto-reload for development:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### ❌ Model Not Loading

**Problem**: API fails to start with model loading error

**Solutions**:
```bash
# Verify required files exist
ls trained_action_recognition_model.keras
ls model_metadata.json

# Check file permissions
# Ensure files are not corrupted
# Re-download or re-train the model if necessary
```

#### ❌ Memory Errors

**Problem**: Out of memory during processing

**Solutions**:
- Reduce image size before uploading
- Close other memory-intensive applications
- Restart the API server
- Consider using a machine with more RAM

#### ❌ Slow LIME Explanations

**Problem**: LIME generation taking too long

**Solutions**:
```bash
# Use fewer samples (faster but less accurate)
curl -X POST "http://localhost:8000/explain?num_samples=100" \
     -F "file=@image.jpg"

# Use smaller images (auto-resized to 224x224 anyway)
# Consider caching frequently explained images
```

#### ❌ CORS Errors in Browser

**Problem**: Browser blocking API requests

**Solutions**:
1. Ensure API is running: `http://localhost:8000`
2. Check browser console for specific errors
3. Verify CORS settings in [`app.py`](app.py)
4. Use the provided [`test_client.html`](test_client.html) which is pre-configured

#### ❌ Import Errors

**Problem**: Missing Python packages

**Solutions**:
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# If specific package fails, try:
pip install --upgrade tensorflow
pip install --upgrade lime
```

### HTTP Status Codes

| Code | Meaning | Typical Cause |
|------|---------|---------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid file type or parameters |
| 422 | Validation Error | Missing required fields |
| 500 | Server Error | Model loading or processing failure |

---

## 📁 Project Structure

```
LIME/
├── app.py                                    # Main FastAPI application
├── run_api.py                               # API launcher script
├── start_api.bat                            # Windows batch starter
├── model_loader.py                          # Model loading utilities
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
├── trained_action_recognition_model.keras   # Trained model (required)
├── trained_action_recognition_model.h5      # Alternative model format
├── model_metadata.json                      # Model configuration (required)
├── best_action_model.h5                     # Additional model variant
├── best_multiclass_model.h5                 # Multiclass model variant
├── test_client.html                         # Web testing interface
├── frontend/                                # Frontend assets
│   ├── index.html                          # Main frontend page
│   ├── script.js                           # Frontend JavaScript
│   └── styles.css                          # Frontend styles
├── notebooks/                               # Training notebooks
│   ├── LIME.ipynb                          # Main LIME notebook
│   ├── LIME 2.ipynb                        # LIME variants
│   └── LIME with image generate.ipynb      # LIME with image generation
└── VOCdevkit/                               # Training dataset
    └── VOC2012/                            # PASCAL VOC 2012 dataset
        ├── Annotations/                    # XML annotations
        ├── JPEGImages/                     # Training images
        ├── ImageSets/                      # Dataset splits
        ├── SegmentationClass/              # Segmentation masks
        └── SegmentationObject/             # Object segmentation
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to new functions
- Update README for new features
- Test your changes thoroughly
- Include example usage in PR description

---

## 📊 Model Information

- **Architecture**: Custom CNN with regularization and dropout
- **Input Size**: 224×224 RGB images
- **Output**: 10 action classes with probability distribution
- **Training Dataset**: PASCAL VOC 2012 action dataset
  - 📥 [Download from Kaggle](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset)
- **Accuracy**: ~20.5% (baseline for 10-class action recognition)
- **Framework**: TensorFlow 2.15 / Keras

### Model Training

To retrain the model or view training details:

1. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset)
2. **Extract to** `VOCdevkit/` directory in the project root
3. **Open the training notebooks**:
   - [`LIME.ipynb`](LIME.ipynb) - Main training notebook
   - [`LIME 2.ipynb`](LIME%202.ipynb) - Alternative training approaches
   - [`LIME with image generate.ipynb`](LIME%20with%20image%20generate.ipynb) - Data augmentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Dataset License

The PASCAL VOC 2012 dataset is used for model training. Please refer to the [PASCAL VOC website](http://host.robots.ox.ac.uk/pascal/VOC/) for dataset licensing information.

---

## 🙏 Acknowledgments

- **LIME**: [Local Interpretable Model-agnostic Explanations](https://github.com/marcotcr/lime)
- **FastAPI**: [Modern web framework for building APIs](https://fastapi.tiangolo.com/)
- **TensorFlow**: [Machine learning framework](https://www.tensorflow.org/)
- **PASCAL VOC**: [Visual Object Classes Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)

---

## 📮 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/dsttamal/Explainable-AI-based-Decision-Making-Framework-for-Cognitive-Collaborative-Robots-in-SharedWorkspace/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dsttamal/Explainable-AI-based-Decision-Making-Framework-for-Cognitive-Collaborative-Robots-in-SharedWorkspace/discussions)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ for the AI and Computer Vision community

</div>
