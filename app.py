from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
import json
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from lime import lime_image
from tensorflow.keras.preprocessing import image
import os
import tempfile
from typing import List, Dict, Any
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Server configuration
HOST = "0.0.0.0"  # Allow external connections
PORT = 8000       # Default port

# Global variables for model and metadata
model = None
metadata = None
action_classes = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, metadata, action_classes
    
    print("🔄 Loading model on startup...")
    model, metadata = load_trained_model()
    if model is None or metadata is None:
        print("❌ Failed to load model on startup!")
        print("Make sure 'trained_action_recognition_model.keras' and 'model_metadata.json' exist")
    else:
        action_classes = metadata['action_classes']
        print(f"🚀 API started with {len(action_classes)} action classes")
        print(f"📊 Model accuracy: {metadata['model_accuracy']*100:.2f}%")
        print(f"🌐 Server running on http://{HOST}:{PORT}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Action Recognition API with LIME",
    description="AI-powered action recognition with explainable AI using LIME",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve the main HTML file at root
@app.get("/")
async def serve_frontend():
    """Serve the main frontend HTML file"""
    return FileResponse("frontend/index.html")



def load_trained_model(model_path='trained_action_recognition_model.keras', 
                      metadata_path='model_metadata.json'):
    """Load the trained action recognition model and metadata"""
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded from {metadata_path}")
        
        return model, metadata
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def predict_single_image(model, metadata, img_array):
    """Predict action for a single image array"""
    # Normalize to [0,1]
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    
    # Add batch dimension if needed
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    action_classes = metadata['action_classes']
    predicted_action = action_classes[predicted_class_idx]
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[::-1][:3]
    top_predictions = []
    for idx in top_indices:
        top_predictions.append({
            'action': action_classes[idx],
            'confidence': float(predictions[0][idx])
        })
    
    return predicted_action, confidence, top_predictions

def generate_lime_explanation_api(model, metadata, img_array, num_samples=300):
    """Generate LIME explanation for API response"""
    
    def predict_fn(images):
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        if images.max() > 1.0:
            images = images / 255.0
        return model.predict(images, verbose=0)
    
    # Prepare image for LIME
    if len(img_array.shape) == 4:
        img_for_lime = img_array[0]
    else:
        img_for_lime = img_array.copy()
    
    if img_for_lime.max() <= 1.0:
        img_for_lime = (img_for_lime * 255).astype(np.uint8)
    else:
        img_for_lime = img_for_lime.astype(np.uint8)
    
    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate explanation
    explanation = explainer.explain_instance(
        img_for_lime.astype('double'),
        predict_fn,
        top_labels=3,
        hide_color=0,
        num_samples=num_samples
    )
    
    # Get predictions
    predictions = predict_fn(img_for_lime.astype('float32') / 255.0)
    top_classes = np.argsort(predictions[0])[::-1][:3]
    
    # Generate visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_for_lime)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # LIME explanation for top prediction
    if len(explanation.top_labels) > 0:
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            top_label, positive_only=True, num_features=10, hide_rest=False
        )
        
        axes[1].imshow(temp)
        class_name = metadata['action_classes'][top_label]
        confidence = predictions[0][top_label] * 100
        axes[1].set_title(f'LIME: {class_name}\n({confidence:.1f}%)')
        axes[1].axis('off')
        
        # Mask
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title(f'Importance Mask')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    # Prepare explanation data
    explanation_data = {
        'top_predictions': [
            {
                'action': metadata['action_classes'][idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_classes
        ],
        'lime_visualization': img_base64
    }
    
    return explanation_data



@app.get("/")
async def root():
    """Serve the main frontend page"""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return {"message": "AI Action Recognition API is running", "status": "healthy", "note": "Frontend not found - API only mode"}
    
@app.get("/info")
async def api_info():
    """Root endpoint with API information"""
    if model is None:
        return {"error": "Model not loaded. Please check model files."}
    
    return {
        "message": "Action Recognition API with LIME",
        "status": "ready",
        "model_accuracy": f"{metadata['model_accuracy']*100:.2f}%",
        "action_classes": action_classes,
        "endpoints": {
            "/predict": "POST - Upload image for action prediction",
            "/explain": "POST - Upload image for LIME explanation",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "metadata_loaded": metadata is not None,
        "server": {
            "host": HOST,
            "port": PORT,
            "url": f"http://{HOST}:{PORT}"
        },
        "model_info": {
            "accuracy": f"{metadata['model_accuracy']*100:.2f}%" if metadata else None,
            "classes": len(action_classes) if action_classes else 0,
            "version": "1.0"
        }
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if metadata is None:
        raise HTTPException(status_code=500, detail="Model metadata not loaded")
    
    return {
        "model_accuracy": f"{metadata['model_accuracy']*100:.2f}%",
        "num_classes": metadata['num_classes'],
        "action_classes": metadata['action_classes'],
        "input_shape": metadata['input_shape'],
        "training_samples": metadata['training_samples'],
        "validation_samples": metadata['validation_samples'],
        "architecture": metadata['model_architecture']
    }

@app.post("/predict")
async def predict_action(file: UploadFile = File(...)):
    """Predict action from uploaded image"""
    if model is None or metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Make prediction
        predicted_action, confidence, top_predictions = predict_single_image(
            model, metadata, img_array
        )
        
        return {
            "filename": file.filename,
            "predicted_action": predicted_action,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "model_accuracy": f"{metadata['model_accuracy']*100:.2f}%"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/explain")
async def explain_prediction(file: UploadFile = File(...), num_samples: int = 300):
    """Generate LIME explanation for uploaded image"""
    if model is None or metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate num_samples
    if num_samples < 100 or num_samples > 1000:
        raise HTTPException(status_code=400, detail="num_samples must be between 100 and 1000")
    
    try:
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Generate LIME explanation
        explanation_data = generate_lime_explanation_api(
            model, metadata, img_array, num_samples
        )
        
        return {
            "filename": file.filename,
            "explanation": explanation_data,
            "num_samples_used": num_samples,
            "model_accuracy": f"{metadata['model_accuracy']*100:.2f}%"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """Predict actions for multiple images"""
    if model is None or metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "error": "File must be an image"
            })
            continue
        
        try:
            # Read and process image
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)
            
            # Make prediction
            predicted_action, confidence, top_predictions = predict_single_image(
                model, metadata, img_array
            )
            
            results.append({
                "filename": file.filename,
                "predicted_action": predicted_action,
                "confidence": confidence,
                "top_predictions": top_predictions
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Error processing image: {str(e)}"
            })
    
    return {
        "results": results,
        "total_processed": len(results),
        "model_accuracy": f"{metadata['model_accuracy']*100:.2f}%"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Action Recognition API...")
    print(f"📝 API Documentation will be available at: http://localhost:{PORT}/docs")
    print(f"🔍 Interactive API explorer at: http://localhost:{PORT}/redoc")
    print(f"🌐 Server accessible at: http://{HOST}:{PORT}")
    print("⏹️  Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "app:app", 
        host=HOST, 
        port=PORT, 
        reload=False,  # Disable reload to avoid import issues
        log_level="info"
    )