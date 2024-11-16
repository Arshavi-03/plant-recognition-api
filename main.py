from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import boto3
import joblib
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_REGION')
)

# S3 Configuration
BUCKET_NAME = 'virtual-herbal-garden-3d-models'
MODEL_PATH = 'deployment_model/plant_recognition_model_best.h5'
LABEL_ENCODER_PATH = 'deployment_model/plant_label_encoder.joblib'

# Global variables
model = None
label_encoder = None

def load_model_from_s3():
    """Load the model and label encoder from S3"""
    global model, label_encoder
    
    try:
        # Download model file
        logger.info("Downloading model file...")
        model_obj = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_PATH)
        model_bytes = BytesIO(model_obj['Body'].read())
        
        # Download label encoder
        logger.info("Downloading label encoder...")
        le_obj = s3.get_object(Bucket=BUCKET_NAME, Key=LABEL_ENCODER_PATH)
        le_bytes = BytesIO(le_obj['Body'].read())
        
        # Save temporarily and load model
        temp_model_path = '/tmp/model.h5'
        with open(temp_model_path, 'wb') as f:
            f.write(model_bytes.getvalue())
        
        logger.info("Loading model...")
        model = load_model(temp_model_path)
        
        # Load label encoder
        logger.info("Loading label encoder...")
        label_encoder = joblib.load(le_bytes)
        
        # Clean up temporary file
        os.remove(temp_model_path)
        
        logger.info("Model and label encoder loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the image for model input"""
    # Resize image
    image = image.resize((224, 224))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    load_model_from_s3()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None
    }

@app.post("/api/recognize-plant")
async def recognize_plant(file: UploadFile = File(...)) -> Dict:
    """Recognize plant from uploaded image"""
    if not model or not label_encoder:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get plant name
        plant_name = label_encoder.inverse_transform([predicted_class])[0]
        
        return {
            "plant": plant_name,
            "confidence": float(confidence),
            "probabilities": {
                plant: float(prob) 
                for plant, prob in zip(label_encoder.classes_, predictions[0])
            }
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)