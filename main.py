from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import boto3
import pickle
from io import BytesIO
import tensorflow as tf
import os
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Plant Recognition API",
    description="API for recognizing medicinal plants",
    version="1.0.0"
)

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
MODEL_PATH = 'deployment_model/model.h5'
LABEL_ENCODER_PATH = 'deployment_model/label_encoder.joblib'

# Global variables
model = None
label_encoder = None
CLASSES = ['Amla', 'Ashwagandha', 'Neem', 'Tulsi']

def create_model():
    """Create the model architecture"""
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

class SimpleEncoder:
    """Simple label encoder replacement"""
    def __init__(self, classes):
        self.classes_ = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
    
    def inverse_transform(self, indices):
        return [self.classes_[i] for i in indices]

def load_model_from_s3():
    """Load the model and label encoder from S3"""
    global model, label_encoder
    
    try:
        # Create temp directory
        os.makedirs('/tmp', exist_ok=True)
        
        # Download model file
        logger.info("Downloading model file...")
        model_path = '/tmp/model.h5'
        s3.download_file(BUCKET_NAME, MODEL_PATH, model_path)
        
        # Create model with architecture
        logger.info("Creating model architecture...")
        model = create_model()
        
        # Load weights
        logger.info("Loading model weights...")
        model.load_weights(model_path)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Use simple encoder
        logger.info("Setting up label encoder...")
        label_encoder = SimpleEncoder(CLASSES)
        
        # Verify model works
        logger.info("Testing model...")
        test_input = np.zeros((1, 224, 224, 3))
        _ = model.predict(test_input, verbose=0)
        logger.info("Model loaded and verified successfully")
        
        # Clean up
        os.remove(model_path)
        
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
    try:
        load_model_from_s3()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "service": "Plant Recognition API",
        "endpoints": {
            "health": "/health",
            "recognize": "/api/recognize-plant"
        },
        "supported_plants": CLASSES
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "available_classes": CLASSES,
        "environment": {
            "aws_region": os.environ.get('AWS_REGION'),
            "bucket": BUCKET_NAME
        }
    }

@app.post("/api/recognize-plant")
async def recognize_plant(file: UploadFile = File(...)) -> Dict:
    """
    Recognize plant from uploaded image
    
    Returns:
        Dict containing:
        - plant: recognized plant name
        - confidence: confidence score (0-1)
        - probabilities: probability scores for all classes
    """
    if not model or not label_encoder:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get plant name
        plant_name = label_encoder.inverse_transform([predicted_class])[0]
        
        # Get all class probabilities
        class_probs = {
            name: float(prob)
            for name, prob in zip(CLASSES, predictions[0])
        }
        
        return {
            "plant": plant_name,
            "confidence": confidence,
            "probabilities": class_probs,
            "prediction_time": "fast"
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)