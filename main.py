from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import boto3
import joblib
from io import BytesIO
import tensorflow as tf
import os
import logging
from typing import Dict
import json
import tensorflowjs as tfjs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
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

# Update these paths to match your S3 structure
BUCKET_NAME = 'virtual-herbal-garden-3d-models'
MODEL_DIR = 'trained_models'
MODEL_JSON = f'{MODEL_DIR}/model.json'
LABEL_ENCODER_PATH = f'{MODEL_DIR}/plant_label_encoder.joblib'

# Global variables for model and label encoder
model = None
label_encoder = None

def download_and_save_files():
    """Download all necessary files from S3"""
    os.makedirs('/tmp/model', exist_ok=True)
    
    # Download model.json
    logger.info("Downloading model.json...")
    s3.download_file(BUCKET_NAME, MODEL_JSON, '/tmp/model/model.json')
    
    # Download shard files
    shard_files = [
        'group1-shard1of3.bin',
        'group1-shard2of3.bin',
        'group1-shard3of3.bin'
    ]
    
    for shard in shard_files:
        logger.info(f"Downloading {shard}...")
        s3.download_file(
            BUCKET_NAME, 
            f'{MODEL_DIR}/{shard}', 
            f'/tmp/model/{shard}'
        )
    
    # Download label encoder
    logger.info("Downloading label encoder...")
    s3.download_file(
        BUCKET_NAME,
        LABEL_ENCODER_PATH,
        '/tmp/label_encoder.joblib'
    )

def load_model_from_s3():
    global model, label_encoder
    
    try:
        # Download all files
        download_and_save_files()
        
        # Load the model using tensorflowjs
        logger.info("Loading model with tensorflowjs...")
        model = tfjs.converters.load_keras_model('/tmp/model/model.json')
        
        # Load label encoder
        logger.info("Loading label encoder...")
        label_encoder = joblib.load('/tmp/label_encoder.joblib')
        
        logger.info("Model and label encoder loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"AWS Region: {os.environ.get('AWS_REGION')}")
        logger.error(f"Bucket: {BUCKET_NAME}")
        raise

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize image to match training input size
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
    try:
        logger.info("Starting application...")
        logger.info("Checking AWS credentials...")
        logger.info(f"AWS Region: {os.environ.get('AWS_REGION')}")
        logger.info(f"Bucket: {BUCKET_NAME}")
        load_model_from_s3()
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/test-s3")
async def test_s3():
    """Endpoint to test S3 connectivity and list files"""
    try:
        # List objects in the trained_models directory
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix='trained_models/'
        )
        files = [obj['Key'] for obj in response.get('Contents', [])]
        return {
            "status": "success",
            "message": "S3 connection successful",
            "files": files
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "aws_region": os.environ.get('AWS_REGION'),
            "bucket": BUCKET_NAME
        }

@app.post("/api/recognize-plant")
async def recognize_plant(file: UploadFile = File(...)) -> Dict:
    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get plant name from label encoder
        plant_name = label_encoder.inverse_transform([predicted_class])[0]
        
        return {
            "plant": plant_name,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)