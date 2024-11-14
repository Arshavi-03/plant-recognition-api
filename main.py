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

BUCKET_NAME = 'virtual-herbal-garden-3d-models'
MODEL_PATH = 'plant_recognition_model_best.h5'
LABEL_ENCODER_PATH = 'plant_label_encoder.joblib'

# Global variables for model and label encoder
model = None
label_encoder = None

def load_model_from_s3():
    global model, label_encoder
    try:
        # Download and load the model
        model_obj = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_PATH)
        model_bytes = model_obj['Body'].read()
        
        with open('/tmp/model.h5', 'wb') as f:
            f.write(model_bytes)
        model = tf.keras.models.load_model('/tmp/model.h5')
        
        # Download and load the label encoder
        le_obj = s3.get_object(Bucket=BUCKET_NAME, Key=LABEL_ENCODER_PATH)
        label_encoder = joblib.load(BytesIO(le_obj['Body'].read()))
        
        logger.info("Model and label encoder loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
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
    load_model_from_s3()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

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