from dotenv import load_dotenv 
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import joblib
import os
import tempfile
from google.cloud import storage


load_dotenv()

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "your-gcs-bucket-name")
GCS_MODEL_BLOB_PREFIX = os.getenv("GCS_MODEL_BLOB_PREFIX", "fine_tuned_distilbert_bookmark_classifier/")

# use cpu
device = torch.device("cpu")

#Global Variables for Model & Tokenizer
tokenizer = None
model = None
label_encoder = None
classifier_pipeline = None

# load model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, label_encoder, classifier_pipeline

    print("Loading model..")

    try:
        temp_model_dir = tempfile.mkdtemp()
        print(f"Downloading model files to temporary directory: {temp_model_dir}")

        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)

        model_files = [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "label_encoder.pkl"
        ]

        for filename in model_files:
            gcs_blob_path = f"{GCS_MODEL_BLOB_PREFIX}{filename}"
            local_file_path = os.path.join(temp_model_dir, filename)
            blob = bucket.blob(gcs_blob_path)
            blob.download_to_filename(local_file_path)
            print(f"Downloaded {gcs_blob_path} to {local_file_path}")

        tokenizer = AutoTokenizer.from_pretrained(temp_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(temp_model_dir)
        model.to(device) 
        model.eval() 

        label_encoder = joblib.load(os.path.join(temp_model_dir, "label_encoder.pkl"))

        # Hugging Face pipeline
        classifier_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1
        )
        print("Model, tokenizer, and label encoder loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model components: {e}")
        raise
    finally:
        yield
        print("Shutting down - cleaning up temporary model directory.")
        import shutil

        try:
            shutil.rmtree(temp_model_dir)
            print(f"Cleaned up temporary directory: {temp_model_dir}")
        except OSError as e:
            print(f"Error removing temporary directory {temp_model_dir}: {e}")

# ------ FastAPI stuff ------------

app = FastAPI(
    title= "SocialLoom BERT backend",
    version = "1.0.0",
    lifespan=lifespan
)

# pydantic model

class BookmarkRequest(BaseModel):
    text: str

# endpoints

# POST Receives text input and returns the predicted category and confidence. IF CONFIDENCE <.5 RETURN NO PREDICTION

@app.post("/predict_category")
async def predict_category(request: BookmarkRequest):
    if classifier_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet")
    
    classify_text = request.text
    if not classify_text:
        raise HTTPException(status_code=500, detail="Text to clasify is not given")

    try:
        prediction_result = classifier_pipeline(classify_text)[0]

        predicted_label_id = prediction_result['label']
        score = prediction_result['score']

        numerical_id = int(predicted_label_id.replace("LABEL_", ""))
        predicted_original_label = label_encoder.inverse_transform([numerical_id])[0]

        return {
            "input_text": classify_text,
            "predicted_category": predicted_original_label,
            "confidence": score
        }
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server error prediction: {e}")


@app.get("/")
async def root():
    return {"message": "Bookmark Categorization API is running!"}