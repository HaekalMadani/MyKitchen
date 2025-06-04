from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import joblib
import os

MODEL_PATH = "./fine_tuned_distilbert_bookmark_classifier"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found at: {MODEL_PATH}. "
                            "Please ensure your fine_tuned_distilbert_bookmark_classifier folder exists.")

# use cpu
device = torch.device("cpu")

#Global Variables for Model & Tokenizer
tokenizer = None
model = None
label_encoder = None
classifier_pipeline = None

# ------ FastAPI stuff ------------

app = FastAPI(
    title= "SocialLoom BERT backend",
    version = "1.0.0"
)

# load model

@app.on_event("startup")
async def load_model():
    global tokenizer, model, label_encoder, classifier_pipeline

    print("Loading model..")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device) 
        model.eval() 

        label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))

        # Hugging Face pipeline
        classifier_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1
        )
        print("Model, tokenizer, and label encoder loaded successfully.")
    except Exception as e:
        print(f"Error loading model components: {e}")

# pydantic model

class BookmarkRequest(BaseModel):
    text: str

# endpoints

# POST Receives text input and returns the predicted category and confidence. IF CONFIDENCE <.5 RETURN NO PREDICTION
