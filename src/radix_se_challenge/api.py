"""API endpoints for predicting movie genres."""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd 
import pickle
import os
from typing import List

app = FastAPI(
    title="SE challenge",
    description="Hiring challenge for the Software Engineer applicants.",
    version="0.0.0",  
)

class PredictionRequest(BaseModel):
    """Request model for predicting genres from a synopsis."""
    synopsis: str
        
class Prediction(BaseModel):
    """Response model for predicting genres."""
    genres: List[str]

@app.get("/")
def index():
    """Index page."""
    return {"message": "Welcome to the SE challenge!"}

@app.post("/genres/predict", response_model=Prediction)
def predict_genres(request: PredictionRequest):
    """Predict movie genres from synopsis.
    
    The goal is to load saved model and makes prediction on the synopsis text.
    Returns predicted genres.
    """

    # Validate input
    if not request.synopsis:
        raise HTTPException(status_code=400, detail="Synopsis required")

    # Load the saved model
    model_path = Path("src/radix_se_challenge/model/binaries")
    if not (model_path / "classifier.pickle").exists():
        raise HTTPException(status_code=500, detail="Model file not found")

    with open(model_path / "classifier.pickle", "rb") as f:
        clf = pickle.load(f)

    with open(model_path / "binarizer.pickle", "rb") as f:
        binarizer = pickle.load(f)

    with open(model_path / "tfidf.pickle", "rb") as f:
        tfidf = pickle.load(f)

    # Make prediction
    df = pd.DataFrame([{"synopsis": request.synopsis}]) 
    probalities = clf.predict_proba(tfidf.transform(df.synopsis))

    # Transform probabilities to genres
    preds = []
    for args in (-probalities).argsort():
        preds.append([binarizer.classes_[idx] for idx in args[:5]])

    return Prediction(genres=preds[0])
