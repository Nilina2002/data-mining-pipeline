from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "model"
MODEL_PATH = MODEL_DIR / "rf_model.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
STATS_PATH = MODEL_DIR / "dataset_stats.json"


class PredictionRequest(BaseModel):
    recency: float = Field(ge=0, description="Days since last purchase")
    frequency: float = Field(ge=0, description="Number of orders")
    monetary: float = Field(ge=0, description="Total spend")


app = FastAPI(title="RFM Segment API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_assets():
    if not MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists() or not STATS_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run notebooks/save_model.py first."
        )

    with MODEL_PATH.open("rb") as file_handle:
        model = pickle.load(file_handle)

    with LABEL_ENCODER_PATH.open("rb") as file_handle:
        label_encoder = pickle.load(file_handle)

    with STATS_PATH.open("r", encoding="utf-8") as file_handle:
        stats = json.load(file_handle)

    return model, label_encoder, stats


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/stats")
def model_stats():
    _, _, stats = load_assets()
    return stats


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        model, label_encoder, _ = load_assets()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    input_frame = pd.DataFrame(
        [[request.recency, request.frequency, request.monetary]],
        columns=["Recency", "Frequency", "Monetary"],
    )

    probabilities = model.predict_proba(input_frame)[0]
    decoded_classes = label_encoder.inverse_transform(model.classes_)

    probability_map = {
        str(segment): float(probability)
        for segment, probability in zip(decoded_classes, probabilities)
    }

    predicted_index = int(probabilities.argmax())
    predicted_segment = str(decoded_classes[predicted_index])

    return {
        "predicted": predicted_segment,
        "confidence": float(probabilities[predicted_index]),
        "probabilities": probability_map,
        "input": request.model_dump(),
    }
