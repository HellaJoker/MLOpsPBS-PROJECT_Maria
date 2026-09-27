import os
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fitcheck.features import BOUNDS, CLASS_LABELS, RAW_FEATURES, bmi, build_baseline_features

app = FastAPI(title="FitCheck Microservice", version="1.1.0")

# Models live in MODEL_DIR (/app/Models in the image). Previously the service looked in
# the current directory, found nothing, and silently started with model=None.
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "Models"))
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _load(filename):
    """Load a joblib artifact, returning None (and logging why) if it is missing or broken."""
    path = MODEL_DIR / filename
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"[fitcheck] could not load {path}: {exc}")
        return None


# Baseline: GradientBoostingClassifier trained on one-hot features (needs model_features.joblib).
baseline_model = _load("fitcheck_gb_model.joblib")
baseline_features = _load("model_features.joblib") or []
# Optimized: full sklearn Pipeline that takes the raw inputs directly (created by the notebook).
optimized_model = _load("fitcheck_gb_model_optimized.joblib")

MODELS = {
    name: model
    for name, model in {"baseline": baseline_model, "optimized": optimized_model}.items()
    if model is not None
}


class PredictionInput(BaseModel):
    # Bounds match the ranges the training data was clipped to.
    height_cm: float = Field(ge=BOUNDS["height_cm"][0], le=BOUNDS["height_cm"][1])
    weight_kg: float = Field(ge=BOUNDS["weight_kg"][0], le=BOUNDS["weight_kg"][1])
    garment_chest_cm: float = Field(ge=BOUNDS["garment_chest_cm"][0], le=BOUNDS["garment_chest_cm"][1])
    fabric_stretch_pct: float = Field(ge=BOUNDS["fabric_stretch_pct"][0], le=BOUNDS["fabric_stretch_pct"][1])
    product_type_id: Literal[0, 1, 2]  # 0: Tops, 1: Bottoms, 2: Jackets
    # Which model to use; defaults to the baseline to keep the original API behaviour.
    model: Optional[Literal["baseline", "optimized"]] = "baseline"


@app.get("/", include_in_schema=False)
def web_ui():
    """Serve the fit-check web interface."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": baseline_model is not None,
        "expected_features_count": len(baseline_features),
        "available_models": list(MODELS),
    }


@app.get("/models")
def list_models():
    """Models the UI can choose from, plus the valid input ranges."""
    return {"available_models": list(MODELS), "bounds": BOUNDS, "classes": CLASS_LABELS}


@app.post("/predict")
def predict(data: PredictionInput):
    model = MODELS.get(data.model)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model '{data.model}' is not loaded")

    # 1. Raw input row (same columns for both models)
    raw = pd.DataFrame([data.model_dump(include=set(RAW_FEATURES))])[RAW_FEATURES]

    # 2. Baseline needs the hand-built one-hot schema; the optimized pipeline transforms internally
    model_input = build_baseline_features(raw, baseline_features) if data.model == "baseline" else raw

    # 3. Predict class & probability of every class (ordered by model.classes_)
    probabilities = model.predict_proba(model_input)[0]
    class_ids = [int(c) for c in model.classes_]
    best = int(probabilities.argmax())
    prediction_id = class_ids[best]

    return {
        "prediction_class": CLASS_LABELS.get(prediction_id, "Unknown"),
        "confidence": round(float(probabilities[best]), 4),
        "raw_class_id": prediction_id,
        "imc_index": round(float(bmi(data.height_cm, data.weight_kg)), 2),
        "model": data.model,
        "probabilities": {
            CLASS_LABELS.get(cid, str(cid)): round(float(p), 4) for cid, p in zip(class_ids, probabilities)
        },
    }


if __name__ == "__main__":
    # Allows `python api/app.py` as well as the uvicorn CMD used in the Dockerfile.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "3000")))
