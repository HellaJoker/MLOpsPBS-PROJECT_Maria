from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="FitCheck Microservice", version="1.0.0")

# Load trained model and features schema
try:
    model = joblib.load("fitcheck_gb_model.joblib")
    model_features = joblib.load("model_features.joblib")
except Exception:
    model = None
    model_features = []

class PredictionInput(BaseModel):
    height_cm: float
    weight_kg: float
    garment_chest_cm: float
    fabric_stretch_pct: float
    product_type_id: int

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "expected_features_count": len(model_features)
    }

@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")

    # 1. Calculate dynamic BMI
    imc_index = data.weight_kg / ((data.height_cm / 100.0) ** 2)

    # 2. Map one-hot encoding structure
    input_dict = {
        "height_cm": data.height_cm,
        "weight_kg": data.weight_kg,
        "imc_index": imc_index,
        "garment_chest_cm": data.garment_chest_cm,
        "fabric_stretch_pct": data.fabric_stretch_pct,
        "product_type_id_0": 1 if data.product_type_id == 0 else 0,
        "product_type_id_1": 1 if data.product_type_id == 1 else 0,
        "product_type_id_2": 1 if data.product_type_id == 2 else 0,
    }

    # 3. Align DataFrame schema with training features
    df_input = pd.DataFrame([input_dict]).reindex(columns=model_features, fill_value=0)

    # 4. Predict class & probability score
    prediction_id = int(model.predict(df_input)[0])
    probabilities = model.predict_proba(df_input)[0]
    confidence = float(probabilities[prediction_id])

    # 5. Sizing class label mapping
    class_map = {0: "Too Small", 1: "Good Fit", 2: "Too Large"}

    return {
        "prediction_class": class_map.get(prediction_id, "Unknown"),
        "confidence": round(confidence, 4),
        "raw_class_id": prediction_id,
        "imc_index": round(imc_index, 2)
    }
