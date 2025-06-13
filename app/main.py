from fastapi import FastAPI
from app.schemas import SignalInput, PredictionResponse
from app.db import init_db, insert_signal, get_last_7_features
from app.model.predictor import predict_from_scores

app = FastAPI()

@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: SignalInput):
    insert_signal(input_data)
    scores = get_last_7_features(input_data.patient_id)
    if not scores:
        return {"prediction": 0, "risk_probability": 0.0}
    return predict_from_scores(scores)
