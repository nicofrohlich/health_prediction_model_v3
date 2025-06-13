from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class SignalInput(BaseModel):
    patient_id: str
    timestamp: datetime
    heart_rate: float = Field(..., ge=30, le=200)
    spo2: float = Field(..., ge=70, le=100)
    systolic_bp: float = Field(..., ge=70, le=250)
    bmi: float = Field(..., ge=10, le=60)
    age: int = Field(..., ge=0, le=120)

class PredictionResponse(BaseModel):
    prediction: int
    risk_probability: float
