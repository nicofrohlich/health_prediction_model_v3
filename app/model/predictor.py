import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
model = joblib.load(MODEL_PATH)

def calculate_news_w_row(hr, spo2, bp, bmi, age):
    def s_hr(x): return 3 if x <= 40 or x > 130 else 1 if x <= 50 or x > 110 else 0
    def s_spo2(x): return 3 if x <= 91 else 2 if x <= 93 else 1 if x <= 95 else 0
    def s_bp(x): return 3 if x <= 90 or x >= 220 else 2 if x <= 100 else 1 if x <= 110 else 0
    def s_bmi(x): return 1 if x < 18.5 or x >= 35 else 0
    def s_age(x): return 1 if x >= 65 else 0
    scores = [s_hr(hr), s_spo2(spo2), s_bp(bp), s_bmi(bmi), s_age(age)]
    if max(scores[:3]) == 3:
        return max(sum(scores), 5)
    return sum(scores)

def predict_from_scores(features):
    import pandas as pd
    if len(features) != 13:
        raise ValueError("Esperadas 13 features para predição.")
    columns = [f"ews_w_day_{i+1}" for i in range(7)] + [
        "media", "mediana", "desvio", "trend",
        "escore_max", "escore_min"
    ]
    X = pd.DataFrame([features], columns=columns)
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    return {"prediction": int(pred), "risk_probability": float(prob)}