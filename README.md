# Health Prediction Model

Este projeto tem como objetivo principal realizar a predição antecipada de deterioração clínica em pacientes utilizando dados fisiológicos coletados por dispositivos vestíveis (wearables). O modelo implementado utiliza algoritmos de aprendizado de máquina, especificamente Random Forest, além de técnicas de Feature Engineering para aumentar a precisão das previsões.

## Arquitetura

```
health_prediction_model_v3/
├── app/
│   ├── main.py
│   ├── db.py
│   ├── schemas.py
│   └── model/
│       ├── predictor.py
│       └── model.joblib
├── dataset/
│   ├── patients.csv
│   └── observations.csv
├── metrics/
├── utils/
├── preprocess_dataset.py
├── train_model.py
├── analyze_model.py
├── processed_dataset.csv
├── janelas_threshold.csv
├── requirements.txt
└── data.db  ← gerado automaticamente ao rodar a API
```

## Requisitos

- Python 3.10+
- SQLite
- FastAPI, Scikit-learn, Pandas, NumPy, Joblib

Instale as dependências utilizando:

```bash
pip install -r requirements.txt
```

## Passo a passo (Se oo modelo já está treinado siga direto para o passo 4)

### 1. Adicione os arquivos de entrada

Insira os arquivos `patients.csv` e `observations.csv` na pasta `dataset/`.

### 2. Pré-processar os dados

```bash
python preprocess_dataset.py
```

### 3. Treinar o modelo

```bash
python train_model.py
```

### 4. Executar a API (Inicie daqui caso modelo ja esteja treinado)

```bash
uvicorn app.main:app --reload
```

## Utilizando a API

### Interface Swagger:

```url
http://127.0.0.1:8000/docs
```

### Exemplo de requisição para predição:

```json
POST /predict
{
  "patient_id": "123",
  "timestamp": "2025-05-07T10:00:00",
  "heart_rate": 85,
  "spo2": 98,
  "systolic_bp": 122,
  "bmi": 24,
  "age": 60
}
```
