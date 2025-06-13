import pandas as pd
import joblib
from utils.utils import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from preprocess_dataset import preprocess

# 1) Pré-processar dados da pasta de entrada
dataset_folder = "dataset"
output_csv_path = "janelas_threshold.csv"
preprocess(dataset_folder, output_csv_path)

# 2) Carregar dados processados
df = pd.read_csv(output_csv_path)

df_features = df.apply(extract_features, axis=1)
df = pd.concat([df, df_features], axis=1)

# 4) Preparar X e y com FE
feature_cols = [col for col in df.columns if col.startswith("ews_w_day_")] + list(df_features.columns)
X = df[feature_cols]
y = df["risk_next_3_days"]

# 4) Preparar X e y sem FE
# feature_cols = [col for col in df.columns if col.startswith("ews_w_day_")]
# X = df[feature_cols]
# y = df["risk_next_3_days"]

# 5) Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.30, random_state=42
)

# # 6) Treinar modelo Random Forest
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 6) Treinar modelo Logistic Regression
# model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
# model.fit(X_train, y_train)

# 7) Salvar o modelo treinado
joblib.dump(model, "app/model/model.joblib")
