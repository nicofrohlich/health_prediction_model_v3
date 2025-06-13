from utils.utils import extract_features
import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    ConfusionMatrixDisplay
)

# Criar diretório de saída se não existir
output_dir = "metrics"
os.makedirs(output_dir, exist_ok=True)

print("Iniciando carregamento de dados e modelo...")
# Carregar dados e modelo
df = pd.read_csv("janelas_threshold.csv")
model = joblib.load("app/model/model.joblib")
print("Dados e modelo carregados.")

print("Extraindo features derivadas...")
df_features = df.apply(extract_features, axis=1)
df = pd.concat([df, df_features], axis=1)
print("Features derivadas extraídas.")

# Preparar X e y com FE
feature_cols = [col for col in df.columns if col.startswith("ews_w_day_")] + list(df_features.columns)
X = df[feature_cols]
y = df["risk_next_3_days"]
print(f"Shape de X: {X.shape}")

# Preparar X e y sem FE
# feature_cols = [col for col in df.columns if col.startswith("ews_w_day_")]
# X = df[feature_cols]
# y = df["risk_next_3_days"]
# print(f"Shape de X: {X.shape}")

print("Realizando previsões...")
# Previsões
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]
print("Previsões concluídas.")

# Avaliação e métricas
print("\n=== AVALIAÇÃO ===")
print(f"Acurácia: {accuracy_score(y, y_pred):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred, digits=4))

print("Gerando matriz de confusão...")
# Matriz de confusão
ConfusionMatrixDisplay.from_predictions(y, y_pred)
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()
print("Matriz de confusão gerada.")

print("Gerando curva ROC...")
# Curva ROC
fpr, tpr, _ = roc_curve(y, y_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/roc_curve.png")
plt.close()
print("Curva ROC gerada.")

print("Gerando curva Precision-Recall...")
# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y, y_prob)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.tight_layout()
plt.savefig(f"{output_dir}/precision_recall_curve.png")
plt.close()
print("Curva Precision-Recall gerada.")

print("Calculando importância das features...")
# Importância das features
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
elif hasattr(model, "coef_"):
    importances = model.coef_[0]
else:
    raise ValueError("O modelo não possui atributo de importância disponível.")

feature_importance = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
feature_importance.to_csv(f"{output_dir}/feature_importance.csv")

feature_importance.plot(kind="bar")
plt.title("Importância das Features")
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_importance.png")
plt.close()
print("Importância das features calculada.")

# print("Iniciando cálculo SHAP (pode demorar dependendo do tamanho de X)...")
# # SHAP para Random Forest (classificação binária)
# explainer = shap.TreeExplainer(model)
# # CUIDADO AQUI: Considere usar explainer.shap_values(X) se shap_interaction_values(X) for muito lento
# shap_interaction = explainer.shap_interaction_values(X)
#
# # As linhas abaixo não são estritamente necessárias para o summary_plot e podem ser removidas se quiser
# mean_pred = np.mean(y_pred)
# sum_shap = np.sum(shap_interaction[0])
#
# # Plot e salvar
# shap.summary_plot(shap_interaction, X, show=False)
# plt.savefig(f"{output_dir}/shap_summary.png", bbox_inches="tight")
# plt.close()
# print("Cálculo SHAP e plot concluídos.")

print("✅ Análises geradas com sucesso")