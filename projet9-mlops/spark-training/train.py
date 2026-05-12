# spark-training/train.py

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI",  "http://localhost:5000")
MLFLOW_S3_ENDPOINT   = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID    = os.getenv("AWS_ACCESS_KEY_ID",    "minioadmin")
AWS_SECRET_ACCESS_KEY= os.getenv("AWS_SECRET_ACCESS_KEY","minioadmin123")
DATA_PATH            = os.getenv("DATA_PATH", "data/raw/diabetes.csv")
EXPERIMENT_NAME      = "diabetes_classification"
MODEL_NAME           = "diabetes_model"

os.environ["MLFLOW_S3_ENDPOINT_URL"]  = MLFLOW_S3_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]       = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"]   = AWS_SECRET_ACCESS_KEY

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ─── Chargement des données ───────────────────────────────────────────────────
print(f"\n📦 Chargement des données depuis : {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"   Shape : {df.shape}")
print(f"   Colonnes : {list(df.columns)}")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"   Train : {X_train.shape} | Test : {X_test.shape}")

# ─── Modèles à entraîner ──────────────────────────────────────────────────────
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=3, random_state=42
    ),
    "SVM": SVC(
        kernel="rbf", probability=True, random_state=42
    )
}

# ─── Expérience MLflow ────────────────────────────────────────────────────────
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

print(f"\n🚀 Lancement des expériences MLflow...")
print("-" * 60)

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):

        # Entraînement
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Métriques
        metrics = {
            "accuracy"  : accuracy_score(y_test, y_pred),
            "f1_score"  : f1_score(y_test, y_pred),
            "precision" : precision_score(y_test, y_pred),
            "recall"    : recall_score(y_test, y_pred),
            "roc_auc"   : roc_auc_score(y_test, y_proba)
        }

        # Paramètres
        params = model.get_params()

        # Log MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Enregistrer le modèle
        mlflow.sklearn.log_model(
            sk_model        = model,
            artifact_path   = "model",
            registered_model_name = MODEL_NAME
        )

        print(f"✅ {model_name:<22}"
              f"→ Accuracy: {metrics['accuracy']:.4f}"
              f"  ROC AUC: {metrics['roc_auc']:.4f}")

print("-" * 60)
print(f"\n✅ Toutes les expériences sont terminées !")
print(f"📊 Voir les résultats sur : {MLFLOW_TRACKING_URI}")
print(f"   Expérience : {EXPERIMENT_NAME}")
print(f"   Modèle     : {MODEL_NAME}")
