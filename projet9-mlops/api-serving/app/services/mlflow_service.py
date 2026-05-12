# api-serving/app/services/mlflow_service.py

import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging

logger = logging.getLogger(__name__)

class MLflowService:
    def __init__(self):
        self.tracking_uri  = os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://mlflow:5000"
        )
        self.model_name    = os.getenv("MODEL_NAME",  "diabetes_model")
        self.model_stage   = os.getenv("MODEL_STAGE", "Production")
        self.model         = None
        self.model_version = None

        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"MLflowService initialisé — URI: {self.tracking_uri}")

    def load_model(self):
        """Chargement du modèle — appelé depuis main.py au démarrage"""
        try:
            model_uri    = f"models:/{self.model_name}/{self.model_stage}"
            logger.info(f"Chargement du modèle depuis : {model_uri}")
            self.model   = mlflow.sklearn.load_model(model_uri)
            self.model_version = self._get_version()
            logger.info(f"✅ Modèle chargé : {self.model_name} v{self.model_version}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle : {e}")
            raise e

    def _get_version(self):
        try:
            client   = MlflowClient()
            versions = client.get_latest_versions(
                self.model_name,
                stages=[self.model_stage]
            )
            return versions[0].version if versions else "unknown"
        except Exception:
            return "unknown"

    def is_loaded(self):
        return self.model is not None

    def predict(self, features):
        if not self.is_loaded():
            raise RuntimeError("Modèle non chargé")
        return self.model.predict(features)

    def predict_proba(self, features):
        if not self.is_loaded():
            raise RuntimeError("Modèle non chargé")
        return self.model.predict_proba(features)

    def get_model_version(self):
        return self.model_version or "unknown"


# ✅ Instance globale — importée par main.py et predict.py
mlflow_service = MLflowService()
