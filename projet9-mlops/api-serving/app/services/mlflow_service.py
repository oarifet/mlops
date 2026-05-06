import mlflow
import mlflow.spark
import os
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Mapping index -> nom de l'espèce
SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

class MLflowService:
    def __init__(self):
        self.tracking_uri  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.model_name    = os.getenv("MODEL_NAME", "iris-classifier")
        self.model_stage   = os.getenv("MODEL_STAGE", "Production")
        self.model         = None
        self.model_version = None

        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"MLflow Tracking URI : {self.tracking_uri}")

    def load_model(self):
        """Charger le modèle depuis le registre MLflow"""
        try:
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            logger.info(f"Chargement du modèle : {model_uri}")

            self.model = mlflow.pyfunc.load_model(model_uri)

            # Récupérer la version du modèle
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(
                self.model_name,
                stages=[self.model_stage]
            )
            if versions:
                self.model_version = versions[0].version

            logger.info(f"Modèle chargé - Version : {self.model_version}")
            return True

        except Exception as e:
            logger.error(f"Erreur chargement modèle : {e}")
            return False

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Prédiction pour une seule instance"""
        if self.model is None:
            raise RuntimeError("Modèle non chargé")

        df = pd.DataFrame([features])
        prediction = self.model.predict(df)
        pred_index = int(prediction[0])

        return {
            "prediction":    pred_index,
            "species":       SPECIES_MAP.get(pred_index, "unknown"),
            "confidence":    1.0,
            "model_version": self.model_version
        }

    def predict_batch(self, instances: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Prédiction par lots"""
        if self.model is None:
            raise RuntimeError("Modèle non chargé")

        df          = pd.DataFrame(instances)
        predictions = self.model.predict(df)

        results = []
        for pred in predictions:
            pred_index = int(pred)
            results.append({
                "prediction":    pred_index,
                "species":       SPECIES_MAP.get(pred_index, "unknown"),
                "confidence":    1.0,
                "model_version": self.model_version
            })
        return results

# Instance globale
mlflow_service = MLflowService()