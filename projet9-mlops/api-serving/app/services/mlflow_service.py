import mlflow
import mlflow.spark
import os
import sys

# Forcer JAVA_HOME avant d'importer PySpark
os.environ["JAVA_HOME"] = "/usr/lib/jvm/default-java"
os.environ["PATH"] = f"/usr/lib/jvm/default-java/bin:{os.environ.get('PATH', '')}"

from pyspark.sql import SparkSession
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

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
        self.spark         = None

        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"MLflow Tracking URI : {self.tracking_uri}")

    def _get_spark(self):
        """Créer ou récupérer la SparkSession"""
        if self.spark is None:
            logger.info(f"JAVA_HOME = {os.environ.get('JAVA_HOME')}")
            self.spark = SparkSession.builder \
                .appName("MLOps-API") \
                .master("local[*]") \
                .config("spark.driver.memory", "1g") \
                .config("spark.ui.enabled", "false") \
                .getOrCreate()
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info("SparkSession créée avec succès")
        return self.spark

    def load_model(self):
        """Charger le modèle Spark depuis le registre MLflow"""
        try:
            logger.info("Initialisation de SparkSession...")
            self._get_spark()

            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            logger.info(f"Chargement du modèle : {model_uri}")

            self.model = mlflow.spark.load_model(model_uri)

            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(
                self.model_name,
                stages=[self.model_stage]
            )
            if versions:
                self.model_version = versions[0].version

            logger.info(f"Modèle Spark chargé - Version : {self.model_version}")
            return True

        except Exception as e:
            logger.error(f"Erreur chargement modèle : {e}")
            return False

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Prédiction pour une seule instance"""
        if self.model is None:
            raise RuntimeError("Modèle non chargé")

        spark = self._get_spark()
        df = spark.createDataFrame([features])
        predictions = self.model.transform(df)
        pred_index = int(predictions.select("prediction").first()[0])

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

        spark = self._get_spark()
        df = spark.createDataFrame(instances)
        predictions = self.model.transform(df)
        pred_rows = predictions.select("prediction").collect()

        results = []
        for row in pred_rows:
            pred_index = int(row[0])
            results.append({
                "prediction":    pred_index,
                "species":       SPECIES_MAP.get(pred_index, "unknown"),
                "confidence":    1.0,
                "model_version": self.model_version
            })
        return results

mlflow_service = MLflowService()
