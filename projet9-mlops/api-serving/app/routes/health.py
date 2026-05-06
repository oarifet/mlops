from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health", tags=["Monitoring"])
def health_check():
    """Vérifier l'état de l'API"""
    return {
        "status":    "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service":   "MLOps Prediction API"
    }

@router.get("/ready", tags=["Monitoring"])
def readiness_check():
    """Vérifier si le modèle est chargé et prêt"""
    from app.services.mlflow_service import mlflow_service
    model_loaded = mlflow_service.model is not None
    return {
        "status":        "ready" if model_loaded else "not_ready",
        "model_loaded":  model_loaded,
        "model_version": mlflow_service.model_version,
        "timestamp":     datetime.utcnow().isoformat()
    }