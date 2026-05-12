# api-serving/app/routes/health.py

from fastapi import APIRouter
from datetime import datetime
from app.services.mlflow_service import mlflow_service

router = APIRouter()


@router.get("/health", tags=["Health"])
def health():
    return {
        "status"    : "healthy",
        "timestamp" : datetime.utcnow().isoformat(),
        "service"   : "MLOps Diabetes Prediction API"
    }


@router.get("/ready", tags=["Health"])
def ready():
    loaded = mlflow_service.is_loaded()
    return {
        "status"        : "ready" if loaded else "not ready",
        "model_loaded"  : loaded,
        "model_name"    : mlflow_service.model_name,
        "model_version" : mlflow_service.get_model_version()
    }
