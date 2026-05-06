from fastapi import APIRouter, HTTPException
from app.models.prediction import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from app.services.mlflow_service import mlflow_service
import logging
import time

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """Prédiction pour une seule fleur Iris"""
    try:
        start = time.time()
        features = request.features.model_dump()
        result   = mlflow_service.predict(features)
        duration = round((time.time() - start) * 1000, 2)
        logger.info(f"Prédiction : {result['species']} en {duration}ms")
        return PredictionResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Prédiction par lots pour plusieurs fleurs Iris"""
    try:
        start     = time.time()
        instances = [i.model_dump() for i in request.instances]
        results   = mlflow_service.predict_batch(instances)
        duration  = round((time.time() - start) * 1000, 2)
        logger.info(f"Batch de {len(instances)} prédictions en {duration}ms")
        return BatchPredictionResponse(
            predictions   = [PredictionResponse(**r) for r in results],
            total         = len(results),
            model_version = mlflow_service.model_version
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur batch : {e}")
        raise HTTPException(status_code=500, detail=str(e))