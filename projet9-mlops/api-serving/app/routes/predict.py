# api-serving/app/routes/predict.py

from fastapi import APIRouter, HTTPException
from app.models.prediction import (
    DiabetesPredictionRequest,
    DiabetesBatchRequest,
    DiabetesPredictionResponse,
    DiabetesBatchResponse
)
# ✅ Importer l'instance globale — PAS la classe
from app.services.mlflow_service import mlflow_service

router = APIRouter()
# ❌ Supprimer cette ligne — elle créait une 2ème instance en conflit
# mlflow_service = MLflowService()


def build_features(patient: DiabetesPredictionRequest) -> list:
    """Construire le vecteur de features dans le bon ordre"""
    return [[
        patient.pregnancies,
        patient.glucose,
        patient.blood_pressure,
        patient.skin_thickness,
        patient.insulin,
        patient.bmi,
        patient.diabetes_pedigree,
        patient.age
    ]]


def get_risk_level(probability: float) -> str:
    """Déterminer le niveau de risque selon la probabilité"""
    if probability < 0.3:
        return "Faible"
    elif probability < 0.6:
        return "Modéré"
    elif probability < 0.8:
        return "Élevé"
    else:
        return "Très élevé"


def format_response(prediction, proba, model_version) -> dict:
    """Formater la réponse de prédiction"""
    prob_diabetic = float(proba[1])
    prob_healthy  = float(proba[0])
    return {
        "prediction"           : int(prediction),
        "class_name"           : "diabétique" if prediction == 1 else "non diabétique",
        "probability_diabetic" : round(prob_diabetic, 4),
        "probability_healthy"  : round(prob_healthy, 4),
        "risk_level"           : get_risk_level(prob_diabetic),
        "model_version"        : model_version
    }


# ─── Prédiction unitaire ──────────────────────────────────────────────────────
@router.post(
    "/predict",
    response_model=DiabetesPredictionResponse,
    summary="Prédiction diabète pour un patient"
)
async def predict(request: DiabetesPredictionRequest):
    try:
        features      = build_features(request)
        prediction    = mlflow_service.predict(features)
        proba         = mlflow_service.predict_proba(features)
        model_version = mlflow_service.get_model_version()
        return format_response(prediction[0], proba[0], model_version)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Prédiction batch ─────────────────────────────────────────────────────────
@router.post(
    "/predict/batch",
    response_model=DiabetesBatchResponse,
    summary="Prédiction diabète pour plusieurs patients"
)
async def predict_batch(request: DiabetesBatchRequest):
    try:
        results       = []
        model_version = mlflow_service.get_model_version()

        for patient in request.patients:
            features   = build_features(patient)
            prediction = mlflow_service.predict(features)
            proba      = mlflow_service.predict_proba(features)
            results.append(
                format_response(prediction[0], proba[0], model_version)
            )

        return {
            "predictions"  : results,
            "count"        : len(results),
            "model_version": model_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
