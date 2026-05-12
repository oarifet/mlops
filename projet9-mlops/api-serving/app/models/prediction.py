# api-serving/app/models/prediction.py

from pydantic import BaseModel, Field
from typing import List

class DiabetesPredictionRequest(BaseModel):
    pregnancies: int = Field(
        ..., ge=0, le=20,
        description="Nombre de grossesses",
        example=2
    )
    glucose: float = Field(
        ..., ge=0, le=300,
        description="Concentration en glucose (mg/dL)",
        example=120.0
    )
    blood_pressure: float = Field(
        ..., ge=0, le=200,
        description="Pression artérielle (mm Hg)",
        example=70.0
    )
    skin_thickness: float = Field(
        ..., ge=0, le=100,
        description="Épaisseur du pli cutané (mm)",
        example=20.0
    )
    insulin: float = Field(
        ..., ge=0, le=900,
        description="Insuline sérique (µU/ml)",
        example=80.0
    )
    bmi: float = Field(
        ..., ge=0, le=70,
        description="Indice de masse corporelle (kg/m²)",
        example=25.5
    )
    diabetes_pedigree: float = Field(
        ..., ge=0, le=3,
        description="Fonction pedigree diabète",
        example=0.5
    )
    age: int = Field(
        ..., ge=1, le=120,
        description="Âge du patient",
        example=33
    )

class DiabetesBatchRequest(BaseModel):
    patients: List[DiabetesPredictionRequest]

class DiabetesPredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probability_diabetic: float
    probability_healthy: float
    risk_level: str
    model_version: str

class DiabetesBatchResponse(BaseModel):
    predictions: List[DiabetesPredictionResponse]
    count: int
    model_version: str
