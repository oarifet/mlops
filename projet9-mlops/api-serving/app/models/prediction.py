from pydantic import BaseModel, Field
from typing import List, Optional

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Longueur du sépale (cm)")
    sepal_width:  float = Field(..., example=3.5, description="Largeur du sépale (cm)")
    petal_length: float = Field(..., example=1.4, description="Longueur du pétale (cm)")
    petal_width:  float = Field(..., example=0.2, description="Largeur du pétale (cm)")

class PredictionRequest(BaseModel):
    features: IrisFeatures

class BatchPredictionRequest(BaseModel):
    instances: List[IrisFeatures]

class PredictionResponse(BaseModel):
    prediction:   int
    species:      str
    confidence:   float
    model_version: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    predictions:  List[PredictionResponse]
    total:        int
    model_version: Optional[str] = None