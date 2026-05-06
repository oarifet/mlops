from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import predict, health
from app.services.mlflow_service import mlflow_service
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage : charger le modèle
    logger.info("Démarrage de l'API - Chargement du modèle MLflow...")
    mlflow_service.load_model()
    yield
    # Arrêt
    logger.info("Arrêt de l'API")

app = FastAPI(
    title       = "MLOps Prediction API",
    description = "API REST pour servir les prédictions du modèle Iris via MLflow",
    version     = "1.0.0",
    lifespan    = lifespan
)

# Inclure les routes
app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "MLOps Prediction API",
        "docs":    "/docs",
        "health":  "/health"
    }