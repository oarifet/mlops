import os
import mlflow
from mlflow.tracking import MlflowClient

# ─── Configuration ────────────────────────────────────────
os.environ["MLFLOW_TRACKING_URI"]    = "http://localhost:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"]      = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "minioadmin123"

client     = MlflowClient("http://localhost:5000")
MODEL_NAME = "diabetes_model"

# ─── Afficher toutes les versions ─────────────────────────
print("\n📋 Versions disponibles :")
print(f"{'Version':<10} {'Run ID':<15} {'Stage':<15} {'Modèle'}")
print("─" * 60)

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
for v in versions:
    run = client.get_run(v.run_id)
    model_type = run.data.params.get("model_type", "unknown")
    print(f"v{v.version:<9} {v.run_id[:12]:<15} {v.current_stage:<15} {model_type}")

# ─── Promouvoir la version 3 (GradientBoosting) ──────────
BEST_VERSION = "3"

print(f"\n🚀 Promotion de la version {BEST_VERSION} en Production...")

# Archiver les anciennes versions en Production
for v in versions:
    if v.current_stage == "Production":
        client.transition_model_version_stage(
            name    = MODEL_NAME,
            version = v.version,
            stage   = "Archived"
        )
        print(f"  📦 Version {v.version} archivée")

# Promouvoir la meilleure version
client.transition_model_version_stage(
    name    = MODEL_NAME,
    version = BEST_VERSION,
    stage   = "Production"
)

print(f"  ✅ Version {BEST_VERSION} (GradientBoosting) promue en Production !")
print(f"\n📊 Voir sur : http://localhost:5000/#/models/{MODEL_NAME}")
