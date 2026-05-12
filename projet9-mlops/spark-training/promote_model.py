# spark-training/promote_model.py

import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI",   "http://localhost:5000")
MLFLOW_S3_ENDPOINT    = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID",     "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
MODEL_NAME            = "diabetes_model"

os.environ["MLFLOW_S3_ENDPOINT_URL"]  = MLFLOW_S3_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]       = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"]   = AWS_SECRET_ACCESS_KEY

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# ─── Lister toutes les versions ───────────────────────────────────────────────
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

print(f"\n📋 Versions disponibles :")
print(f"{'Version':<10} {'Run ID':<16} {'Stage':<16} {'ROC AUC':<10} {'Modèle'}")
print("─" * 75)

best_version    = None
best_roc_auc    = -1
best_model_name = "unknown"

for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
    run_id = v.run_id
    stage  = v.current_stage

    # Récupérer les métriques du run
    try:
        run      = client.get_run(run_id)
        roc_auc  = run.data.metrics.get("roc_auc", 0)
        run_name = run.data.tags.get("mlflow.runName", "unknown")
    except Exception:
        roc_auc  = 0
        run_name = "unknown"

    print(f"v{v.version:<9} {run_id[:16]:<16} {stage:<16} {roc_auc:<10.4f} {run_name}")

    # Trouver le meilleur modèle
    if roc_auc > best_roc_auc:
        best_roc_auc    = roc_auc
        best_version    = v.version
        best_model_name = run_name

print("─" * 75)
print(f"\n🏆 Meilleur modèle : v{best_version} ({best_model_name})"
      f" — ROC AUC : {best_roc_auc:.4f}")

# ─── Archiver les versions en Production ──────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

for v in versions:
    if v.current_stage == "Production":
        client.transition_model_version_stage(
            name    = MODEL_NAME,
            version = v.version,
            stage   = "Archived"
        )
        print(f"  📦 Version {v.version} archivée")

# ─── Promouvoir le meilleur modèle ────────────────────────────────────────────
print(f"\n🚀 Promotion de la version {best_version} en Production...")

client.transition_model_version_stage(
    name                         = MODEL_NAME,
    version                      = best_version,
    stage                        = "Production",
    archive_existing_versions    = True
)

print(f"  ✅ Version {best_version} ({best_model_name}) promue en Production !")
print(f"\n📊 Voir sur : {MLFLOW_TRACKING_URI}/#/models/{MODEL_NAME}")
