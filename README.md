# MLOps Diabetes Prediction

Pipeline MLOps complet de prédiction du diabète. Le projet couvre tout le cycle : chargement des données, entraînement de modèles en Python (Scikit-learn) et en Scala (Apache Spark MLlib), tracking avec MLflow, sélection automatique du meilleur modèle, et mise en production via une API REST FastAPI.

---

## Vue d'ensemble

Le dataset utilisé est le Pima Indians Diabetes (768 patients, 8 features cliniques). À partir de là, le pipeline :

- charge et prépare les données avec Pandas/Scikit-learn et Spark MLlib
- entraîne 4 modèles Python et 3 modèles Scala/Spark
- logue chaque expérience dans MLflow avec ses métriques et artefacts
- stocke les modèles dans MinIO (compatible S3)
- sélectionne automatiquement le meilleur modèle selon le ROC AUC
- le promeut en Production dans le Model Registry
- expose les prédictions via FastAPI
- containerise toute l'infrastructure avec Docker Compose
![Pipeline MLOps](projet9-mlops/docs/pipeline.png)

---

## Architecture

L'infrastructure tourne entièrement dans Docker Compose avec six services :

- PostgreSQL (port 5432) — stocke les métadonnées MLflow
- MinIO (port 9000/9001) — stocke les artefacts et modèles sérialisés
- MLflow (port 5000) — interface de tracking et Model Registry
- FastAPI (port 8000) — sert les prédictions en production
- Spark Master (port 7077/8080) — orchestration du cluster Spark
- Spark Worker (port 8081) — exécution des jobs Spark

---

## Prérequis

- Python 3.10+
- Java 11+
- Scala 2.12+
- SBT 1.9+
- Docker 20.10+ et Docker Compose 2.0+
- pip 23.0+
- 4 Go de RAM minimum, 10 Go de disque

---

## Installation

Cloner le projet et créer l'environnement virtuel :

```bash
git clone https://github.com/votre-utilisateur/projet9-mlops.git
cd projet9-mlops

python3 -m venv projet9-env
source projet9-env/bin/activate
pip install -r requirements.txt
```

Démarrer l'infrastructure :

```bash
docker compose up -d
docker compose ps
```

Tous les services doivent apparaître en état `Up (healthy)`.

---

## Utilisation

### 1. Variables d'environnement

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
```

### 2. Pipeline Python — Scikit-learn

Entraîner les modèles :

```bash
python3 spark-training/train.py
```

```
Train : 614 lignes  |  Test : 154 lignes

LogisticRegression   → Accuracy: 0.7143  ROC AUC: 0.8230
RandomForest         → Accuracy: 0.7597  ROC AUC: 0.8147
GradientBoosting     → Accuracy: 0.7532  ROC AUC: 0.8389
SVM                  → Accuracy: 0.7532  ROC AUC: 0.7924
```

Promouvoir le meilleur modèle :

```bash
python3 spark-training/promote_model.py
```

```
Meilleur modèle : v7 (GradientBoosting) — ROC AUC : 0.8389
Version 7 promue en Production.
```

### 3. Pipeline Scala — Apache Spark MLlib

Compiler et assembler le JAR :

```bash
cd spark-training
sbt compile
sbt assembly
```

```
[success] Built: target/scala-2.12/diabetes-mlops-assembly-1.0.jar
```

Lancer le pipeline Spark :

```bash
export DATA_PATH=../data/raw/diabetes.csv

spark-submit \
  --class com.mlops.TrainingPipeline \
  --master local[*] \
  --driver-memory 2g \
  --conf spark.hadoop.fs.s3a.endpoint=http://localhost:9000 \
  --conf spark.hadoop.fs.s3a.access.key=minioadmin \
  --conf spark.hadoop.fs.s3a.secret.key=minioadmin123 \
  --conf spark.hadoop.fs.s3a.path.style.access=true \
  target/scala-2.12/diabetes-mlops-assembly-1.0.jar
```

```
Spark version : 3.5.1  |  Master : local[*]

Données chargées : 768 lignes, 9 colonnes
Split : Train 645 lignes | Test 123 lignes

LogisticRegression  → Accuracy: 0.7642   ROC AUC: 0.8619
RandomForest        → Accuracy: 0.7805   ROC AUC: 0.8462
GradientBoosting    → Accuracy: 0.7642   ROC AUC: 0.8157

Terminé. Résultats disponibles sur : http://localhost:5000
Expérience : diabetes_classification_scala
```

### 4. Vérifier que l'API est prête

```bash
curl http://localhost:8000/ready
```

```json
{
  "status": "ready",
  "model_loaded": true,
  "model_name": "diabetes_model",
  "model_version": "7"
}
```

### 5. Faire une prédiction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 8,
    "glucose": 183.0,
    "blood_pressure": 64.0,
    "skin_thickness": 0.0,
    "insulin": 0.0,
    "bmi": 23.3,
    "diabetes_pedigree": 0.672,
    "age": 32
  }'
```

```json
{
  "prediction": 1,
  "class_name": "diabétique",
  "probability_diabetic": 0.7772,
  "probability_healthy": 0.2228,
  "risk_level": "Élevé",
  "model_version": "7"
}
```

---

## Pipeline de données

| Étape | Outil | Description |
|-------|-------|-------------|
| Chargement Python | pd.read_csv() | Lecture du CSV (768 patients) |
| Chargement Spark | spark.read.csv() | Lecture distribuée du CSV |
| Séparation | train_test_split / randomSplit | 80% train / 20% test |
| Normalisation | StandardScaler | Moyenne=0, écart-type=1 |
| Vectorisation | VectorAssembler | Assemblage des features Spark |

### Features utilisées

| Feature | Description | Unité |
|---------|-------------|-------|
| Pregnancies | Nombre de grossesses | — |
| Glucose | Concentration en glucose | mg/dL |
| BloodPressure | Pression artérielle diastolique | mmHg |
| SkinThickness | Épaisseur du pli cutané | mm |
| Insulin | Taux d'insuline sérique | µU/mL |
| BMI | Indice de masse corporelle | kg/m² |
| DiabetesPedigreeFunction | Score d'antécédents familiaux | — |
| Age | Âge de la patiente | années |

---

## Résultats des modèles

### Pipeline Python — Scikit-learn
Expérience MLflow : `diabetes_classification`

| Modèle | Version MLflow | Accuracy | F1 Score | ROC AUC |
|--------|---------------|----------|----------|---------|
| LogisticRegression | v1, v5 | 0.7143 | 0.6234 | 0.8230 |
| RandomForest | v2, v6 | 0.7597 | 0.6789 | 0.8147 |
| GradientBoosting | v3, v7 | 0.7532 | 0.6701 | 0.8389 |
| SVM | v4, v8 | 0.7532 | 0.6701 | 0.7924 |

### Pipeline Scala — Apache Spark MLlib
Expérience MLflow : `diabetes_classification_scala`

| Modèle | Accuracy | ROC AUC |
|--------|----------|---------|
| LogisticRegression | 0.7642 | 0.8619 |
| RandomForest | 0.7805 | 0.8462 |
| GradientBoosting | 0.7642 | 0.8157 |

Le modèle retenu en Production est GradientBoosting v7 (Python), sélectionné automatiquement pour son meilleur ROC AUC global : 0.8389.

### Paramètres du modèle retenu

| Paramètre | Valeur |
|-----------|--------|
| Algorithme | GradientBoostingClassifier |
| n_estimators | 100 |
| learning_rate | 0.1 |
| max_depth | 3 |
| random_state | 42 |

---

## API REST

Base URL : `http://localhost:8000`

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | /health | Health check du service |
| GET | /ready | Vérifie que le modèle est chargé |
| POST | /api/v1/predict | Prédiction pour un patient |
| POST | /api/v1/predict/batch | Prédiction pour plusieurs patients |

### Niveaux de risque

| Niveau | Probabilité diabétique |
|--------|----------------------|
| Faible | moins de 30% |
| Modéré | entre 30% et 60% |
| Élevé | plus de 60% |

---

## Interfaces disponibles

| Interface | URL | Identifiants |
|-----------|-----|-------------|
| API REST | http://localhost:8000 | — |
| Swagger UI | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |
| Spark Master UI | http://localhost:8080 | — |
| Spark Worker UI | http://localhost:8081 | — |

---

## Structure du projet

```
projet9-mlops/
├── data/
│   └── raw/
│       └── diabetes.csv
├── spark-training/
│   ├── train.py
│   ├── promote_model.py
│   ├── build.sbt
│   ├── project/
│   │   └── plugins.sbt
│   └── src/
│       └── main/
│           └── scala/
│               └── com/
│                   └── mlops/
│                       ├── TrainingPipeline.scala
│                       ├── DataPreprocessor.scala
│                       └── ModelEvaluator.scala
├── api-serving/
│   └── app/
│       ├── main.py
│       ├── routes/
│       │   ├── predict.py
│       │   └── health.py
│       └── services/
│           └── mlflow_service.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Technologies

| Technologie | Version | Rôle |
|-------------|---------|------|
| Python | 3.10 | Pipeline ML principal |
| Scala | 2.12 | Pipeline ML distribué |
| Scikit-learn | 1.3+ | Entraînement des modèles Python |
| Apache Spark | 3.5.1 | Entraînement distribué (MLlib) |
| SBT | 1.9.9 | Build tool Scala |
| Pandas | 2.0+ | Manipulation des données |
| MLflow | 2.11 | Tracking et Model Registry |
| FastAPI | 0.100+ | API REST |
| PostgreSQL | 15 | Métadonnées MLflow |
| MinIO | Latest | Stockage des artefacts |
| Docker Compose | 2.0+ | Containerisation |

---

## Auteur

Oussema ARIFET — Projet MLOps 2026

## Licence

MIT