import os
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_PATH = os.getenv("DATA_PATH", "/home/oussema/mlops/mlops/projet9-mlops/data/raw/iris.csv")
EXPERIMENT_NAME = "iris-classification"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Spark Session avec support S3
print("🔧 Initialisation de Spark avec support S3...")
spark = SparkSession.builder \
    .appName("MLOps Training Pipeline") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin123") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Charger les données
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

# Preprocessing
label_indexer = StringIndexer(inputCol="species", outputCol="label")
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Classifier
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="scaled_features",
    numTrees=100,
    maxDepth=5,
    seed=42
)

# Pipeline
pipeline = Pipeline(stages=[label_indexer, assembler, scaler, rf])

# Split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# MLflow Run
with mlflow.start_run() as run:
    
    # Log params
    mlflow.log_param("num_trees", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("train_ratio", 0.8)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("dataset", "iris")
    
    # Train
    print("🚀 Entraînement du modèle...")
    model = pipeline.fit(train_data)
    
    # Predict
    predictions = model.transform(test_data)
    
    # Evaluate
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    
    # ✅ Log model dans MLflow avec Spark 3.5.1
    print("📦 Logging du modèle dans MLflow...")
    mlflow.spark.log_model(model, "spark-model")
    
    print(f"✅ Modèle loggé dans MLflow (Run ID: {run.info.run_id})")

spark.stop()
