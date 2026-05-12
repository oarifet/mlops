package com.mlops

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.Pipeline
import org.mlflow.api.proto.Service.RunStatus
import org.mlflow.tracking.MlflowClient
import org.mlflow.spark.SparkModelFlavor
import org.slf4j.LoggerFactory

object TrainingPipeline {

  private val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {

    val mlflowTrackingUri = sys.env.getOrElse("MLFLOW_TRACKING_URI", "http://localhost:5000")
    val dataPath          = sys.env.getOrElse("DATA_PATH", "/opt/data/raw/iris.csv")
    val experimentName    = "iris-classification"

    val spark = SparkSession.builder()
      .appName("MLOps Training Pipeline")
      .master(sys.env.getOrElse("SPARK_MASTER", "local[*]"))
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession démarrée")

    val mlflowClient = new MlflowClient(mlflowTrackingUri)

    val experimentOptional = mlflowClient.getExperimentByName(experimentName)
    val experimentId: String = if (experimentOptional.isPresent) {
      experimentOptional.get().getExperimentId
    } else {
      mlflowClient.createExperiment(experimentName)
    }

    logger.info(s"Expérience MLflow : $experimentName (ID: $experimentId)")

    val activeRun = mlflowClient.createRun(experimentId)
    val runId     = activeRun.getRunId

    logger.info(s"Run MLflow démarré : $runId")

    try {
      val numTrees   = 100
      val maxDepth   = 5
      val trainRatio = 0.8

      mlflowClient.logParam(runId, "num_trees",   numTrees.toString)
      mlflowClient.logParam(runId, "max_depth",   maxDepth.toString)
      mlflowClient.logParam(runId, "train_ratio", trainRatio.toString)
      mlflowClient.logParam(runId, "model_type",  "RandomForestClassifier")
      mlflowClient.logParam(runId, "dataset",     "iris")

      val rawData   = DataPreprocessor.loadData(spark, dataPath)
      val cleanData = DataPreprocessor.cleanData(rawData)
      val (trainData, testData) = DataPreprocessor.splitData(cleanData, trainRatio)

      mlflowClient.logMetric(runId, "train_size", trainData.count().toDouble)
      mlflowClient.logMetric(runId, "test_size",  testData.count().toDouble)

      val preprocessingPipeline = DataPreprocessor.buildPreprocessingPipeline()

      val classifier = new RandomForestClassifier()
        .setLabelCol("label")
        .setFeaturesCol("scaled_features")
        .setNumTrees(numTrees)
        .setMaxDepth(maxDepth)
        .setSeed(42)

      val fullPipeline = new Pipeline()
        .setStages(preprocessingPipeline.getStages :+ classifier)

      logger.info("Entraînement du modèle...")
      val startTime = System.currentTimeMillis()
      val model     = fullPipeline.fit(trainData)
      val trainTime = (System.currentTimeMillis() - startTime) / 1000.0

      logger.info(f"Entraînement terminé en $trainTime%.2f secondes")
      mlflowClient.logMetric(runId, "training_time_seconds", trainTime)

      val predictions = model.transform(testData)
      val metrics     = ModelEvaluator.evaluate(predictions)
      ModelEvaluator.printConfusionMatrix(predictions)

      metrics.foreach { case (name, value) =>
        mlflowClient.logMetric(runId, name, value)
      }

      // ✅ Logger le modèle dans MLflow avec Spark 3.5.1
      logger.info("Logging du modèle dans MLflow...")
      SparkModelFlavor.logModel(
        model,
        "spark-model",
        runId,
        mlflowClient
      )
      logger.info("✅ Modèle loggé dans MLflow")

      mlflowClient.setTag(runId, "model_type", "spark-pipeline")

      mlflowClient.setTerminated(runId, RunStatus.FINISHED)
      logger.info(s"✅ Run MLflow terminé avec succès : $runId")
      logger.info(s"✅ Accuracy : ${metrics("accuracy")}")

    } catch {
      case e: Exception =>
        logger.error(s"Erreur durant l'entraînement : ${e.getMessage}", e)
        mlflowClient.setTerminated(runId, RunStatus.FAILED)
        throw e
    } finally {
      spark.stop()
    }
  }
}
