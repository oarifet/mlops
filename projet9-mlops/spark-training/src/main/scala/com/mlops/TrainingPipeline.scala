package com.mlops

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{
  LogisticRegression,
  RandomForestClassifier,
  GBTClassifier
}
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.RunStatus
import org.slf4j.LoggerFactory

object TrainingPipeline {

  private val logger = LoggerFactory.getLogger(getClass)

  // ── Configuration ──────────────────────────────────────────────────────────
  val MLFLOW_URI  = sys.env.getOrElse("MLFLOW_TRACKING_URI",    "http://localhost:5000")
  val S3_ENDPOINT = sys.env.getOrElse("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
  val ACCESS_KEY  = sys.env.getOrElse("AWS_ACCESS_KEY_ID",      "minioadmin")
  val SECRET_KEY  = sys.env.getOrElse("AWS_SECRET_ACCESS_KEY",  "minioadmin123")
  val DATA_PATH   = sys.env.getOrElse("DATA_PATH",
    "../data/raw/diabetes.csv")
  val EXP_NAME    = "diabetes_classification_scala"

  def main(args: Array[String]): Unit = {

    // ── Session Spark ────────────────────────────────────────────────────────
    println("\n🔥 Initialisation de la session Spark...")
    logger.info("Démarrage de la session Spark...")

    val spark = SparkSession.builder()
      .appName("DiabetesPrediction-Scala")
      .master("local[*]")
      .config("spark.sql.shuffle.partitions",          "4")
      .config("spark.driver.memory",                   "2g")
      .config("spark.hadoop.fs.s3a.endpoint",          S3_ENDPOINT)
      .config("spark.hadoop.fs.s3a.access.key",        ACCESS_KEY)
      .config("spark.hadoop.fs.s3a.secret.key",        SECRET_KEY)
      .config("spark.hadoop.fs.s3a.path.style.access", "true")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    println(s"   ✅ Spark version : ${spark.version}")
    println(s"   ✅ Master        : ${spark.sparkContext.master}")
    logger.info(s"Spark démarré — version ${spark.version}")

    // ── Chargement et prétraitement ──────────────────────────────────────────
    val rawDf              = DataPreprocessor.loadData(spark, DATA_PATH)
    val (_, transformedDf) = DataPreprocessor.buildPreprocessingPipeline(rawDf)
    val (trainDf, testDf)  = DataPreprocessor.splitData(transformedDf)

    // ── Modèles Spark MLlib ──────────────────────────────────────────────────
    val models = Seq(

      ("LogisticRegression", new LogisticRegression()
        .setLabelCol("Outcome")
        .setFeaturesCol("features")
        .setMaxIter(100)
        .setRegParam(0.01)),

      ("RandomForest", new RandomForestClassifier()
        .setLabelCol("Outcome")
        .setFeaturesCol("features")
        .setNumTrees(100)
        .setSeed(42L)),

      ("GradientBoosting", new GBTClassifier()
        .setLabelCol("Outcome")
        .setFeaturesCol("features")
        .setMaxIter(100)
        .setMaxDepth(3)
        .setStepSize(0.1)
        .setSeed(42L))
    )

    // ── MLflow Client ────────────────────────────────────────────────────────
    val client = new MlflowClient(MLFLOW_URI)

    val experimentId = {
      val expOpt = client.getExperimentByName(EXP_NAME)
      if (expOpt.isPresent) expOpt.get().getExperimentId
      else client.createExperiment(EXP_NAME)
    }

    logger.info(s"Expérience MLflow : $EXP_NAME (ID: $experimentId)")
    println(s"\n🚀 Lancement des expériences MLflow...")
    println("-" * 65)

    // ── Entraînement de chaque modèle ────────────────────────────────────────
    models.foreach { case (modelName, model) =>

      val runId = client.createRun(experimentId).getRunId
      logger.info(s"Run démarré pour $modelName : $runId")

      try {
        // Tags
        client.setTag(runId, "model_name",    modelName)
        client.setTag(runId, "framework",     "Spark MLlib Scala")
        client.setTag(runId, "spark_version", spark.version)
        client.setTag(runId, "language",      "Scala 2.12")
        client.setTag(runId, "dataset",       "Pima Indians Diabetes")

        // Paramètres
        client.logParam(runId, "model_type",  modelName)
        client.logParam(runId, "train_size",  trainDf.count().toString)
        client.logParam(runId, "test_size",   testDf.count().toString)
        client.logParam(runId, "spark_master", spark.sparkContext.master)

        // Entraînement
        val startTime   = System.currentTimeMillis()
        val fittedModel = model.fit(trainDf)
        val trainTime   = (System.currentTimeMillis() - startTime) / 1000.0

        client.logParam(runId, "training_time_seconds", f"$trainTime%.2f")

        // Prédictions et métriques
        val predictions = fittedModel.transform(testDf)
        val metrics     = ModelEvaluator.evaluate(predictions)

        // Log métriques dans MLflow
        metrics.foreach { case (name, value) =>
          client.logMetric(runId, name, value)
        }

        // Affichage matrice de confusion
        ModelEvaluator.printConfusionMatrix(predictions)

        // Affichage résumé
        ModelEvaluator.printMetrics(modelName, metrics)

        client.setTerminated(runId, RunStatus.FINISHED)
        logger.info(s"✅ $modelName terminé — ROC AUC: ${metrics("roc_auc")}")

      } catch {
        case e: Exception =>
          println(s"❌ Erreur $modelName : ${e.getMessage}")
          logger.error(s"Erreur pour $modelName : ${e.getMessage}", e)
          client.setTerminated(runId, RunStatus.FAILED)
      }
    }

    println("-" * 65)

    // ── Arrêt Spark ──────────────────────────────────────────────────────────
    spark.stop()
    println("\n🔥 Session Spark arrêtée proprement")
    println(s"✅ Terminé ! Voir les résultats sur : $MLFLOW_URI")
    println(s"   Expérience : $EXP_NAME")
    logger.info("Pipeline terminé avec succès")
  }
}
