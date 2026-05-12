package com.mlops

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.{
  BinaryClassificationEvaluator,
  MulticlassClassificationEvaluator
}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._
import org.slf4j.LoggerFactory

object ModelEvaluator {

  private val logger = LoggerFactory.getLogger(getClass)

  // ── Calcul des métriques ───────────────────────────────────────────────────
  def evaluate(predictions: DataFrame): Map[String, Double] = {
    logger.info("Calcul des métriques d'évaluation...")

    // ROC AUC — métrique principale pour classification médicale
    val rocEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol("Outcome")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    // Accuracy
    val accEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Outcome")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // F1 Score
    val f1Evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Outcome")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    // Precision
    val precEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Outcome")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")

    // Recall
    val recEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Outcome")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")

    val results = Map(
      "roc_auc"   -> rocEvaluator.evaluate(predictions),
      "accuracy"  -> accEvaluator.evaluate(predictions),
      "f1_score"  -> f1Evaluator.evaluate(predictions),
      "precision" -> precEvaluator.evaluate(predictions),
      "recall"    -> recEvaluator.evaluate(predictions)
    )

    logger.info("=== Métriques ===")
    results.foreach { case (name, value) =>
      logger.info(f"  $name : $value%.4f")
    }

    results
  }

  // ── Matrice de confusion ───────────────────────────────────────────────────
  def printConfusionMatrix(predictions: DataFrame): Unit = {
    val predAndLabels = predictions
      .select(col("prediction"), col("Outcome").cast("double"))
      .rdd
      .map(row => (row.getDouble(0), row.getDouble(1)))

    val metrics = new MulticlassMetrics(predAndLabels)
    println("\n📊 Matrice de Confusion :")
    println(metrics.confusionMatrix.toString)
    logger.info(s"Matrice de confusion :\n${metrics.confusionMatrix}")
  }

  // ── Affichage résumé ───────────────────────────────────────────────────────
  def printMetrics(modelName: String, metrics: Map[String, Double]): Unit = {
    println(
      f"✅ ${modelName}%-22s" +
      f"→ Accuracy: ${metrics("accuracy")}%.4f  " +
      f"ROC AUC: ${metrics("roc_auc")}%.4f"
    )
  }
}
