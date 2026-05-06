package com.mlops

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.PipelineModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._
import org.slf4j.LoggerFactory

object ModelEvaluator {

  private val logger = LoggerFactory.getLogger(getClass)

  /**
   * Calculer toutes les métriques d'évaluation
   */
  def evaluate(predictions: DataFrame): Map[String, Double] = {
    logger.info("Calcul des métriques d'évaluation...")

    // Préparer les données pour MulticlassMetrics
    val predictionAndLabels = predictions
      .select(col("prediction"), col("label"))
      .rdd
      .map(row => (row.getDouble(0), row.getDouble(1)))

    val metrics = new MulticlassMetrics(predictionAndLabels)

    val results = Map(
      "accuracy"          -> metrics.accuracy,
      "weighted_precision" -> metrics.weightedPrecision,
      "weighted_recall"   -> metrics.weightedRecall,
      "weighted_f1"       -> metrics.weightedFMeasure
    )

    // Afficher les métriques
    logger.info("=== Métriques d'évaluation ===")
    results.foreach { case (name, value) =>
      logger.info(f"  $name : $value%.4f")
    }

    results
  }

  /**
   * Afficher la matrice de confusion
   */
  def printConfusionMatrix(predictions: DataFrame): Unit = {
    val predictionAndLabels = predictions
      .select(col("prediction"), col("label"))
      .rdd
      .map(row => (row.getDouble(0), row.getDouble(1)))

    val metrics = new MulticlassMetrics(predictionAndLabels)
    logger.info("=== Matrice de Confusion ===")
    logger.info(metrics.confusionMatrix.toString)
  }
}