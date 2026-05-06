package com.mlops

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, StringIndexer}
import org.apache.spark.ml.Pipeline
import org.slf4j.LoggerFactory

object DataPreprocessor {

  private val logger = LoggerFactory.getLogger(getClass)

  val numericFeatures = Array(
    "sepal_length", "sepal_width", "petal_length", "petal_width"
  )

  val labelCol    = "species"
  val labelIndex  = "label"
  val featuresCol = "features"
  val scaledCol   = "scaled_features"

  def loadData(spark: SparkSession, path: String): DataFrame = {
    logger.info(s"Chargement des données depuis : $path")
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)
    logger.info(s"Données chargées : ${df.count()} lignes, ${df.columns.length} colonnes")
    df
  }

  def cleanData(df: DataFrame): DataFrame = {
    logger.info("Nettoyage des données...")
    val cleaned = df.dropDuplicates().na.drop()
    logger.info(s"Après nettoyage : ${cleaned.count()} lignes")
    cleaned
  }

  def buildPreprocessingPipeline(): Pipeline = {
    val labelIndexer = new StringIndexer()
      .setInputCol(labelCol)
      .setOutputCol(labelIndex)
      .setHandleInvalid("keep")

    val assembler = new VectorAssembler()
      .setInputCols(numericFeatures)
      .setOutputCol(featuresCol)

    val scaler = new StandardScaler()
      .setInputCol(featuresCol)
      .setOutputCol(scaledCol)
      .setWithMean(true)
      .setWithStd(true)

    new Pipeline().setStages(Array(labelIndexer, assembler, scaler))
  }

  def splitData(df: DataFrame, trainRatio: Double = 0.8, seed: Long = 42): (DataFrame, DataFrame) = {
    logger.info(s"Split train/test : ${trainRatio * 100}% / ${(1 - trainRatio) * 100}%")

    // FIX : utiliser index au lieu de pattern matching Array()
    val splits = df.randomSplit(Array(trainRatio, 1 - trainRatio), seed)
    val train  = splits(0)
    val test   = splits(1)

    logger.info(s"Train : ${train.count()} lignes | Test : ${test.count()} lignes")
    (train, test)
  }
}
