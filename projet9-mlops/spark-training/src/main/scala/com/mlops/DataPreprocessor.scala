package com.mlops

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.slf4j.LoggerFactory

object DataPreprocessor {

  private val logger = LoggerFactory.getLogger(getClass)

  // ── Features du dataset Diabetes ──────────────────────────────────────────
  val featureCols = Array(
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
  )

  val labelCol    = "Outcome"   // 0 = sain, 1 = diabétique
  val featuresCol = "features"

  // ── Chargement des données ─────────────────────────────────────────────────
  def loadData(spark: SparkSession, path: String): DataFrame = {
    logger.info(s"Chargement des données depuis : $path")

    val df = spark.read
      .option("header",      "true")
      .option("inferSchema", "true")
      .csv(path)

    logger.info(s"Données chargées : ${df.count()} lignes, ${df.columns.length} colonnes")

    println(s"\n📦 Données chargées depuis : $path")
    println(s"   Lignes   : ${df.count()}")
    println(s"   Colonnes : ${df.columns.mkString(", ")}")
    df.show(5)
    df.describe().show()

    df
  }

  // ── Pipeline de prétraitement Spark ML ────────────────────────────────────
  def buildPreprocessingPipeline(df: DataFrame): (PipelineModel, DataFrame) = {
    println("\n⚙️  Construction du Pipeline Spark ML...")
    logger.info("Construction du pipeline de prétraitement...")

    // Étape 1 : VectorAssembler → regroupe les 8 features en un vecteur
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features_raw")
      .setHandleInvalid("skip")

    // Étape 2 : StandardScaler → normalise (moyenne=0, écart-type=1)
    val scaler = new StandardScaler()
      .setInputCol("features_raw")
      .setOutputCol(featuresCol)
      .setWithMean(true)
      .setWithStd(true)

    // Pipeline Spark ML
    val pipeline      = new Pipeline().setStages(Array(assembler, scaler))
    val pipelineModel = pipeline.fit(df)
    val dfTransformed = pipelineModel.transform(df)

    println("   ✅ Pipeline Spark ML appliqué")
    println("   Aperçu après transformation :")
    dfTransformed.select(featuresCol, labelCol).show(5, truncate = false)

    logger.info("Pipeline de prétraitement appliqué avec succès")
    (pipelineModel, dfTransformed)
  }

  // ── Split Train / Test ─────────────────────────────────────────────────────
  def splitData(
    df: DataFrame,
    trainRatio: Double = 0.8,
    seed: Long = 42L
  ): (DataFrame, DataFrame) = {
    println(s"\n✂️  Split Train / Test (80% / 20%)...")
    logger.info(s"Split train/test : ${trainRatio * 100}% / ${(1 - trainRatio) * 100}%")

    val splits = df.randomSplit(Array(trainRatio, 1 - trainRatio), seed)
    val train  = splits(0)
    val test   = splits(1)

    println(s"   Train : ${train.count()} lignes")
    println(s"   Test  : ${test.count()} lignes")
    logger.info(s"Train : ${train.count()} | Test : ${test.count()}")

    (train, test)
  }
}
