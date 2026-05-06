name := "projet9-mlops"
version := "1.0.0"
scalaVersion := "2.13.12"

val sparkVersion = "3.5.1"

libraryDependencies ++= Seq(
  // Spark
  "org.apache.spark" %% "spark-core"  % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql"   % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",

  // MLflow
  "org.mlflow" % "mlflow-client" % "2.11.1",

  // AWS S3 (pour MinIO)
  "org.apache.hadoop" % "hadoop-aws"          % "3.3.4",
  "com.amazonaws"     % "aws-java-sdk-bundle"  % "1.12.262",

  // Logging
  "org.slf4j"      % "slf4j-api"       % "1.7.36",
  "ch.qos.logback" % "logback-classic" % "1.2.11"
)

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "reference.conf"              => MergeStrategy.concat
  case x                             => MergeStrategy.first
}
