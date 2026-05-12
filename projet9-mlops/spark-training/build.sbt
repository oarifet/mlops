name         := "diabetes-mlops"
version      := "1.0"
scalaVersion := "2.12.18"

val sparkVersion  = "3.5.0"
val mlflowVersion = "2.11.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % sparkVersion,
  "org.apache.spark" %% "spark-sql"   % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.mlflow"        % "mlflow-client" % mlflowVersion,
  "org.apache.hadoop" % "hadoop-aws"          % "3.3.4",
  "com.amazonaws"     % "aws-java-sdk-bundle" % "1.12.262",
  "org.slf4j"         % "slf4j-simple" % "1.7.36"
)

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "reference.conf"              => MergeStrategy.concat
  case _                             => MergeStrategy.first
}

mainClass := Some("com.mlops.TrainingPipeline")
