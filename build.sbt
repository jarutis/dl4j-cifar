name := "DL4J Cifar Spark example"

version := "0.0.1"

scalaVersion := "2.10.4"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.4.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.4.1" % "provided",
  "org.apache.spark" %% "spark-sql" % "1.4.1" % "provided",
  "org.apache.spark" %% "spark-hive" % "1.4.1" % "provided",
  "org.deeplearning4j" % "dl4j-spark-ml" % "0.4-rc3.2" excludeAll(
    ExclusionRule(organization = "org.apache.spark"),
    ExclusionRule(organization = "org.apache.hadoop")
  ),
  "org.nd4j" % "nd4j-x86" % "0.4-rc3.2"
)

assemblyJarName in assembly := "cifar.jar"

mergeStrategy in assembly := {
  case x if x.startsWith("META-INF") => MergeStrategy.discard
  case x if x.endsWith(".html") => MergeStrategy.discard
  case x if x.contains("slf4j-api") => MergeStrategy.last
  case _ => MergeStrategy.first
}
