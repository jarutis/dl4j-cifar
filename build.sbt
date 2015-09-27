name := "DL4J Cifar Spark example"

version := "0.0.1"

scalaVersion := "2.10.4"

resolvers ++= Seq(
  "releases"  at "https://oss.sonatype.org/content/repositories/releases"
)

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.2-SNAPSHOT" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.5.2-SNAPSHOT" % "provided",
  "org.apache.spark" %% "spark-sql" % "1.5.2-SNAPSHOT" % "provided",
  "org.apache.spark" %% "spark-hive" % "1.5.2-SNAPSHOT" % "provided",
  "org.deeplearning4j" % "dl4j-spark-ml" % "0.4-rc3.4-SNAPSHOT" excludeAll(
    ExclusionRule(organization = "org.apache.spark"),
    ExclusionRule(organization = "org.apache.hadoop")
  ),
  "com.twelvemonkeys.imageio" % "imageio-core" % "3.1.1",
  "org.nd4j" % "nd4j-x86" % "0.4-rc4.4-SNAPSHOT",
  "org.slf4j" % "slf4j-api" % "1.7.5"
)

assemblyJarName in assembly := "cifar.jar"

mergeStrategy in assembly := {
  case x if x.startsWith("META-INF") => MergeStrategy.discard
  case x if x.endsWith(".html") => MergeStrategy.discard
  case x if x.contains("slf4j-api") => MergeStrategy.last
  case _ => MergeStrategy.first
}
