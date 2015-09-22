package lt.jjarutis.cifar

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.hive.HiveContext
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.ml.classification.NeuralNetworkClassification
import org.nd4j.linalg.lossfunctions.LossFunctions

object CifarClassificationSparkML {
  val rows = 32
  val columns = 32
  val channels = 3
  val seed = 123
  val outputNum = 10
  val batchSize = 500
  val iterations = 1

  def main(args: Array[String]) {
    val sparkConf = new SparkConf()
      .setExecutorEnv(Array(("LD_LIBRARY_PATH", "/home/jjarutis/lib/:/usr/lib/hadoop/lib/native/"),
                            ("PATH", "/opt/spark-1.4.1/bin:/home/jjarutis/lib:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin")))
      .setAppName("Cifar classification example")

    val sc = new SparkContext(sparkConf)
    val hiveCtx = new HiveContext(sc)

    val cifar = hiveCtx.read.parquet("cifar/vectorized")
    val Array(trainingData, testData) = cifar.randomSplit(Array(0.7, 0.3))

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")
    val scaler = new StandardScaler()
      .setWithMean(true).setWithStd(true)
      .setInputCol("image_vector").setOutputCol("scaledFeatures")
    val classification = new NeuralNetworkClassification()
      .setLabelCol("labelIndex")
      .setFeaturesCol("scaledFeatures")
      .setConf(conf.SceneConfiguration.getConfiguration)
    val pipeline = new Pipeline().setStages(Array(indexer, scaler, classification))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.write.parquet("pred")
  }
}
