package lt.jjarutis.cifar

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import org.apache.commons.io.FileUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.feature.StandardScalerModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.util.MLLibUtil
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

object CifarClassificationSpark {
  val rows = 32
  val columns = 32
  val channels = 3
  val seed = 123
  val outputNum = 10
  val batchSize = 500
  val iterations = 1
  val labels = Map(
    "airplane" -> 0.0,
    "bird" -> 1.0,
    "deer" -> 2.0,
    "frog" -> 3.0,
    "ship" -> 4.0,
    "automobile" -> 5.0,
    "cat" -> 6.0,
    "dog" -> 7.0,
    "horse" -> 8.0,
    "truck" -> 9.0
  )

  def main(args: Array[String]) {
    val sparkConf = new SparkConf()
      .setExecutorEnv(Array(("LD_LIBRARY_PATH", "/home/jjarutis/lib/:/usr/lib/hadoop/lib/native/"),
                            ("PATH", "/opt/spark-1.4.1/bin:/home/jjarutis/lib:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin")))
      .setAppName("Cifar classification example")

    val sc = new SparkContext(sparkConf)
    val hiveCtx = new HiveContext(sc)
    import hiveCtx.implicits._

    val cifar = hiveCtx.read.parquet("cifar/vectorized")

    val scaler = new StandardScaler().fit(cifar.map(_.getAs[Vector]("image_vector")))

    val normalised = cifar.map { row =>
      new LabeledPoint(
        label = labels(row.getAs[String]("label")),
        features = scaler.transform(row.getAs[Vector]("image_vector")))
    }

//    val Array(train, test) = normalised.randomSplit(Array(0.6, 0.4))
    val train = normalised
    val test = normalised
    val trainLayer = new SparkDl4jMultiLayer(sc, conf.SceneConfiguration.getConfiguration)
    System.out.println("train size: " + train.count())
    //fit on the training set
    val trainedNetwork = trainLayer.fit(train, 100)
    val trainedNetworkWrapper = new SparkDl4jMultiLayer(sc, trainedNetwork)

    val predictionAndLabels = test.map { point =>
      val prediction = trainedNetworkWrapper.predict(point.features)
      var max = 0.0
      var idx = 0.0
      for(i <- 0 until prediction.size) {
        if(prediction.apply(i) > max) {
          idx = i.toDouble
          max = prediction.apply(i)
        }
      }
      (idx, point.label)
    }

    System.out.println("Saving model...");

    val bos = new BufferedOutputStream(new FileOutputStream("model.bin"));
    Nd4j.write(bos,trainedNetwork.params());
    FileUtils.write(new File("conf.yaml"),trainedNetwork.conf().toYaml());

    predictionAndLabels.toDF("pred", "label").write.parquet("pred")

    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.fMeasure
    System.out.println("F1 = " + precision)
  }
}
