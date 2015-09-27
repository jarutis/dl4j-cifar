package lt.jjarutis.cifar

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.hive.HiveContext
import org.deeplearning4j.spark.ml.classification.NeuralNetworkClassification

object CifarClassificationSparkML {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf()
      .setAppName("Cifar classification example")

    val sc = new SparkContext(sparkConf)
    val hiveCtx = new HiveContext(sc)

    val cifar = hiveCtx.read.parquet("cifar/vectorized").repartition(100)
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
      .setConf(conf.AdamSceneConfiguration.getConfiguration)
    val pipeline = new Pipeline().setStages(Array(indexer, scaler, classification))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)
    predictions.write.mode("overwrite").parquet("pred")

    val predictionAndLabels = predictions
      .map(row => (row.getAs[Double]("labelIndex"), row.getAs[Double]("prediction")))

    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val precision = metrics.precision
    val recall = metrics.recall // same as true positive rate
    val f1Score = metrics.fMeasure
    println("Summary Statistics")
    println(s"Precision = $precision")
    println(s"Recall = $recall")
    println(s"F1 Score = $f1Score")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
  }
}
