package lt.jjarutis.cifar

import java.io.File
import org.canova.api.split.FileSplit
import org.canova.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator
import org.deeplearning4j.datasets.rearrange.LocalUnstructuredDataFormatter
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.StandardScaler
import org.nd4j.linalg.factory.Nd4j


object CifarClassificationLocal {

  val numRows = 32
  val numColumns = 32
  val nChannels = 3

  val outputNum = 10

  val batchSize = 1000
  val numberOfSamples = 50000
  val iterations = 1
  val seed = 123
  val trainPercentage = 0.8

  val labeledPath = System.getProperty("user.home") + "/data/cifar"
  val testTrainSplitPath = System.getProperty("user.home") + "/data/cifarsplit"

  val meanFile = new File("mean.bin")
  val stdFile = new File("std.bin")
  val trainingFile = new File("train.bin")
  val testingFile = new File("test.bin")

  def main(args: Array[String]) {
    Nd4j.factory().setDType(DataBuffer.Type.FLOAT)
    Nd4j.dtype = DataBuffer.Type.FLOAT
    Nd4j.ENFORCE_NUMERICAL_STABILITY = true

    val nIn = numColumns * numRows * nChannels

    // split raw data into test and train datasets
    System.out.println("Splitting raw dataset!")
    val splitTestTrainRoot = new File(testTrainSplitPath)
    if(!splitTestTrainRoot.exists()) {
      val formatter = new LocalUnstructuredDataFormatter(
        splitTestTrainRoot,
        new File(labeledPath),
        LocalUnstructuredDataFormatter.LabelingType.DIRECTORY,
        trainPercentage
      )
      formatter.rearrange()
    }

    // create training dataset reader
    System.out.println("Creating training reader!")
    val trainReader = new ImageRecordReader(numRows, numColumns, nChannels, true)
    trainReader.initialize(new FileSplit(new File(new File(splitTestTrainRoot, "split"), "train")))

    // create testing dataset reader
    System.out.println("Creating testing reader!")
    val testReader = new ImageRecordReader(numRows, numColumns, nChannels, true)
    testReader.initialize(new FileSplit(new File(new File(splitTestTrainRoot, "split"), "test")))

    // create scaler
    System.out.println("Creating scaler!")
    val scaler = new StandardScaler()
    if(!meanFile.exists() || !stdFile.exists()) {
      val scalerIterator = new RecordReaderDataSetIterator(trainReader, numberOfSamples, nIn, outputNum)
      scaler.fit(scalerIterator.next())
      scaler.save(meanFile, stdFile)
    } else {
      scaler.load(meanFile, stdFile)
    }

    // read train dataset
    System.out.println("Reading train dataset!")
    if(!trainingFile.exists()) {
      val readingIterator = new RecordReaderDataSetIterator(trainReader, numberOfSamples, nIn, outputNum)
      val trainingSet = readingIterator.next()
      trainingSet.save(trainingFile)
    }
    val trainingSet = new DataSet()
    trainingSet.load(trainingFile)

    // read test dataset
    System.out.println("Reading test dataset!")
    if(!testingFile.exists()) {
      val testIterator = new RecordReaderDataSetIterator(trainReader, numberOfSamples, nIn, outputNum)
      val testingSet = testIterator.next()
      testingSet.save(testingFile)
    }
    val testingSet = new DataSet()
    testingSet.load(testingFile)

    // train the network
    System.out.println("Creating training iterator!")
    trainingSet.shuffle()
    val trainIterator = new SamplingDataSetIterator(trainingSet, batchSize, numberOfSamples)

    System.out.println("Creating network!")
    val trainedNetwork = new MultiLayerNetwork(conf.SceneConfiguration.getConfiguration)
    trainedNetwork.init()
    trainedNetwork.setListeners(new ScoreIterationListener(1))

    scaler.transform(trainingSet)
    scaler.transform(testingSet)
    System.out.println("Data scaling done!")
    while(trainIterator.hasNext()) {
      var next = trainIterator.next()
      System.out.println("Loaded data with label distribution " + next.labelCounts())
      trainedNetwork.fit(next)
      System.out.println("Evaluating so far")
      val evaluation = new Evaluation(outputNum)
      val testIterNext = testingSet.sample(100,true)
      evaluation.eval(testIterNext.getLabels(), trainedNetwork.output(testIterNext.getFeatureMatrix(), true))
      System.out.println(evaluation.stats())
      System.out.println("One batch done with score " + trainedNetwork.score())

      System.out.println(evaluation.stats())
    }
  }
}
