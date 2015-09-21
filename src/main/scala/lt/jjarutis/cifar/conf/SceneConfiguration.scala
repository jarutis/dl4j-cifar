package lt.jjarutis.cifar.conf

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.lossfunctions.LossFunctions


object SceneConfiguration {
  val rows = 32
  val columns = 32
  val channels = 3
  val seed = 123
  val outputNum = 10
  val batchSize = 500
  val iterations = 1

  def getConfiguration = {
    val builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .batchSize(batchSize)
      .iterations(iterations).regularization(true)
      .l1(1e-1).l2(2e-4).useDropConnect(true)
      .constrainGradientToUnitNorm(true).miniBatch(true)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .list(6)
      .layer(0, new ConvolutionLayer.Builder(5, 5)
               .nOut(5).dropOut(0.5)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(1, new SubsamplingLayer
               .Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
               .build())
      .layer(2, new ConvolutionLayer.Builder(3, 3)
               .nOut(10).dropOut(0.5)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(3, new SubsamplingLayer
               .Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
               .build())
      .layer(4, new DenseLayer.Builder().nOut(100).activation("relu")
               .build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
               .nOut(outputNum)
               .weightInit(WeightInit.XAVIER)
               .activation("softmax")
               .build())
      .backprop(true).pretrain(false)

    new ConvolutionLayerSetup(builder, rows, columns, channels)

    builder.build()
  }
}
