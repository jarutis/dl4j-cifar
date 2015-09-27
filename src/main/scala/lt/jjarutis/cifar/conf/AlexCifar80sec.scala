package lt.jjarutis.cifar.conf

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.lossfunctions.LossFunctions

// http://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-80sec.cfg

object AlexCifar80sec {
  val rows = 32
  val columns = 32
  val channels = 3
  val seed = 123
  val outputNum = 10
  val batchSize = 50
  val iterations = 100

  val cKernel = Array(5, 5)
  val cStride = Array(1, 1)
  val cPadding = Array(2, 2)

  val sKernel = Array(3, 3)
  val sStride = Array(2, 2)

  def getConfiguration = {
    val builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .batchSize(batchSize)
      .iterations(iterations)
      .constrainGradientToUnitNorm(true)
      .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
      .learningRate(1e-4)
      .regularization(true)
      .l2(0.0001)
      .list(8)
      .layer(0, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(32)
               .dropOut(0.3)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(1, new SubsamplingLayer
               .Builder(SubsamplingLayer.PoolingType.MAX, sKernel, sStride)
               .build())
      .layer(2, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(32)
               .dropOut(0.3)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(3, new SubsamplingLayer
               .Builder(SubsamplingLayer.PoolingType.MAX, sKernel, sStride)
               .build())
      .layer(4, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(64)
               .dropOut(0.2)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(5, new SubsamplingLayer
               .Builder(SubsamplingLayer.PoolingType.MAX, sKernel, sStride)
               .build())
      .layer(6, new DenseLayer.Builder().nOut(64).activation("relu")
               .build())
      .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
               .nOut(outputNum)
               .weightInit(WeightInit.XAVIER)
               .activation("softmax")
               .build())
      .backprop(true)
      .pretrain(false)

    new ConvolutionLayerSetup(builder, rows, columns, channels)

    builder.build()
  }
}
