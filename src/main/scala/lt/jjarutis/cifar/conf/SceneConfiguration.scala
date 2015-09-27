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
  val batchSize = 100
  val iterations = 10

  val cKernel = Array(3, 3)
  val cStride = Array(1, 1)
  val cPadding = Array(1, 1)

  val sKernel = Array(2, 2)
  val sStride = Array(2, 2)

  def getConfiguration = {
    val builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .batchSize(batchSize)
      .iterations(iterations)
      .constrainGradientToUnitNorm(true)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(1e-4)
      .list(16)
      .layer(0, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(64).dropOut(0.3)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(1, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(64)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(2, new SubsamplingLayer
               .Builder(SubsamplingLayer.PoolingType.MAX, sKernel, sStride)
               .build())
      .layer(3, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(128).dropOut(0.4)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(4, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(128)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(5, new SubsamplingLayer
               .Builder(SubsamplingLayer.PoolingType.MAX, sKernel, sStride)
               .build())
      .layer(6, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(256).dropOut(0.4)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(7, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(256).dropOut(0.4)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(8, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(256)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(9, new SubsamplingLayer
               .Builder(SubsamplingLayer.PoolingType.MAX, sKernel, sStride)
               .build())
      .layer(10, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(512).dropOut(0.4)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(11, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(512).dropOut(0.4)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(12, new ConvolutionLayer.Builder(cKernel, cStride, cPadding)
               .nOut(512)
               .weightInit(WeightInit.XAVIER)
               .activation("relu")
               .build())
      .layer(13, new SubsamplingLayer
               .Builder(SubsamplingLayer.PoolingType.MAX, sKernel, sStride)
               .build())
      .layer(14, new DenseLayer.Builder().nOut(512).activation("relu")
               .dropOut(0.5)
               .build())
      .layer(15, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
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
