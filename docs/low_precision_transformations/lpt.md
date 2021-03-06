# OpenVINO™ Low Precision Transformations

## Table of Contents

1. [Introduction](#introduction)  
   1.1. [Low precision transformations pipeline](#low-precision-transformations-pipeline)  
      [Step #1: decomposition](#step-1-decomposition)  
      [Step #2: dequantization operations handling](#step-2-dequantization-operations-handling)  
      [Step #3: cleanup result model](#step-3-cleanup-result-model)  
2. [Plugin transformation pipeline](#plugin-transformation-pipeline)
3. [Input model requirements](#input-model-requirements)
4. [Result model overview](#result-model-overview)
5. [Customization](#customization)
6. [Transformations](#transformations)  
   6.1. [Branch specific transformations](#branch-specific-transformations)  
   6.2. [Decomposition transformations](#decomposition-transformations)  
   6.3. [Main transformations](#main-transformations)  
   6.4. [Cleanup transformations](#cleanup-transformations)  

## Introduction
The goal of `Low Precision Transformations` (LPT) is transform quantized model from original precisions (FP16 or FP32) to low precision (INT8) model to prepare model for low precision inference in OpenVINO™ plugin. As result, operation input tensor precisions will be changed from original to low precision and operations can be inferred by OpenVINO™ plugin in low precision.

### Low precision transformations pipeline
LPT handles input model operation by operation. For each operation there are specific `low precision transformation`. Each transformation uses nGraph pattern matching and is triggered on specific operations pattern. There are several transformation groups, in one group pattern mather is unique per transformation. Decomposition transformation decompose `FakeQuantize` to quantize and dequantization operations. Dequantization operations from previous transformation result is used for the current one and so on, until the end of the model.

Usually, as result all operations are inferred by plugin in low precision. If plugin doesn't support an operation inference in low precision, then corresponding LPT transformation can be disabled and input tensor precisions for the operation will be not changed. In this case the operation is inferred in original precision. 

Low precision transformations pipeline includes three common steps:
* Step #1: decomposition.
* Step #2: dequantization operations handling.
* Step #3: cleanup result model.

### Step #1: decomposition
In this step LPT decomposes each `FakeQuantize` operation to quantize operation (with low precision output) and dequantization operations (revers operations to quantize, with low precision input and original precision output). For dequantization operations LPT uses three operations: `Convert`, `Subtract` and `Multiply`. Element-wise operations have constants on the second branches. This step is implemented in [decomposition transformations](#decomposition-transformations).

Original `FakeQuantize`:  
![TODO: FakeQuantize operation before LPT](img/fq.common.png)

`FakeQuantize` after decomposition to quantize and dequantization operations:   
![TODO: FakeQuantize operation after LPT](img/fq.transformed.png)

### Step #2: dequantization operations handling
In this step LPT moves dequantization operations through existing model operations as more as possible. This step is implemented in [branch specific transformations](#branch-specific-transformations) and [main transformations](#main-transformations).

Original `Convolution` operation with dequantization operations before:  
![TODO: Convolution operation before LPT](img/dq_and_convolution.common.png)

`Convolution` operation after decomposition to quantize and dequantization operations:   
![TODO: Convolution operation after LPT](img/dq_and_convolution.transformed.png)

### Step #3: cleanup result model
In this step LPT cleans up the result model to avoid not handled dequantization operations: fuse dequantization operations if possible or fuse at least `Convert` operations if not. This step is implemented in [cleanup transformations](#cleanup-transformations)

`FakeQuantize` operation with not handled dequantization operations:  
![TODO: FakeQuantize operation with dequantization operations before LPT](img/fakequantize_and_dq.common.png)

`FakeQuantize` operation with fused dequantization operations:  
![TODO: FakeQuantize operation with fused operations after LPT](img/fakequantize_and_dq.transformed.png)

## Plugin transformation pipeline
Typical transformation pipeline:
* Prerequisites: pass manager creation:
```cpp
ngraph::pass::Manager manager;
```
* Step #1: low precision transformation prerequisites:
```cpp
const bool useLpt =
    (conf.lpTransformsMode == Config::LPTransformsMode::On) &&
    ngraph::pass::low_precision::LowPrecisionTransformer::isFunctionQuantized(nGraphFunc);
if (useLpt) {
    manager.register_pass<ngraph::pass::DisableConvertConstantFoldingOnConstPath>(
        std::vector<ngraph::element::Type>{ ngraph::element::i8, ngraph::element::u8 });
}
```
* Step #2: common transformations and operation set conversion:
```cpp
manager.register_pass<ngraph::pass::CommonOptimizations>();
manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
...
if (useLpt) {
        pass_config->set_callback<ngraph::pass::ConvertQuantizeDequantize>([](const_node_ptr &node) -> bool {
            return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node);
        });

        pass_config->set_callback<ngraph::pass::ConvertSubtract>([](const_node_ptr &node) -> bool {
            return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(node);
        });
    }
manager.run_passes(nGraphFunc);
```
* Step #3: low precision transformations
```cpp
if (useLpt) {
    ngraph::pass::Manager manager;
    auto lptPrerequisites = manager.register_pass<ngraph::pass::GraphRewrite>();
    const std::vector<ngraph::element::Type> supportedTypes = { ngraph::element::i8, ngraph::element::u8 };
    lptPrerequisites->add_matcher<PullReshapeThroughDequantization>(supportedTypes);
    lptPrerequisites->add_matcher<PullTransposeThroughDequantization>(supportedTypes);
    lptPrerequisites->add_matcher<ngraph::pass::LinOpSequenceFusion>();
    manager.run_passes(nGraphFunc);

    auto params = LayerTransformation::Params(
        true,
        LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        LayerTransformation::QuantizedTensorAlignment::None,
        true);
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params)
        .add<ConvolutionTransformation, ngraph::opset1::Convolution>(LayerTransformation::Params(params).setPrecisionsOnActivations({ngraph::element::u8}).setSupportAsymmetricQuantization(true))
        .add<GroupConvolutionTransformation, ngraph::opset1::GroupConvolution>(LayerTransformation::Params(params).setPrecisionsOnActivations({ ngraph::element::u8 }).setSupportAsymmetricQuantization(true))
        .addStandaloneCleanup<MultiplyToGroupConvolutionTransformation, graph::opset1::Multiply>(
            LayerTransformation::Params(params).setPrecisionsOnActivations({ ngraph::element::u8 })));

    transformer.transform(nGraphFunc);
}
```

* Step #4: plugin specific transformations
```cpp
ngraph::pass::Manager deviceSpecificManager;
deviceSpecificManager.register_pass<ngraph::pass::device::ConvertOpSet1ToDeviceSpecific>();
deviceSpecificManager.run_passes(nGraphFunc);
```
## Input model requirements

Input model has to be quantized. `FakeQuantize` operation has to be used on activations and on weights for quantization. In this case `FakeQuantize` operations will be added in model before existing model operations and during `FakeQuantize` decomposition, dequantization operations are created. The operation is handled by main LPT transformation if there are dequantization operations on activations and on weights (if it's applicable).

### Low precision tools
There are two tools to quantize a model:
1. [Post-Training Optimization Toolkit](https://docs.openvinotoolkit.org/latest/pot_README.html) (POT)
2. [Neural Network Compression Framework](https://github.com/openvinotoolkit/nncf) (NNCF)

Low precision transformations can handle ONNX quantized model.

### Requirements on activations
There are two supported patterns on activations:  
1. `FakeQuantize` on activations approach  
Input model:  
![](img/fq_and_convolution.common.png)
LPT result model:  
![](img/fq_and_convolution.transformed.png)

2. Quantize and dequantization operations on activations approach  
Input model:  
![](img/qdq_and_convolution.common.png)
LPT result model:  
![](img/qdq_and_convolution.transformed.png)

### Requirements on weights
There are two supported patterns on weights approach:
1. `FakeQuantize` on weights approach  
Input model:  
![](img/fq_and_convolution.common.png)  

LPT result model:  
![](img/fq_and_convolution.transformed.png)  

2. Quantized `Constant` and dequantization operations on weights approach:  
Input model:  
![](img/qdq_and_convolution.common.png)  

LPT result model:  
![](img/qdq_and_convolution.transformed.png)  

## Result model overview
Result model depends on different factors:
* The original model quantization possibility and quantization quality. For some models, some operations are not possible to be quantized by POT and NNCF tools. In this case `FakeQuantize` operations are absent before these operations and they will be inferred in original precision.
* LPT customization and plugin supported operations. If plugin doesn't support INT8 inference for some operation then corresponding LPT transformation should be disabled and the operation will be inferred in original precision.

Let explore quantized [TensorFlow* implementation of ResNet-50](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) model. Use [Model Downloader](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/downloader) tool to download the model from [OpenVINO™ Toolkit - Open Model Zoo repository](https://github.com/openvinotoolkit/open_model_zoo) :
```sh
./downloader.py --name resnet-50-tf --precisions FP16-INT8
```
If you infer the model in OpenVINO™ CPU plugin, then LPT result model key features are:
* All `FakeQuantize` operations are decomposed and have INT8 outputs.
* All dequantization operations were handled, moved thought `MaxPool`, `Convolution` and fused with next `FakeQuantize`. As result all input tensor precisions, except one not quantized `SoftMax` operation at the end of the model, were changed to INT8.
> Note, please:
> 1. LPT transformation for `Add` operation keeps one input branch in FP32. 
> 2. `Add` operation with one constant branch after `Convolution` is, as expected, still in FP32. It's implementation bias values adding and indivisible part of CPU plugin convolution operation implementation.
> 3. `FakeQuantize` is quantization operation and has FP32 input as expected.

As result all operations (except not quantized `SoftMax` at the end of the model) in OpenVINO™ CPU plugin are inferred in low precision. Note, please, in the result model there are `FakeQuantize` operations in FP32 but the plugin responsibility is fuse these operations with previous operations. OpenVINO™ CPU plugin achieves low precision inference for all operations by fusing INT8 `Convolution` with FP32 output with `FakeQuantize` operation with FP32 inputs. 

[TODO: put image here](img/result_model.png)

## Customization
Transformations can be customizable, each transformation (unless otherwise noted) supports following options:
* Update precisions. Transformation member name is `updatePrecisions`. Boolean value is supported: `true` or `false`. All transformations are affected. If `true` then low precision transformations update precisions to low precision and doesn't if `false`. Typically this option is used for plugin debugging.
* Support asymmetric quantization. Transformation member name is `supportAsymmetricQuantization`. Used in `ConvolutionTransformation` and `GroupConvolution` transformations for weights only. Operation with zero point on weights will be not handled if value is `false`.  
* Precisions on activations. Transformation member name is `precisionsOnActivations`. Array of precisions which define result input precisions for transformed operation.
* Precisions on weights. Transformation member name is `precisionsOnWeights`. Array of precisions which define result input precisions for transformed operation.
* Dequantization precision. Transformation member name is `deqPrecision`.
* Support 3D tensor on activations flag. Transformation member name is ``

## Transformations

LPT transformations are grouped in 4 different groups:
1. branch specific transformations,
2. decomposition transformations,
3. main transformations,
4. cleanup transformations.

It's important to group transformations and use groups in this order. Transformation order inside group doesn't matter.

### Branch specific transformations
Key feature of branch specific transformations is handling several operations from different branches in one time. This transformations update several `FakeQuantize` operations and doesn't need their composition before. As result branch specific transformations have to be executed in the pipeline beginning. There are the following branch specific transformations:
* [ConcatMultiChannelsTransformation](movement/concat_multi_channels.md)
* [ConcatTransformation](movement/concat.md)

For example, on the picture below `ConcatTransformation` analyzes all `Concat` operations on different branches which are connected together in cascade and can be inferred in low precision. As result in one `ConcatTransformation` execution several `FakeQuantize` operations are updated. Find out more about `ConcatTransformation` at the link above.

[PICTURE]

After branch specific transformation all other transformations work with one operation only.

### Decomposition transformations
Decomposition transformations decompose `FakeQuantize` operation to quantize (`FakeQuantize` with low precision output) and dequantization operations (`Convert`, `Multiply` and `Subtract`) and have to be executed before other transformations (except branch specific). There are the following decomposition transformations:
* [FakeQuantizeDecompositionTransformation](quantization/fake_quantize_decomposition.md)

After decomposition transformation all other transformations work with dequantization operations. 

### Main transformations
LPT main transformations move dequantization operations through model operations and thus prepare model operation inference in low precision. There are the following main transformations: 
* AddTransformation
* AvgPoolTransformation
* ClampTransformation
* [ConvolutionTransformation](convolution/convolution.md)
* DepthToSpaceTransformation
* FakeQuantizeTransformation
* [GroupConvolutionTransformation](convolution/group_convolution.md)
* InterpolateTransformation
* MatMulTransformation
* MaxPoolTransformation
* MultiplyTransformation
* MVNTransformation
* NormalizeL2Transformation
* PReluTransformation
* ReluTransformation
* ReshapeTransformation
* SqueezeTransformation
* StridedSliceTransformation
* TransposeTransformation
* UnsqueezeTransformation
* InterpolateTransformation

### Cleanup transformations
LPT cleanup transformations is final LPT stage. The goal of these transformations is fusing existing dequantization operations to other model operations to cleanup result model.
* FoldConvertTransformation
* FuseConvertTransformation
* FuseSubtractToFakeQuantizeTransformation
* FuseMultiplyToFakeQuantizeTransformation
* MultiplyToGroupConvolutionTransformation
* SubtractMultiplyToMultiplyAddTransformation

