﻿// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/low_precision/network_helper.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <queue>

#include <ngraph/rt_info.hpp>
#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/common/dequantization_op.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

// Return true if `type` can be castable to at least one of `type`
bool NetworkHelper::is_castable_to_one_of(NodeTypeInfo type, const std::unordered_set<NodeTypeInfo>& types) {
    for (auto another : types) {
        if (type.is_castable(another)) {
            return true;
        }
    }
    return false;
}

// Collect and return a vector with all nodes that consumes any of the `node` output
std::vector<Input<Node>> NetworkHelper::consumer_inputs(std::shared_ptr<Node> node) {
    std::vector<Input<Node>> result;
    for (const auto& output_port : node->outputs()) {
        for (const auto &input : output_port.get_target_inputs()) {
            result.push_back(input);
        }
    }
    return result;
}

std::vector<std::shared_ptr<Node>> NetworkHelper::consumers(std::shared_ptr<Node> node) {
    auto inputs = consumer_inputs(node);
    std::vector<std::shared_ptr<Node>> result(inputs.size());
    std::transform(inputs.begin(), inputs.end(), result.begin(), [](Input<Node> input){ return input.get_node()->shared_from_this(); });
    return result;
}

int NetworkHelper::onWeightsInDepth(std::shared_ptr<Node> layer) {
    const std::vector<std::shared_ptr<Node>> children = consumers(layer);
    for (std::shared_ptr<Node> child : children) {
        if ((is_type<opset1::Convolution>(child) ||
            is_type<opset1::GroupConvolution>(child) ||
            is_type<opset1::MatMul>(child) ||
            (is_type<opset1::Multiply>(child) && (is_type<opset1::Constant>(layer->get_input_node_shared_ptr(0))))) &&
            (child->inputs().size() >= 2lu)) {
            const std::vector<std::shared_ptr<Node>> parents = getParentsRecursivelyExceptTypes(child, {}, 1);
            for (const std::shared_ptr<Node>& parent : parents) {
                if (parent.get() == layer.get()) {
                    return 1;
                }
            }
            return -1;
        }

        const int result = onWeightsInDepth(child);
        if (result != 0) {
            return result;
        }
    }
    return 0;
}

bool NetworkHelper::onWeights(std::shared_ptr<Node> layer) {
    const int result = onWeightsInDepth(layer);
    return result == 1;
}

size_t NetworkHelper::getOutputChannelsCount(std::shared_ptr<const Node> layer, bool isOnWeights) {
    if (layer->outputs().size() == 0) {
        THROW_TRANSFORMATION_EXCEPTION << "Layer " << layer->get_friendly_name() << " doesn't have output tensors";
    }

    if (layer->outputs().size() > 1) {
        THROW_TRANSFORMATION_EXCEPTION << "Layer " << layer->get_friendly_name() << " has too many output tensors, expected one";
    }

    PartialShape shape = layer->get_output_partial_shape(0);
    if (shape.rank() == 0) {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid dimensions count (0) in output of " << layer->get_friendly_name() << " layer on weights";
    }
    if (isOnWeights) {
        return shape[0].get_length();
    } else {
        if (shape.rank() == 1) {
            return shape[0].get_length();
        }
        return shape[1].get_length();
    }
}

std::vector<std::shared_ptr<Node>> NetworkHelper::getParentsRecursivelyExceptTypes(
        std::shared_ptr<Node> layer,
        const std::unordered_set<NodeTypeInfo>& exceptionLayerTypes,
        const int portIndex) {
    std::vector<std::shared_ptr<Node>> parents;
    size_t i = 0ul;
    for (auto input : layer->inputs()) {
        if ((portIndex == -1) || (portIndex == i)) {
            auto parent = input.get_source_output().get_node_shared_ptr();
            if (is_castable_to_one_of(parent->get_type_info(), exceptionLayerTypes)) {
                const std::vector<std::shared_ptr<Node>> tmpParents = getParentsRecursivelyExceptTypes(parent, exceptionLayerTypes);
                parents.insert(parents.end(), tmpParents.begin(), tmpParents.end());
            } else {
                parents.push_back(parent);
            }
        }

        i++;
    }
    return parents;
}

size_t NetworkHelper::getInputChannelsCount(std::shared_ptr<Node> layer) {
    if (layer->get_input_size() == 0) {
        THROW_TRANSFORMATION_EXCEPTION << "There are no input layers";
    }

    PartialShape shape = layer->get_input_partial_shape(0);
    if (shape.rank().get_length() <= 1) {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid dimensions count (0) in input of " << layer->get_friendly_name();
    }

    return shape[1].get_length();
}

size_t NetworkHelper::getGroupsCount(std::shared_ptr<Node> layer) {
    if (as_type_ptr<opset1::Convolution>(layer)) {
        return 1;
    } else if (auto group_convolution = as_type_ptr<opset1::GroupConvolution>(layer)) {
        return layer->get_input_shape(1)[0];    // input weights for opset1::GC is in format GOI..., see the specification
    } else {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid layer type of " << layer->get_friendly_name() << "; expected Convolutino or GroupConvolution";
    }
}

// Assumin tensor in NC... layout, append necessary number of 1s to shape to align it to a give rank
Shape NetworkHelper::alignShapeForChannelDim(const Shape& shape, Rank rank) {
    assert(shape.size() == 1);
    assert(rank.is_static());
    Shape result = shape;
    result.resize(rank.get_length() - 1, 1);
    return result;
}

void NetworkHelper::removeLayer(std::shared_ptr<Node> layer) {
    ngraph::replace_output_update_name(layer->output(0), layer->input_value(0));
}

std::shared_ptr<Node> NetworkHelper::swapMultiplyAndAdd(std::shared_ptr<opset1::Add> addAfterMultiply, const int multiplyBranch) {
    // Multiply --> Add(addAfterMultiply)  ==>  Add(new) --> Multiply(new)
    // That means x*a + b ==> (x + b/a)*a; tries to fold b/a
    auto multiply = addAfterMultiply->get_input_node_shared_ptr(multiplyBranch);

    auto multiplyParent1 = multiply->get_input_node_shared_ptr(0);
    auto multiplyParent2 = multiply->get_input_node_shared_ptr(1);

    auto multiplyInput = as_type_ptr<opset1::Multiply>(multiplyParent1);
    auto multiplyConst = as_type_ptr<opset1::Constant>(multiplyParent2);
    int multiplyInputBranch = 0;

    if (multiplyConst == nullptr) {
        multiplyInput = as_type_ptr<opset1::Multiply>(multiplyParent2);
        multiplyConst = as_type_ptr<opset1::Constant>(multiplyParent1);
        multiplyInputBranch = 1;
    }

    if (multiplyConst == nullptr)
        return addAfterMultiply;

    auto x = multiply->get_input_node_shared_ptr(multiplyInputBranch);
    auto a = multiply->get_input_node_shared_ptr(multiplyInputBranch == 0 ? 1 : 0);
    auto b = addAfterMultiply->get_input_node_shared_ptr(multiplyBranch == 0 ? 1 : 0);
    auto bDivA = fold<opset1::Divide>(b, a);

    std::vector<std::shared_ptr<Node>> inputs{ {}, {} };

    inputs[0] = x;
    inputs[1] = bDivA;

    std::shared_ptr<opset1::Add> newAdd = std::make_shared<op::TypeRelaxed<opset1::Add>>(
        std::vector<element::Type>{element::f32, element::f32}, std::vector<element::Type>{ element::f32 },
        ngraph::op::TemporaryReplaceOutputType(inputs[0], element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(inputs[1], element::f32).get());
    NetworkHelper::setOutDataPrecision(newAdd, addAfterMultiply->get_output_element_type(0));

    auto newMultiply = std::make_shared<DequantizationMultiply>(newAdd, a);
    replace_node(addAfterMultiply, newMultiply);

    copyInfo(addAfterMultiply, newAdd);
    return newMultiply;
}

bool NetworkHelper::isScalarLike(std::shared_ptr<opset1::Constant> constant) {
    return constant->get_all_data_elements_bitwise_identical();
}

std::shared_ptr<opset1::Constant> NetworkHelper::toScalar(std::shared_ptr<opset1::Constant> constant) {
    assert(isScalarLike(constant));
    return std::make_shared<opset1::Constant>(constant->get_element_type(), Shape{}, constant->get_data_ptr());
}

std::shared_ptr<Node> NetworkHelper::getConstantInput(std::shared_ptr<Node> node) {
    std::shared_ptr<Node> constant1 = as_type_ptr<opset1::Constant>(node->input_value(0).get_node_shared_ptr());
    if (!constant1) {
        constant1 = as_type_ptr<opset1::Constant>(node->input_value(1).get_node_shared_ptr());
    }
    return constant1;
}

std::shared_ptr<ngraph::opset1::Multiply> NetworkHelper::optimizeMultipliesAfter(std::shared_ptr<Node> node) {
    std::shared_ptr<ngraph::opset1::Multiply> multiply = as_type_ptr<opset1::Multiply>(node);
    if (!multiply) {
        THROW_IE_LPT_EXCEPTION(*multiply) << "Unexpected operation type";
    }

    if (multiply->output(0).get_target_inputs().size() == 1) {
        auto constant1 = getConstantInput(multiply);
        if (!constant1 || constant1->output(0).get_target_inputs().size() != 1) {
            return multiply;
        }
        auto nextMultiplyInput = *multiply->output(0).get_target_inputs().begin();
        auto nextMultiply = as_type_ptr<opset1::Multiply>(nextMultiplyInput.get_node()->shared_from_this());
        if (nextMultiply) {
            auto constant2 = getConstantInput(nextMultiply);
            auto constant2Inputs = constant2->output(0).get_target_inputs().size();
            if (!constant2 || constant2->output(0).get_target_inputs().size() != 1) {
                return multiply;
            }

            auto newConst = fold<opset1::Multiply>(constant1, constant2);
            auto newMultiply =
                    std::make_shared<opset1::Multiply>(
                            multiply->input_value(1 - constant1->output(0).get_target_inputs().begin()->get_index()),
                            newConst->output(0));
            replace_node(nextMultiply, newMultiply);
            return newMultiply;
        }
    }

    return nullptr;
}

std::shared_ptr<opset1::Constant> NetworkHelper::roundWithTolerance(std::shared_ptr<Node> node, element::Type target_type, float tolerance) {
    auto constant = as_type_ptr<opset1::Constant>(node);
    assert(constant);
    auto values = constant->cast_vector<float>();

    auto castedConstant = as_type_ptr<opset1::Constant>(fold<opset1::Convert>(constant, target_type));
    auto castedValues = castedConstant->cast_vector<float>();

    // TODO: implement with constant folding when ReduceAnd constant folding is ready
    if (std::equal(values.begin(), values.end(), castedValues.begin(), [tolerance](float a, float b) { return fabs(a - b) < tolerance; })) {
        return castedConstant;
    }

    auto round = [](
        const std::shared_ptr<opset1::Constant>& constant,
        element::Type target_type,
        float tolerance,
        std::vector<float>& values,
        float increaseValue) -> std::shared_ptr<opset1::Constant> {
        const auto castedConstant = as_type_ptr<opset1::Constant>(fold<opset1::Convert>(
            fold<opset1::Add>(constant, std::make_shared<opset1::Constant>(constant->get_output_element_type(0), Shape{ 1 }, increaseValue)),
            target_type));
        const auto castedValues = castedConstant->cast_vector<float>();
        if (std::equal(values.begin(), values.end(), castedValues.begin(), [tolerance](float a, float b) { return fabs(a - b) < tolerance; })) {
            return castedConstant;
        }

        return nullptr;
    };

    castedConstant = round(constant, target_type, tolerance, values, 0.5f);
    if (castedConstant != nullptr) {
        return castedConstant;
    }

    castedConstant = round(constant, target_type, tolerance, values, -0.5f);
    if (castedConstant != nullptr) {
        return castedConstant;
    }

    castedConstant = round(constant, target_type, tolerance, values, 1.f);
    if (castedConstant != nullptr) {
        return castedConstant;
    }

    return constant;
}

std::shared_ptr<Node> NetworkHelper::fold_fake_quantize(const std::shared_ptr<opset1::FakeQuantize>& fq) {
    return foldFakeQuantize(fq, false, false);
}

std::shared_ptr<Node> NetworkHelper::fold_fake_quantize(const std::shared_ptr<opset1::FakeQuantize>& fq, const bool roundValues) {
    return foldFakeQuantize(fq, roundValues, true);
}

std::shared_ptr<Node> NetworkHelper::foldFakeQuantize(
    const std::shared_ptr<opset1::FakeQuantize>& fq,
    const bool roundValuesArg,
    const bool roundValuesWasSet) {
    if (is_type<opset1::Constant>(fq->get_input_node_shared_ptr(0)) &&
        is_type<opset1::Constant>(fq->get_input_node_shared_ptr(1)) &&
        is_type<opset1::Constant>(fq->get_input_node_shared_ptr(2)) &&
        is_type<opset1::Constant>(fq->get_input_node_shared_ptr(3)) &&
        is_type<opset1::Constant>(fq->get_input_node_shared_ptr(4)) &&
        op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(1)), 0.f) &&
        op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(2)), 254.f) &&
        op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(3)), -127.f) &&
        op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(4)), 127.f)) {
        return fold<opset1::Add>(fq->input_value(0), fq->input_value(3));
    }

    auto constant = as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(0));

    if (constant) {
        const bool roundValues = roundValuesWasSet ? roundValuesArg : fq->output(0).get_element_type().is_integral();

        Shape constShape = fq->get_output_shape(0);
        if (constShape.empty() || constShape.size() > 5lu) {
            THROW_IE_LPT_EXCEPTION(*fq) << "Unexpected dimensions count " << constShape.size();
        }

        // OIDHW
        const size_t OC = constShape[0];
        const size_t IC = constShape.size() > 1lu ? constShape[1] : 1;
        const size_t D = constShape.size() > 4lu ? constShape[constShape.size() - 3] : 1;
        const size_t H = constShape.size() > 2lu ? constShape.size() == 3lu ? constShape[2] : constShape[constShape.size() - 2] : 1;
        const size_t W = constShape.size() > 3lu ? constShape[constShape.size() - 1] : 1;

        const auto inputLowValues = as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(1))->cast_vector<float>();
        const auto inputHighValues = as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(2))->cast_vector<float>();
        const auto outputLowValues = as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(3))->cast_vector<float>();
        const auto outputHighValues = as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(4))->cast_vector<float>();

        const size_t inputLowSize = inputLowValues.size();
        const size_t inputHighSize = inputHighValues.size();
        const size_t outputLowSize = outputLowValues.size();
        const size_t outputHighSize = outputHighValues.size();

        const bool isInputLowBroadcasted = inputLowSize != OC;
        if ((inputLowSize != 1) && (inputLowSize != OC)) {
            THROW_IE_LPT_EXCEPTION(*fq) << "Unexpected input low values count " << inputLowSize << " for " << OC << " channels";
        }
        const bool isInputHighBroadcasted = inputHighSize != OC;
        if ((inputHighSize != 1) && (inputHighSize != OC)) {
            THROW_IE_LPT_EXCEPTION(*fq) << "Unexpected input high values count " << inputHighSize << " for " << OC << " channels";
        }
        const bool isOutputLowBroadcasted = outputLowSize != OC;
        if ((outputLowSize != 1) && (outputLowSize != OC)) {
            THROW_IE_LPT_EXCEPTION(*fq) << "Unexpected output low values count " << outputLowSize << " for " << OC << " channels";
        }
        const bool isOutputHighBroadcasted = outputHighSize != OC;
        if ((outputHighSize != 1) && (outputHighSize != OC)) {
            THROW_IE_LPT_EXCEPTION(*fq) << "Unexpected output high values count " << outputHighSize << " for " << OC << " channels";
        }

        auto levels_1 = fq->get_levels() - 1.f;

        //const size_t DHW = D * H * W;
        const size_t IDHW = IC * D * H * W;

        const auto values = constant->cast_vector<float>();
        std::vector<float> quantizedValues(OC * IC * D * H * W);

        for (int oc = 0; oc < OC; ++oc) {
            for (int iidx = 0; iidx < IDHW; ++iidx) {
                const float inputLow = inputLowValues[isInputLowBroadcasted ? 0 : oc];
                const float inputHigh = inputHighValues[isInputHighBroadcasted ? 0 : oc];
                const float outputLow = outputLowValues[isOutputLowBroadcasted ? 0 : oc];
                const float outputHigh = outputHighValues[isOutputHighBroadcasted ? 0 : oc];

                const size_t idx = oc * IDHW + iidx;

                if (values[idx] <= inputLow) {
                    quantizedValues[idx] = roundValues ? std::roundf(outputLow) : outputLow;
                } else if (values[idx] > inputHigh) {
                    quantizedValues[idx] = roundValues ? std::roundf(outputHigh) : outputHigh;
                } else {
                    const float value = std::roundf((values[idx] - inputLow) / (inputHigh - inputLow) * levels_1) /
                        levels_1 * (outputHigh - outputLow) + outputLow;
                    quantizedValues[idx] = roundValues ? std::roundf(value) : value;
                }
            }
        }

        return std::make_shared<opset1::Constant>(fq->get_output_element_type(0), constShape, quantizedValues);
    }

    return fq;
}

// Decompose FakeQuantize to FakeQuantize with output integer limits (quantize), dequatized MultiplyAdd
// To align types the resulting sequence is FakeQuantize -> Convert -> Convert -> MultiplyAdd
std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> NetworkHelper::decomposeFakeQuantize(
    std::shared_ptr<opset1::FakeQuantize> fq,
    const element::Type precision,
    const float min,
    const float max,
    const bool hasZeroPoint,
    const bool updatePrecision) {
    using std::make_shared;

    const auto newMin = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, min);
    const auto newMax = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, max);

    const auto outputLow = fq->input_value(3);
    const auto outputHigh = fq->input_value(4);

    // TODO: threshold values have to used here to avoid shifts
    const std::shared_ptr<Node> scale = fold<opset1::Divide>(
        fold<opset1::Subtract>(outputHigh, outputLow),
        fold<opset1::Subtract>(newMax, newMin));

    std::shared_ptr<Node> shift = hasZeroPoint ?
        fold<opset1::Divide>(
            fold<opset1::Subtract>(fold<opset1::Multiply>(newMin, outputHigh), fold<opset1::Multiply>(newMax, outputLow)),
            fold<opset1::Subtract>(outputHigh, outputLow)) :
        nullptr;

    // Build a substitution sub-graph:

    std::shared_ptr<ngraph::Node> newFQ = fold_fake_quantize(std::make_shared<op::TypeRelaxed<opset1::FakeQuantize>>(
        fq->input_value(0),
        fq->input_value(1),
        fq->input_value(2),
        newMin->output(0),
        newMax->output(0),
        fq->get_levels(),
        fq->get_auto_broadcast()), true);
    // TODO: for debuging only - remove later
    newFQ->set_friendly_name(fq->get_friendly_name() + "_original");

    if (shift != nullptr) {
        std::shared_ptr<opset1::Constant> shiftConst = as_type_ptr<opset1::Constant>(shift);
        if (isScalarLike(shiftConst)) {
            auto scalar = toScalar(shiftConst);
            if (op::util::constantIsEqualTo(scalar, 0)) {
                shift = nullptr;
            }
        }
    }

    std::shared_ptr<ngraph::Node> convert2;
    if (updatePrecision) {
        std::shared_ptr<Node> convert;
        std::shared_ptr<opset1::Constant> newFqConstant = as_type_ptr<opset1::Constant>(newFQ);

        if (is_type<opset1::Constant>(newFQ)) {
            convert = fold<opset1::Convert>(newFQ, precision);
        } else if (is_type<opset1::FakeQuantize>(newFQ)) {
            newFQ = setOutDataPrecision(as_type_ptr<opset1::FakeQuantize>(newFQ), precision);
            convert = newFQ;
        } else {
            THROW_IE_LPT_EXCEPTION(*newFQ) << "unexpected operation type";
        }

        convert2 = std::make_shared<DequantizationConvert>(convert, fq->get_output_element_type(0));
        convert2->set_friendly_name(convert->get_friendly_name() + "_Convert");
    }

    // TODO: why type relaxed?
    const std::shared_ptr<ngraph::Node> sub = shift == nullptr ?
        nullptr :
        std::make_shared<ngraph::op::TypeRelaxed<DequantizationSubtract>>(convert2 == nullptr ? newFQ : convert2, shift);
    if (sub != nullptr) {
        sub->set_friendly_name(newFQ->get_friendly_name() + "_Subtract");
    }

    const std::shared_ptr<ngraph::opset1::Multiply> dequantize = std::make_shared<DequantizationMultiply>(
        sub == nullptr ? (convert2 == nullptr ? newFQ : convert2) : sub,
        scale);
    dequantize->set_friendly_name(newFQ->get_friendly_name() + "_Multiply");

    replace_node(fq, dequantize);

    return std::make_tuple(newFQ, dequantize);
}

std::shared_ptr<opset1::FakeQuantize> NetworkHelper::updateFakeQuantize(
    std::shared_ptr<opset1::FakeQuantize> fq,
    element::Type precision,
    float min,
    float max) {
    auto newMin = std::make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, min);
    auto newMax = std::make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, max);

    std::shared_ptr<opset1::FakeQuantize> newFQ = std::make_shared<ngraph::op::TypeRelaxed<opset1::FakeQuantize>>(
            fq->input_value(0),
            fq->input_value(1),
            fq->input_value(2),
            newMin->output(0),
            newMax->output(0),
            fq->get_levels(),
            fq->get_auto_broadcast());

    NetworkHelper::setOutDataPrecision(newFQ, precision);
    replace_node(fq, newFQ);

    newFQ->set_friendly_name(fq->get_friendly_name());
    return newFQ;
}

FakeQuantizeDequantization NetworkHelper::makeDequantization(
    const float dequantizationMul,
    const float dequantizationSub,
    const ngraph::element::Type originalPrecision,
    const ngraph::Shape dataNodeOutputShape,
    element::Type precision,
    float min,
    float max) {
    // TODO: we create input here! we really need it here?
    const std::shared_ptr<opset1::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(precision, dataNodeOutputShape);
    std::shared_ptr<ngraph::Node> parent = input;

    // TODO: convert should be optional: where is updatePrecision?
    std::shared_ptr<DequantizationConvert> convert;
    {
        convert = std::make_shared<DequantizationConvert>(
            input,
            originalPrecision);
        parent = convert;
    }

    std::shared_ptr<DequantizationSubtract> subtract;
    if (dequantizationSub != 0.f) {
        subtract = std::make_shared<ngraph::op::TypeRelaxed<DequantizationSubtract>>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(originalPrecision, ngraph::Shape({}), std::vector<float>({ dequantizationSub })));
        subtract->set_output_type(0, originalPrecision, subtract->get_output_partial_shape(0));
        parent = subtract;
    }

    // mandatory
    std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<DequantizationMultiply>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(originalPrecision, ngraph::Shape({}), std::vector<float>({ dequantizationMul })));

    return FakeQuantizeDequantization(input, convert, subtract, multiply);
}

FakeQuantizeDequantization NetworkHelper::createDequantizationFromFakeQuantize(
    std::shared_ptr<opset1::FakeQuantize> fq,
    element::Type precision,
    float min,
    float max,
    const bool hasZeroPoint,
    const bool updatePrecision) {
    using std::make_shared;

    const ngraph::element::Type_t fqPrecision = fq->get_output_element_type(0);
    auto newMin = make_shared<opset1::Constant>(fqPrecision, Shape{}, min);
    auto newMax = make_shared<opset1::Constant>(fqPrecision, Shape{}, max);

    auto outputLow = fq->input_value(3);
    auto outputHigh = fq->input_value(4);

    // TODO: threshold values have to used here to avoid shifts

    const std::shared_ptr<Node> scale = fold<opset1::Divide>(
        fold<opset1::Subtract>(outputHigh, outputLow),
        fold<opset1::Subtract>(newMax, newMin));

    std::shared_ptr<Node> shift = hasZeroPoint ?
        fold<opset1::Divide>(
            fold<opset1::Subtract>(fold<opset1::Multiply>(newMin, outputHigh), fold<opset1::Multiply>(newMax, outputLow)),
            fold<opset1::Subtract>(outputHigh, outputLow)) :
        nullptr;

    if (shift != nullptr) {
        std::shared_ptr<opset1::Constant> shiftConst = as_type_ptr<opset1::Constant>(shift);
        if (isScalarLike(shiftConst)) {
            auto scalar = toScalar(shiftConst);
            if (op::util::constantIsEqualTo(scalar, 0)) {
                shift = nullptr;
            }
        }
    }

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, fq->get_output_shape(0));
    const std::shared_ptr<ngraph::opset1::Convert> convert = std::make_shared<DequantizationConvert>(
        input,
        fq->get_output_element_type(0));

    const std::shared_ptr<ngraph::opset1::Subtract> subtract = shift == nullptr ?
        nullptr :
        make_shared<ngraph::op::TypeRelaxed<DequantizationSubtract>>(convert, shift);
    if (subtract != nullptr) {
        subtract->set_output_type(0, fq->get_output_element_type(0), subtract->get_output_partial_shape(0));
    }

    const std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<DequantizationMultiply>(
        subtract == nullptr ? static_cast<std::shared_ptr<Node>>(convert) : subtract,
        scale);

    return FakeQuantizeDequantization(fq, convert, subtract, multiply);
}

FakeQuantizeDequantization NetworkHelper::getDequantization(const std::shared_ptr<Node> node, const size_t parentIndex) {
    auto getDataIndex = [](const std::shared_ptr<ngraph::Node>& node) {
        if (is_type<opset1::Constant>(node->get_input_node_ptr(1))) {
            return 0ul;
        } else {
            return 1ul;
        }
    };

    Output<Node> dataNode = node->input_value(parentIndex);

    const std::shared_ptr<ngraph::opset1::Multiply> multiply =
        (dataNode.get_node_shared_ptr()->get_input_size() > 1ul) &&
        (is_type<opset1::Constant>(dataNode.get_node_shared_ptr()->get_input_node_ptr(1)) ||
            is_type<opset1::Constant>(dataNode.get_node_shared_ptr()->get_input_node_ptr(0))) ?
        as_type_ptr<ngraph::opset1::Multiply>(dataNode.get_node_shared_ptr()) :
        nullptr;

    if (multiply != nullptr) {
        dataNode = multiply->get_input_source_output(getDataIndex(multiply));
    }

    const std::shared_ptr<opset1::Subtract> subtract =
        (dataNode.get_node_shared_ptr()->get_input_size() > 1ul) &&
        (is_type<opset1::Constant>(dataNode.get_node_shared_ptr()->get_input_node_ptr(1)) ||
            is_type<opset1::Constant>(dataNode.get_node_shared_ptr()->get_input_node_ptr(0))) ?
        as_type_ptr<ngraph::opset1::Subtract>(dataNode.get_node_shared_ptr()) :
        nullptr;

    if (subtract != nullptr) {
        dataNode = subtract->get_input_source_output(getDataIndex(subtract));
    }

    const std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(dataNode.get_node_shared_ptr());
    if (convert != nullptr) {
        dataNode = convert->get_input_source_output(0);
    }

    return FakeQuantizeDequantization(dataNode, convert, subtract, multiply);
}

FakeQuantizeDequantizationValues NetworkHelper::createEmptyValues(const FakeQuantizeDequantization& dequantization) {
    std::shared_ptr<Node> parent = dequantization.convert ? dequantization.convert : dequantization.data.get_node_shared_ptr();

    std::shared_ptr<Node> multiply1Const = dequantization.multiply ?
        dequantization.multiply->get_input_node_shared_ptr(1)->clone_with_new_inputs({}) :
        std::make_shared<opset1::Constant>(parent->get_output_element_type(0), Shape({}), std::vector<float>({ 1.f }));

    std::shared_ptr<Node> subtract1Const = dequantization.subtract ?
        dequantization.subtract->get_input_node_shared_ptr(1)->clone_with_new_inputs({}) :
        std::make_shared<opset1::Constant>(parent->get_output_element_type(0), Shape({}), std::vector<float>({ 0.f }));

    subtract1Const->set_output_type(0, multiply1Const->get_output_element_type(0), subtract1Const->get_output_partial_shape(0));

    return FakeQuantizeDequantizationValues(subtract1Const, multiply1Const);
}

bool NetworkHelper::isZeroConst(const std::shared_ptr<Node>& node) {
    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(node);

    if (constant == nullptr)
        return false;

    if (NetworkHelper::isScalarLike(constant)) {
        auto scalar = NetworkHelper::toScalar(constant);
        if (op::util::constantIsEqualTo(scalar, 0)) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

std::shared_ptr<Node> NetworkHelper::optimizeSubtract(std::shared_ptr<opset1::Subtract> subtract) {
    auto convertOnSubtract = subtract->input_value(0).get_node_shared_ptr();
    if (as_type_ptr<opset1::Convert>(convertOnSubtract) == nullptr) {
        return subtract;
    }

    // TODO: replace assert to condition and omit conversion part if there is no convert
    // TODO: also check convertInputType to understand if we really want to propagate type
    assert(as_type_ptr<opset1::Convert>(convertOnSubtract));
    const element::Type convertInputType = convertOnSubtract->get_input_element_type(0);
    const element::Type convertOutputType = convertOnSubtract->get_output_element_type(0);

    if (!convertOutputType.is_real()) {
        return subtract;
    }

    auto data = convertOnSubtract->input_value(0);
    auto shift = subtract->input_value(1).get_node_shared_ptr();
    auto roundedShift = NetworkHelper::roundWithTolerance(shift, convertInputType);

    std::shared_ptr<Node> replacement;
    if (roundedShift->get_element_type() == convertInputType) {
        // Propagate convertInputType down
        replacement = std::make_shared<op::TypeRelaxed<opset1::Subtract>>(data, roundedShift);
        NetworkHelper::copyInfo(subtract, replacement);
        NetworkHelper::setOutDataPrecisionForTypeRelaxed(replacement, convertOutputType);
        replace_node(subtract, replacement);
    }

    // We lose the tail conversion here; not needed if the next node is a TypeRelaxed
    // TODO: check cases when Convert should be preserved

    // Try to optimize Add out if constant is zero
    // TODO: don't remove operation here: don't create this Subtraction operation in FQ decomposition
    // if (isScalarLike(roundedShift)) {
    //    auto scalar = distillToScalar(roundedShift);
    //    if (op::util::constantIsEqualTo(scalar, 0)) {
    //        replace_node(replacement, replacement->input_value(0).get_node_shared_ptr());
    //        replacement = nullptr;
    //    }
    // }

    return replacement;
}

NetworkHelper::InsertDequantizationResult NetworkHelper::moveDequantizationAfter(
    const std::shared_ptr<ngraph::Node>& operation,
    const FakeQuantizeDequantization& dequantization,
    const bool updatePrecision,
    const bool moveSubtract) {
    std::vector<Output<Node>> inputs(operation->get_input_size());
    for (size_t i = 0; i < operation->get_input_size(); ++i) {
        inputs[i] = operation->get_input_node_shared_ptr(i);
    }

    const size_t dequantizationIndex = getInputIndex(dequantization.multiply, operation);
    inputs[dequantizationIndex] = moveSubtract ?
        dequantization.data :
        (dequantization.subtract == nullptr ? dequantization.data : dequantization.subtract);

    const std::shared_ptr<ngraph::Node> newOperation = operation->clone_with_new_inputs(inputs);
    newOperation->set_friendly_name(operation->get_friendly_name());
    // copyInfo(operation, newOperation);

    if (updatePrecision) {
        auto op = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(newOperation);
        if (op == nullptr) {
            THROW_IE_LPT_EXCEPTION(*newOperation) << "not possible to update precision for not TypeRelaxedBase operation";
        }
        op->set_overridden_output_type(newOperation->get_input_element_type(0));
        std::dynamic_pointer_cast<ngraph::Node>(newOperation)->validate_and_infer_types();
    }

    const bool shouldConvert = (newOperation->get_output_element_type(0) != dequantization.multiply->get_output_element_type(0));

    auto parent = newOperation;
    if (shouldConvert) {
        parent = std::make_shared<DequantizationConvert>(parent, dequantization.convert->get_output_element_type(0));
    }
    if (moveSubtract && (dequantization.subtract != nullptr)) {
        auto subtractConstant = dequantization.subtract->get_input_node_shared_ptr(1);
        parent = std::make_shared<DequantizationSubtract>(parent, subtractConstant);
    }
    if (dequantization.multiply != nullptr) {
        auto multiplyConstant = dequantization.multiply->get_input_node_shared_ptr(1);
        parent = std::make_shared<DequantizationMultiply>(parent, multiplyConstant);
    }
    replace_node(operation, parent);

    if ((!moveSubtract) && (dequantization.convert != nullptr) && (dequantization.subtract != nullptr)) {
        optimizeSubtract(dequantization.subtract);
    }

    return InsertDequantizationResult(newOperation, parent);
}

void NetworkHelper::removeConvertIfPossible(
    const std::shared_ptr<ngraph::Node>& operation,
    const FakeQuantizeDequantization& dequantization) {
    const element::Type precisionBeforeConvert = dequantization.convert->input(0).get_element_type();

    if (checkConstantValuePrecision(precisionBeforeConvert, dequantization.subtract->get_input_node_shared_ptr(1))) {
        auto newSubtract = dequantization.subtract->clone_with_new_inputs({
            dequantization.convert->get_input_node_shared_ptr(0),
            fold<opset1::Convert>(dequantization.subtract->get_input_node_shared_ptr(1), precisionBeforeConvert) });
        replace_node(dequantization.subtract, newSubtract);
    }
}

bool NetworkHelper::checkConstantValuePrecision(const element::Type expectedPrecision, const std::shared_ptr<Node>& constant) {
    if (expectedPrecision.is_signed()) {
        return true;
    }

    std::shared_ptr<opset1::Constant> constantOp = as_type_ptr<opset1::Constant>(constant);
    if (constantOp == nullptr) {
        return false;
    }

    const auto values = constantOp->cast_vector<float>();
    const bool convertCanBeRemoved =
        (expectedPrecision.is_signed() || (std::all_of(values.begin(), values.end(), [](const float value) { return value >= 0.f; })));
    return convertCanBeRemoved;
}

size_t NetworkHelper::getInputIndex(const std::shared_ptr<ngraph::Node>& parent, const std::shared_ptr<ngraph::Node>& child) {
    for (size_t i = 0; i < child->get_input_size(); ++i) {
        if (parent.get() == child->get_input_node_ptr(i)) {
            return i;
        }
    }
    THROW_IE_LPT_EXCEPTION(*child) << " input index for " << parent->get_friendly_name() << " was not found";
}

std::vector<Output<Node>> NetworkHelper::getInputs(const std::shared_ptr<ngraph::Node>& node) {
    std::vector<Output<Node>> inputs(node->get_input_size());
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        inputs[i] = node->get_input_node_shared_ptr(i);
    }
    return inputs;
}

std::shared_ptr<Node> NetworkHelper::toScalarIfPossible(std::shared_ptr<Node> node) {
    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(node);
    if (constant == nullptr) {
        return node;
    }

    if (!NetworkHelper::isScalarLike(constant)) {
        return node;
    }

    return NetworkHelper::toScalar(constant);
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
