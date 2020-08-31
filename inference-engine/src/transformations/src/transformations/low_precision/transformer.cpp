﻿// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/transformer.hpp"
#include "transformations/low_precision/network_helper.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <iostream>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ngraph_ops/type_relaxed.hpp"

// branch specific transformations
#include "transformations/low_precision/concat.hpp"
#include "transformations/low_precision/concat_multi_channels.hpp"

// general transformations
#include "transformations/low_precision/add.hpp"
#include "transformations/low_precision/avg_pool.hpp"
#include "transformations/low_precision/clamp.hpp"
#include "transformations/low_precision/convolution.hpp"
#include "transformations/low_precision/depth_to_space.hpp"
#include "transformations/low_precision/fake_quantize.hpp"
#include "transformations/low_precision/group_convolution.hpp"
#include "transformations/low_precision/interpolate.hpp"
#include "transformations/low_precision/mat_mul.hpp"
#include "transformations/low_precision/max_pool.hpp"
#include "transformations/low_precision/multiply.hpp"
#include "transformations/low_precision/mvn.hpp"
#include "transformations/low_precision/normalize_l2.hpp"
#include "transformations/low_precision/prelu.hpp"
#include "transformations/low_precision/reshape.hpp"
#include "transformations/low_precision/relu.hpp"
#include "transformations/low_precision/squeeze.hpp"
#include "transformations/low_precision/subtract.hpp"
#include "transformations/low_precision/split.hpp"
#include "transformations/low_precision/transpose.hpp"
#include "transformations/low_precision/unsqueeze.hpp"
#include "transformations/low_precision/variadic_split.hpp"

// cleanup transformations
#include "transformations/low_precision/convert.hpp"
#include "transformations/low_precision/fuse_convert.hpp"
#include "transformations/low_precision/fuse_fake_quantize.hpp"
#include "transformations/low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "transformations/low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "transformations/low_precision/multiply_to_group_convolution.hpp"
#include "transformations/low_precision/subtract_multiply_to_multiply_add.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

LowPrecisionTransformations::LowPrecisionTransformations(
    const std::map<std::string, LayerTransformationPtr>& branchSpecificTransformations,
    const std::map<std::string, LayerTransformationPtr>& transformations,
    const std::map<std::string, std::vector<std::pair<std::string, LayerTransformationPtr>>>& cleanupTransformations,
    const std::vector<StandaloneCleanup>& standaloneCleanupTransformations) :
    branchSpecificTransformations(branchSpecificTransformations),
    transformations(transformations),
    cleanupTransformations(cleanupTransformations),
    standaloneCleanupTransformations(standaloneCleanupTransformations) {}

void LowPrecisionTransformations::setUpdatePrecisions(const bool updatePrecisions) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setUpdatePrecisions(updatePrecisions);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setUpdatePrecisions(updatePrecisions);
    }
}

void LowPrecisionTransformations::setQuantizedTensorAlignmentOnActivations(
    const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnActivations) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnActivations(quantizedTensorAlignmentOnActivations);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnActivations(quantizedTensorAlignmentOnActivations);
    }
}

void LowPrecisionTransformations::setQuantizedTensorAlignmentOnWeights(
    const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnWeights) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnWeights(quantizedTensorAlignmentOnWeights);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnWeights(quantizedTensorAlignmentOnWeights);
    }
}

LowPrecisionTransformations& LowPrecisionTransformations::remove(const std::string& operationType) {
    removeBranchSpecificTransformations(operationType);
    removeTransformations(operationType);
    removeCleanupTransformations(operationType);
    return *this;
}

LowPrecisionTransformations& LowPrecisionTransformations::removeBranchSpecificTransformations(const std::string& operationType) {
    branchSpecificTransformations.erase(operationType);
    return *this;
}

LowPrecisionTransformations& LowPrecisionTransformations::removeTransformations(const std::string& operationType) {
    transformations.erase(operationType);
    return *this;
}

LowPrecisionTransformations& LowPrecisionTransformations::removeCleanupTransformations(const std::string& operationType) {
    cleanupTransformations.erase(operationType);
    return *this;
}

std::vector<LayerTransformationPtr> LowPrecisionTransformations::find(const std::string& transformationKey) const {
    auto it = branchSpecificTransformations.find(transformationKey);
    std::vector<LayerTransformationPtr> res;
    if (it != branchSpecificTransformations.end()) {
        res.emplace_back(it->second);
    }

    it = transformations.find(transformationKey);
    if (it != transformations.end()) {
        res.emplace_back(it->second);
    }

    const auto it1 = cleanupTransformations.find(transformationKey);
    if (it1 != cleanupTransformations.end()) {
        for (const auto& transformation : it1->second) {
            res.emplace_back(transformation.second);
        }
    }

    for (const auto& transformation : standaloneCleanupTransformations) {
        if (transformation.typeName == transformationKey) {
            res.emplace_back(transformation.transformation);
        }
    }

    return res;
}

void LowPrecisionTransformations::setParamsManager(IParamsManager* paramsManager) noexcept {
    setParamsManager(paramsManager, branchSpecificTransformations);
    setParamsManager(paramsManager, transformations);
    setParamsManager(paramsManager, cleanupTransformations);
    setParamsManager(paramsManager, standaloneCleanupTransformations);
}

void LowPrecisionTransformations::setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept {
    setLayerTransformationsManager(layerTransformationsManager, branchSpecificTransformations);
    setLayerTransformationsManager(layerTransformationsManager, transformations);
    setLayerTransformationsManager(layerTransformationsManager, cleanupTransformations);
    setLayerTransformationsManager(layerTransformationsManager, standaloneCleanupTransformations);
}

void LowPrecisionTransformations::setParamsManager(
    IParamsManager* paramsManager,
    std::map<std::string, LayerTransformationPtr>& transformations) noexcept {
    for (auto it : transformations) {
        it.second->setParamsManager(paramsManager);
    }
}

void LowPrecisionTransformations::setParamsManager(
    IParamsManager* paramsManager,
    std::map<std::string, std::vector<std::pair<std::string, LayerTransformationPtr>>>& transformations) noexcept {
    for (auto it : transformations) {
        for (auto transform : it.second) {
            transform.second->setParamsManager(paramsManager);
        }
    }
}

void LowPrecisionTransformations::setParamsManager(
    IParamsManager* paramsManager,
    std::vector<StandaloneCleanup>& transformations) noexcept {
    for (auto it : transformations) {
        it.transformation->setParamsManager(paramsManager);
    }
}

void LowPrecisionTransformations::setLayerTransformationsManager(
    ILayerTransformationsManager* layerTransformationsManager,
    std::map<std::string, LayerTransformationPtr>& transformations) noexcept {
    for (auto it : transformations) {
        it.second->setLayerTransformationsManager(layerTransformationsManager);
    }
}

void LowPrecisionTransformations::setLayerTransformationsManager(
    ILayerTransformationsManager* layerTransformationsManager,
    std::map < std::string, std::vector < std::pair<std::string,  LayerTransformationPtr >> > & transformations) noexcept {
    for (auto it : transformations) {
        for (auto transform : it.second) {
            transform.second->setLayerTransformationsManager(layerTransformationsManager);
        }
    }
}

void LowPrecisionTransformations::setLayerTransformationsManager(
    ILayerTransformationsManager* layerTransformationsManager,
    std::vector<StandaloneCleanup>& transformations) noexcept {
    for (auto it : transformations) {
        it.transformation->setLayerTransformationsManager(layerTransformationsManager);
    }
}

LowPrecisionTransformations LowPrecisionTransformer::getAllTransformations(const LayerTransformation::Params& params) {
    using namespace pass::low_precision;

    auto transformer = LowPrecisionTransformations().
        addBranchSpecific<pass::low_precision::ConcatMultiChannelsTransformation, opset1::Concat>(params).

        add<AddTransformation, opset1::Add>(params).
        add<AvgPoolTransformation, opset1::AvgPool>(params).
        add<ClampTransformation, opset1::Clamp>(params).
        add<ConvolutionTransformation, opset1::Convolution>(params).
        add<DepthToSpaceTransformation, opset1::DepthToSpace>(params).
        add<FakeQuantizeTransformation, opset1::FakeQuantize>(params).
        add<GroupConvolutionTransformation, opset1::GroupConvolution>(params).
        add<InterpolateTransformation, opset1::Interpolate>(params).
        add<MatMulTransformation, opset1::MatMul>(params).
        add<MaxPoolTransformation, opset1::MaxPool>(params).
        add<MultiplyTransformation, opset1::Multiply>(params).
        add<MVNTransformation, op::MVN>(params).
        add<NormalizeL2Transformation, opset1::NormalizeL2>(params).
        add<PReluTransformation, opset1::PRelu>(params).
        add<ReluTransformation, opset1::Relu>(params).
        add<ReshapeTransformation, opset1::Reshape>(params).
        add<SqueezeTransformation, opset1::Squeeze>(params).
        add<TransposeTransformation, opset1::Transpose>(params).
        add<UnsqueezeTransformation, opset1::Unsqueeze>(params).

        addCleanup<FuseConvertTransformation, opset1::Multiply>(params).
        addCleanup<FuseFakeQuantizeTransformation, opset1::FakeQuantize>(params).
        // addCleanup<FuseMultiplyToFakeQuantizeTransformation, opset1::Multiply>(params).
        // addCleanup<FuseSubtractToFakeQuantizeTransformation, opset1::Subtract>(params).
        // addCleanup<ConvertTransformation, opset1::Convert>(params);

        // addStandaloneCleanup<FuseConvertTransformation>(params).
        // addStandaloneCleanup<FuseFakeQuantizeTransformation>(params).

        addStandaloneCleanup<FuseSubtractToFakeQuantizeTransformation, opset1::Subtract>(params).
        addStandaloneCleanup<FuseMultiplyToFakeQuantizeTransformation, opset1::Multiply>(params).
        addStandaloneCleanup<MultiplyToGroupConvolutionTransformation, opset1::Multiply>(params).
        addStandaloneCleanup<SubtractMultiplyToMultiplyAddTransformation, opset1::Multiply>(params);

    return transformer;
}

LowPrecisionTransformer::LowPrecisionTransformer(): transformations(LowPrecisionTransformer::getAllTransformations()) {}

template <typename BaseOp>
void make_matcher_type_relaxed(ngraph::pass::GraphRewrite* transformation) {
    using namespace ngraph;

    auto is_op_type = [](std::shared_ptr<Node> n) {
        return !!as_type_ptr<BaseOp>(n);
    };

    auto p_node = std::make_shared<pattern::op::Label>(element::f32, Shape{}, is_op_type);

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher &m) {
        auto l_node = std::dynamic_pointer_cast<BaseOp>(m.get_match_root());
        if (!l_node) {
            std::cerr << "Error my matcher 1!!!\n";
            return false;
        }
        // std::cerr << "My matcher pass was triggered: " << l_node->get_friendly_name() << " with " << l_node->get_inputs().size() << " inputs\n";
        // TODO: replaces only operation with one output port
        auto replacement = std::make_shared<ngraph::op::TypeRelaxed<BaseOp>>(*l_node, l_node->get_output_element_type(0));
        copy_runtime_info(l_node, replacement);
        replace_node(l_node, replacement);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(p_node, "TypeRelaxedReplacer");
    transformation->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}

TypeRelaxedReplacer::TypeRelaxedReplacer() {
    make_matcher_type_relaxed<opset1::Add>(this);
    make_matcher_type_relaxed<opset1::AvgPool>(this);
    make_matcher_type_relaxed<opset1::Clamp>(this);
    make_matcher_type_relaxed<opset1::Concat>(this);
    make_matcher_type_relaxed<opset1::Convolution>(this);
    make_matcher_type_relaxed<opset1::DepthToSpace>(this);
    make_matcher_type_relaxed<opset1::FakeQuantize>(this);
    make_matcher_type_relaxed<opset1::GroupConvolution>(this);
    make_matcher_type_relaxed<opset1::PRelu>(this);
    make_matcher_type_relaxed<opset1::Subtract>(this);
    make_matcher_type_relaxed<opset1::Interpolate>(this);
    make_matcher_type_relaxed<opset1::Multiply>(this);
    make_matcher_type_relaxed<op::MVN>(this);
    make_matcher_type_relaxed<opset1::NormalizeL2>(this);
}

LowPrecisionTransformer::LowPrecisionTransformer(const LowPrecisionTransformations& transformations)
    : transformations(transformations) {}

void LowPrecisionTransformer::transform(std::shared_ptr<Function> network) {
    transformations.setParamsManager(this);
    transformations.setLayerTransformationsManager(this);

    TransformationContext context(network);

    // Extend necessary operations with polymorphic semantics
    {
        TypeRelaxedReplacer pass;
        pass.run_on_function(network);
    }

    {
        // Branch specific transformations
        GraphRewrite pass;
        registerAllMatchers(transformations.branchSpecificTransformations, pass, context);
        pass.run_on_function(network);
    }

    {
        // Step #1: FakeQuantize layer transformation execution
        LayerTransformationPtr fqTransformation = transformations.find<opset1::FakeQuantize>()[0];
        if (fqTransformation == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "FakeQuantize transformation was not found";
        }
        GraphRewrite pass;
        fqTransformation->registerMatcherIn(pass, context);
        pass.run_on_function(network);
    }

    {
        // Step #2: layer transformations execution
        GraphRewrite pass;
        registerAllMatchers(transformations.transformations, pass, context);
        pass.run_on_function(network);
    }

    {
        // Step #3: cleanup transformations execution
        GraphRewrite pass;
        registerAllMatchers(transformations.cleanupTransformations, pass, context);
        pass.run_on_function(network);
    }

    {
        // Step #4: standalone cleanup transformations execution

        for (auto it : transformations.standaloneCleanupTransformations) {
            GraphRewrite pass;
            it.transformation->registerMatcherIn(pass, context);
            pass.run_on_function(network);
        }
    }
}

std::vector<element::Type> LowPrecisionTransformer::precisionIntersection(
    const std::vector<element::Type>& v1,
    const std::vector<element::Type>& v2) const noexcept {
    std::vector<element::Type> v3;

    auto v1Copy = v1;
    auto v2Copy = v2;

    std::sort(v1Copy.begin(), v1Copy.end());
    std::sort(v2Copy.begin(), v2Copy.end());

    std::set_intersection(v1Copy.begin(), v1Copy.end(),
        v2Copy.begin(), v2Copy.end(),
        std::back_inserter(v3));
    return v3;
}

std::vector<element::Type> LowPrecisionTransformer::getPrecisionsOnActivations(const Node& op) const noexcept {
    const std::string operantionType = LowPrecisionTransformations::getType(op);
    const std::vector<LayerTransformationPtr> transformation = transformations.find(operantionType);
    if (transformation.empty()) {
        return std::vector<element::Type>();
    }
    std::vector<element::Type> precisions = transformation[0]->getPrecisionsOnActivations();

    for (const auto& transform : transformation) {
        precisions = precisionIntersection(precisions, transform->getPrecisionsOnActivations());
    }
    return precisions;
}

bool LowPrecisionTransformer::isQuantized(const std::shared_ptr<Node>& layer) const noexcept {
    const std::string operantionType = LowPrecisionTransformations::getType(*layer);
    const std::vector<LayerTransformationPtr> transformation = transformations.find(operantionType);
    if (transformation.empty()) {
        return false;
    }

    for (const auto& transform : transformation) {
        if (transform->isQuantized(layer)) {
            return true;
        }
    }
    return false;
}

bool LowPrecisionTransformer::isPrecisionPreserved(const std::shared_ptr<Node>& layer) const noexcept {
    const std::string operantionType = LowPrecisionTransformations::getType(*layer);
    const std::vector<LayerTransformationPtr> transformation = transformations.find(operantionType);
    if (transformation.empty()) {
        return false;
    }

    for (const auto& transform : transformation) {
        if (!transform->isPrecisionPreserved(layer)) {
            return false;
        }
    }
    return true;
}

void LowPrecisionTransformer::registerAllMatchers(
    std::map<std::string, LayerTransformationPtr> transformations,
    GraphRewrite& pass,
    TransformationContext& context) {
    for (auto it : transformations) {
        it.second->registerMatcherIn(pass, context);
    }
}

void LowPrecisionTransformer::registerAllMatchers(
    std::map<std::string, std::vector<std::pair<std::string, LayerTransformationPtr>>> transformations,
    GraphRewrite& pass,
    TransformationContext& context) {
    for (auto it : transformations) {
        for (auto transform : it.second) {
            transform.second->registerMatcherIn(pass, context);
        }
    }
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph