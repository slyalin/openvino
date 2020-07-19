// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/concat_multi_channels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include "transformations/low_precision/common/fake_quantize_dequantization.hpp"
#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/common/subgraph.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

bool isMultiChannel(const std::vector<ngraph::opset1::Concat*>& concatLayers) {
    for (ngraph::opset1::Concat* concat : concatLayers) {
        std::shared_ptr<ngraph::Node> concatPtr = concat->shared_from_this();
        const std::vector<std::shared_ptr<ngraph::Node>> children = NetworkHelper::getChildrenRecursivelyExceptTypes(
            concatPtr,
            { "Pooling", "Resample" });

        for (const std::shared_ptr<ngraph::Node>& child : children) {
            if (is_type<ngraph::opset1::Convolution>(child.get())) {
                return false;
            }
        }
    }
    return true;
}

void ConcatMultiChannelsTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::Concat>(pass, context);
}

void ConcatMultiChannelsTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<ngraph::opset1::Concat> concat = ngraph::as_type_ptr<ngraph::opset1::Concat>(m.get_match_root());

    ngraph::pass::low_precision::Subgraph subgraph;
    std::unordered_set<std::string> handledLayers;
    if (!subgraph.fillSubgraphForConcat(*concat, handledLayers)) {
        return;
    }

    if (subgraph.quantizationLayers.empty() || isHandled(subgraph.quantizationLayers)) {
        return;
    }

    if (!isMultiChannel(subgraph.concatLayers)) {
        ConcatTransformation::transform(context, m);
        return;
    }

    // TODO: update later
    // TODO: check if precisions are different and return
    ngraph::Node& quantizationLayer = *subgraph.quantizationLayers[0];
    std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(quantizationLayer.shared_from_this());
    DataPrecision dataPrecision = getDataPrecision(fq, QuantizationDetails::getDetails(fq), false, false);
    if (dataPrecision.precision == ngraph::element::undefined) {
        return;
    }

    // TODO: use raw pointer instead names
    std::unordered_map<std::string, ngraph::pass::low_precision::FakeQuantizeDequantization> dequantizations;

    for (size_t i = 0; i < subgraph.quantizationLayers.size(); ++i) {
        ngraph::Node* fakeQuantizeLayer = subgraph.quantizationLayers[i];
        std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(fakeQuantizeLayer->shared_from_this());

        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(fq);

        // TODO: uncomment
        // const size_t channelsCount = CNNNetworkHelper::getOutputChannelsCount(fq);
        // const size_t channelsCount = 3ul;

        // std::vector<float> dequantizationScales(channelsCount);
        // std::vector<float> dequantizationShifts(channelsCount);
        // for (size_t i = 0ul; i < channelsCount; ++i) {
        //    dequantizationScales[i] = QuantizationDetails::isSupportedLevel(quantizationDetails.levels) ?
        //        (quantizationDetails.getOutputHighValue(i) - quantizationDetails.getOutputLowValue(i)) / (dataPrecision.max - dataPrecision.min) :
        //        1.0;

        //    dequantizationShifts[i] = QuantizationDetails::isSupportedLevel(quantizationDetails.levels) ?
        //        (quantizationDetails.getOutputHighValue(i) - (quantizationDetails.getOutputHighValue(i) - quantizationDetails.getOutputLowValue(i)) *
        //        (dataPrecision.max / (dataPrecision.max - dataPrecision.min))) :
        //        0.f;
        // }
        // checkAndUpdateDequantizationShiftWithZero(quantizationDetails, dequantizationShifts);

        // 1. get data for dequantization. Dequantization data will be used several times later.
        FakeQuantizeDequantization fakeQuantizeDequantization = ngraph::pass::low_precision::NetworkHelper::createDequantizationFromFakeQuantize(
            fq, dataPrecision.precision, dataPrecision.min, dataPrecision.max);
        dequantizations[fakeQuantizeLayer->get_friendly_name()] = fakeQuantizeDequantization;

        // 2. update FakeQuantize - one time action
        subgraph.quantizationLayers[i] = ngraph::pass::low_precision::NetworkHelper::updateFakeQuantize(
            fq,
            updatePrecisions ? dataPrecision.precision : fakeQuantizeLayer->get_output_element_type(0),
            roundf(dataPrecision.min),
            roundf(dataPrecision.max)).get();
    }

    //if (updatePrecisions) {
    //    for (const auto it : subgraph.layers) {
    //        const CNNLayer* layer = it.second;
    //        CNNNetworkHelper::setOutDataPrecision(*layer, dataPrecision.precision);
    //    }
    //}

    auto dequantizationValuesCallback = [&](
        ngraph::Node& layer,
        const std::string originalLayerName,
        std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate) {
        if (layer.get_friendly_name() != originalLayerName) {
            const auto update = [](
                const std::string& originalLayerName,
                const std::string& newLayerName,
                std::unordered_map<std::string, FakeQuantizeDequantization>& dequantizationLayers) {
                auto it = dequantizationLayers.find(originalLayerName);
                if (it != dequantizationLayers.end()) {
                    dequantizationLayers.emplace(newLayerName, it->second);
                    dequantizationLayers.erase(it);
                }
            };
            update(originalLayerName, layer.get_friendly_name(), dequantizations);
        }

        fillDequantization(
            layer,
            dequantizations,
            dequantizationsToConcatenate);
    };

    addDequantizationLayers(context, subgraph, dequantizationValuesCallback);

    if (updatePrecisions) {
        for (const auto it : subgraph.layers) {
            ngraph::Node* node = it.second;
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(node->shared_from_this(), dataPrecision.precision);
            // std::cout << "\t" << node->get_friendly_name() << ": " << dataPrecision.precision << std::endl;
        }
    }

    // std::cout << "ConcatMultiChannelsTransformation::transform: done: " << concat->get_friendly_name() << std::endl;
}

bool ConcatMultiChannelsTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return true;
}

// fill dequantizationsToMerge collection for layer with using dequantizationByFakeQuantize
void ConcatMultiChannelsTransformation::fillDequantization(
    ngraph::Node& layer,
    const std::unordered_map<std::string, FakeQuantizeDequantization>& dequantizationByFakeQuantize,
    std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate) {
    std::vector<ngraph::opset1::FakeQuantize*> fakeQuantizes;
    ngraph::opset1::FakeQuantize* currentFakeQuantize = ngraph::as_type<ngraph::opset1::FakeQuantize>(&layer);
    if (currentFakeQuantize != nullptr) {
        fakeQuantizes.push_back(currentFakeQuantize);
    } else {
        fillQuantization(layer, fakeQuantizes);
    }

    for (const ngraph::opset1::FakeQuantize* fakeQuantize : fakeQuantizes) {
        const auto it = dequantizationByFakeQuantize.find(fakeQuantize->get_friendly_name());
        if (it == dequantizationByFakeQuantize.end()) {
            THROW_IE_LPT_EXCEPTION(*fakeQuantize) << "dequantization scale values are not found";
        }
        const FakeQuantizeDequantization& fakeQuantizeDequantization = it->second;
        dequantizationsToConcatenate.push_back(fakeQuantizeDequantization);
    }
}

void ConcatMultiChannelsTransformation::fillQuantization(const ngraph::Node& layer, std::vector<ngraph::opset1::FakeQuantize*>& fakeQuantizes) {
    for (int i = 0; i < layer.get_input_size(); ++i) {
        ngraph::Node* parent = layer.get_input_node_ptr(i);
        ngraph::opset1::FakeQuantize* fakeQuantize = ngraph::as_type<ngraph::opset1::FakeQuantize>(parent);
        if (fakeQuantize != nullptr) {
            fakeQuantizes.push_back(fakeQuantize);
        } else {
            fillQuantization(*parent, fakeQuantizes);
        }
    }
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
