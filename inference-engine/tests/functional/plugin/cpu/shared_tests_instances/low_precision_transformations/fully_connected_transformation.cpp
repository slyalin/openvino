// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fully_connected_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
        ngraph::element::f32,
        // ngraph::element::f16
};

const std::vector<MatMulShapes> shapes = {
    {
        ngraph::Shape{ 1, 16 },
        ngraph::Shape{ 16, 8 },
        false,
        false
    },
    {
        ngraph::Shape{ 1, 16 },
        ngraph::Shape{ 8, 16 },
        false,
        true
    },
    {
        ngraph::Shape{ 16, 1 },
        ngraph::Shape{ 16, 8 },
        true,
        false
    },
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams()
};

const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versions = {
    LayerTestsUtils::LayerTransformation::LptVersion::cnnNetwork,
    LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

INSTANTIATE_TEST_CASE_P(LPT, FullyConnectedTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(versions)),
    FullyConnectedTransformation::getTestCaseName);
}  // namespace




