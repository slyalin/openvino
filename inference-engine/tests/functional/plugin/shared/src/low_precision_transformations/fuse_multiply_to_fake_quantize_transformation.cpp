// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_multiply_to_fake_quantize_transformation.hpp"

#include <tuple>
#include <sstream>
#include <string>
#include <vector>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/fuse_multiply_to_fake_quantize_function.hpp"

namespace LayerTestsDefinitions {

std::string FuseMultiplyToFakeQuantizeTransformation::getTestCaseName(testing::TestParamInfo<FuseMultiplyToFakeQuantizeTransformationParams> obj) {
    std::string targetDevice;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    FuseMultiplyToFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, version, testValues) = obj.param;

    std::ostringstream result;
    result << targetDevice << "_" <<
        version << "_" <<
        testValues.actual.dequantization << "_" <<
        testValues.actual.fakeQuantizeOnData;
    return result.str();
}

void FuseMultiplyToFakeQuantizeTransformation::SetUp() {
    LayerTestsUtils::LayerTransformation::LptVersion version;
    FuseMultiplyToFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, version, testValues) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::FuseMultiplyToFakeQuantizeFunction::get(
        testValues.inputShape,
        testValues.actual.fakeQuantizeOnData,
        testValues.actual.dequantization);

    ngraph::pass::InitNodeInfo().run_on_function(function);
}

TEST_P(FuseMultiplyToFakeQuantizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions