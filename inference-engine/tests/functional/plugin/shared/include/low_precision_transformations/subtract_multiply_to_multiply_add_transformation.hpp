// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {

class SubtractMultiplyToMultiplyAddTransformationTestValues {
public:
    ngraph::Shape inputShape;
    ngraph::element::Type precision;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
};

typedef std::tuple<
    std::string,
    LayerTestsUtils::LayerTransformation::LptVersion,
    SubtractMultiplyToMultiplyAddTransformationTestValues> SubtractMultiplyToMultiplyAddTransformationParams;

class SubtractMultiplyToMultiplyAddTransformation :
    public testing::WithParamInterface<SubtractMultiplyToMultiplyAddTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SubtractMultiplyToMultiplyAddTransformationParams> obj);

protected:
    void SetUp() override;
    void validateNGraph();
    void validateCNNNetwork();
};

}  // namespace LayerTestsDefinitions