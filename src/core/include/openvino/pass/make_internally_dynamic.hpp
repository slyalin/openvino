// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
/**
 * @brief The transformation finds operations that produces partially filled tensors and post-process output
 * to keep only a filled part. It introduce dynamic shapes flow because the shape of filled part is not known
 * statically in advance. Relevant operations: DetectionOutput, Proposal.
 */
class OPENVINO_API MakeInternallyDynamic : public ModelPass {
public:
    OPENVINO_RTTI("MakeInternallyDynamic");

    explicit MakeInternallyDynamic()  {}

    bool run_on_function(std::shared_ptr<ov::Model> f) override;

};
}  // namespace pass
}  // namespace ov
