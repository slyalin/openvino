// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

class FWNodeMatcherPass : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pytorch::pass::FWNodeMatcherPass");
    FWNodeMatcherPass(const std::string& fw_type);
    virtual bool replacer (std::shared_ptr<PtFrameworkNode> fw_node) const = 0;
};

class GenericListConstruct : public FWNodeMatcherPass {
public:
    GenericListConstruct() : FWNodeMatcherPass("prim::ListConstruct") {}
    bool replacer (std::shared_ptr<PtFrameworkNode> fw_node) const override;
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
