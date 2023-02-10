// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pass.hpp>
#include "structural_type_prop.hpp"


namespace ov {
namespace pass {


bool decompose_list_set_item (
    std::shared_ptr<Node> set_item,
    std::shared_ptr<Node> list,
    Output<Node> index_scalar,
    Output<Node> item
);

bool decompose_list_get_item (
    std::shared_ptr<Node> get_item,
    std::shared_ptr<Node> list,
    Output<Node> index
);

bool decompose_list_construct (
    std::shared_ptr<Node> list_construct,
    OutputVector inputs
);


}  // namespace pass
}  // namespace ov
