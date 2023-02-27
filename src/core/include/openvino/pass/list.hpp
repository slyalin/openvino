// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pass.hpp>
#include "structural_type_prop.hpp"


namespace ov {
namespace pass {


OPENVINO_API Output<Node> decompose_list_set_item (
    std::shared_ptr<Node> list,
    Output<Node> index_scalar,
    Output<Node> item
);


OPENVINO_API Output<Node> decompose_list_append (
    std::shared_ptr<Node> list,
    Output<Node> item
);


OPENVINO_API Output<Node> decompose_list_get_item (
    std::shared_ptr<Node> list,
    Output<Node> index
);

OPENVINO_API Output<Node> decompose_list_reserve (
    Output<Node> element_shape,
    Output<Node> num_elements_scalar,
    element::Type element_type = element::i32,
    element::Type shape_type = element::i32
);


OPENVINO_API Output<Node> decompose_list_construct (
    OutputVector inputs,
    element::Type element_type = element::i32,
    element::Type shape_type = element::i32
);

OPENVINO_API Output<Node> decompose_tensor_to_list (
    OutputVector tensor,
    element::Type shape_type = element::i32
);

OPENVINO_API Output<Node> decompose_list_stack (
    std::shared_ptr<Node> list,
    const int dim   // dimension along which we concatinating tensors
);


}  // namespace pass
}  // namespace ov
