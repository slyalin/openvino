// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_log2(NodeContext& context) {
    auto x = context.get_input(0);
    auto two = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {2}));
    auto log2 = context.mark_node(std::make_shared<opset8::Log>(two));
    auto log = context.mark_node(std::make_shared<opset8::Log>(x));
    auto log_f = context.mark_node(std::make_shared<opset8::Convert>(log, element::f32));
    auto res = context.mark_node(std::make_shared<opset8::Divide>(log_f, log2));
    if (!x.get_element_type().is_integral()){
        res = context.mark_node(std::make_shared<opset8::ConvertLike>(res, x));

    }
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov