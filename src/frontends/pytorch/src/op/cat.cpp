// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/str_ops.hpp"
#include "openvino/opsets/opset10.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_cat(NodeContext& context) {
    using namespace opset10;
    num_inputs_check(context, 2, 2);
    while(true)if(auto sp = std::dynamic_pointer_cast<tensorflow::StructPack>(context.get_input(0).get_node_shared_ptr())) {
        auto axis_node = context.get_input(1).get_node_shared_ptr();
        auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(axis_node);
        if (!axis_const) {
            std::cerr << "cat 1\n";
            std::cerr << axis_const << "\n";
            break;
        }
        auto axis = axis_const->cast_vector<int64_t>();
        if (axis.size() != 1) {
            std::cerr << "cat 2\n";
            std::cerr << axis.size() << "\n";
            break;
        }

        auto dim = axis[0];
        std::cerr << "Decomposing cat with list\n";

        // If dim != 0 this is a complex case that requires gathering all items that were put into the list
        // and concatenating them together along non zero axis.
        // TODO: Works for case with multiple appends only (ListConstruct as a decomposition also works)
        // TODO: Provide more generic code for dim=0 when repacking is not required, and the length of
        // list can be dynamic

        auto elements = sp->input_value(3);
        OutputVector items;

        std::shared_ptr<Node> concat = std::dynamic_pointer_cast<Concat>(elements.get_node_shared_ptr());

        while(concat && concat->get_rt_info().find("StructPackConcat") != concat->get_rt_info().end()) {
            if(concat->get_input_size() == 2) {
                items.push_back(concat->get_input_node_ptr(1)->/*Reshape*/input_value(0));
                std::cerr << "Found item: " << items.back() << "\n";
                concat = concat->get_input_node_shared_ptr(0);
                if(auto convert = std::dynamic_pointer_cast<ConvertLike>(concat)) {
                    concat = std::dynamic_pointer_cast<Concat>(convert->get_input_node_shared_ptr(0));
                } else {
                    concat = std::dynamic_pointer_cast<Concat>(concat);
                }
            }
        }

        std::cerr << "Gathered " << items.size() << " items\n";
        auto final_concat = std::make_shared<Concat>(items, dim);
        std::cerr << "Final concat: " << final_concat << "\n";

        return {context.mark_node(final_concat)};
    }

    std::cerr << "Fallback to FrameworkNode\n";
    return {std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs(), 1)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov