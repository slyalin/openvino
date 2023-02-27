// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/str_ops.hpp"
#include "utils.hpp"
#include "openvino/core/validation_util.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_col2im(NodeContext& context) {
    std::cerr << "col2im\n";
    std::cerr << context.get_input(1).get_node_shared_ptr() << '\n';
    std::cerr << context.get_input(2).get_node_shared_ptr() << '\n';
    std::cerr << context.get_input(3).get_node_shared_ptr() << '\n';
    //num_inputs_check(context, 5, 5);
    auto input = context.get_input(0);

    Output<Node> output_size = try_list_of_scalars_concat(context.get_input(1));

    // std::cerr << context.get_input(1).get_node_shared_ptr() << '\n';
    // std::cerr << context.get_input(2).get_node_shared_ptr() << '\n';
    // std::cerr << context.get_input(3).get_node_shared_ptr() << '\n';
    // if(auto sp = std::dynamic_pointer_cast<tensorflow::StructPack>(context.get_input(1).get_node_shared_ptr())) {
    //     // Suppose that this is a list of scalar values
    //     // TODO: Check that they are scalars
    //     auto elements_input = sp->input_value(3);   // SP for lists has 4 inputs: shapes, begins, ends, elements, we are taking elements
    //     std::cerr << "Detected StructPack as 1st input of col2im: " << sp << '\n';
    //     std::cerr << "Elements of that StructPack: " << sp << '\n';
    //     std::cerr << "Rank of stored elements in the list: " << sp->input_value(0).get_partial_shape()[1] << "\n";
    //     auto elements_const = get_constant_from_source(elements_input);
    //     if(elements_const) {
    //         std::cerr << "    elements is represented as constant " << elements_const << "\n";
    //         output_size = elements_const;
    //     } else {
    //         std::cerr << "    elements cannot be represented as constant, leave a tensor elements instead of SP\n";
    //         output_size = elements_input;
    //     }
    // } else {
    //     std::cerr << "    type of input is not recognized as a StructPack, leave as is\n";
    //     output_size = context.get_input(1);
    // }

    OutputVector inputs = context.inputs();
    // We can use elements directly without any reshapes, because it is expected to be 1d tensor
    inputs[1] = output_size;

    // TODO: Implement real conversino code. For now it is just creation of FW node as a stub

    auto node = std::make_shared<PtFrameworkNode>(context.get_decoder(), inputs, 1);
    /*
    node->set_output_type(0,
        context.get_input(0).get_element_type(),
        // FIXME: Set static output rank based on one of the input partially static shape without extra check may cause an exception
        PartialShape(std::vector<Dimension>(output_size.get_partial_shape()[0].get_length() + 2)));
        */

    std::cerr << "Finally produced node for col2im: " << node << "\n";

    return node->outputs();
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov