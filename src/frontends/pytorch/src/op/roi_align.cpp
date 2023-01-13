// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset9.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_roi_align(NodeContext& context) {
    auto input_tensor = context.get_input(0);
    auto rois = context.get_input(1);

    // std::cout << context.get_input(1).get_node_shared_ptr()->get_friendly_name() << std::endl;

    // // auto if_node = std::make_shared<opset8::If>(context.get_input(1));
    // context.mark_node(if_node);
    // std::cout << "HEEERE" << std::endl;
    // std::cout << if_node->get_internal_subgraphs_size() << std::endl;
    // auto else_body = if_node->get_else_body();
    // auto then_body = if_node->get_then_body();
    // auto else_size = else_body->get_graph_size();
    // auto then_size = then_body->get_graph_size();

    // std::cout << "HEEERE" << std::endl;
    // std::cout << else_size << std::endl;
    // std::cout << then_size << std::endl;

    return {rois};
    // auto decoder = context.get_decoder();
    // // OV_FRONTEND_REQUIRE(decoder->get_subgraph_size() == 2);

    // std::cout << decoder->get_subgraph_size() << std::endl;

    // auto then_decoder = decoder->get_subgraph_decoder(0);
    // auto then_body = context.convert_subgraph(0);
    // if_node->set_then_body(then_body);
    // auto then_inputs = then_decoder->inputs();

    // auto else_decoder = decoder->get_subgraph_decoder(1);
    // auto else_body = context.convert_subgraph(1);
    // if_node->set_else_body(else_body);
    // auto else_inputs = else_decoder->inputs();

    // std::cout << "HELLOoooooooooooooooooooooooo" << std::endl;

    // std::set<size_t> input_idxs;
    // input_idxs.insert(then_inputs.begin(), then_inputs.end());
    // input_idxs.insert(else_inputs.begin(), else_inputs.end());


    // std::cout << output->get_friendly_name() << std::endl;

    // return {output};


    
    // auto boxes_tensor = std::make_shared<opset8::Convert>(boxes, element::f32);
    // auto zero_const = opset8::Constant::create(element::i32, Shape{1}, {0});

    // OutputVector res;

    // auto node = rois.get_node_shared_ptr();
    // std::cout << node->get_friendly_name() << std::endl;
    // // auto const_0 = context.mark_node(opset8::Constant::create(element::i32, Shape{1}, {0}));
    // // auto result = std::make_shared<opset8::Concat>(node, const_0);
    // return {input_tensor};



    // auto const_1 = context.mark_node(opset8::Constant::create(element::i32, Shape{1}, {1}));
    // auto axes = context.mark_node(opset8::Constant::create(element::i32, Shape{2}, {0, 1}));
    // auto start = context.mark_node(opset8::Constant::create(element::i32, Shape{2}, {0, 0}));
    // auto stop = context.mark_node(opset8::Constant::create(element::i32, Shape{2}, {2, 4}));
    // auto step = context.mark_node(opset8::Constant::create(element::i32, Shape{2}, {1, 1}));
    // // auto const_minus_1 = context.mark_node(opset8::Constant::create(element::i32, Shape{1}, {-1}));

    // // auto less = context.mark_node(std::make_shared<opset8::Less>(start, const_0));
    // // auto const_1_signed = context.mark_node(std::make_shared<opset8::Select>(less, const_minus_1, const_1));
    // // auto stop = context.mark_node(std::make_shared<opset8::Add>(start, const_1_signed));

    // // auto slice_node =
    // //     context.mark_node(std::make_shared<opset8::Slice>(boxes_tensor, start, stop, step, axes));
    // // auto batch_indices = context.mark_node(opset8::Constant::create(element::i32, Shape{2}, {0, 5}));
    // // const auto roi_align = std::make_shared<opset9::ROIAlign>(input_tensor,
    // //                                                           boxes_tensor,
    // //                                                           batch_indices,
    // //                                                           7,
    // //                                                           7,
    // //                                                           2.0,
    // //                                                           2,
    // //                                                           ov::op::v9::ROIAlign::PoolingMode::MAX,
    // //                                                           ov::op::v9::ROIAlign::AlignedMode::ASYMMETRIC);

    // return {result};

    // return std::make_shared<opset9::ROIAlign>(input_tensor, boxes_tensor, batch_indices, 7, 7, 2, 2.0, )
    

    // auto batch_indices = context.mark_node(std::make_shared<opset8::Squeeze>(slice_node, dim));

    // auto squeeze = context.mark_node(std::make_shared<opset8::Squeeze>(batch_indices, const_1));

    // return {slice_node};





    // const auto output_size = context.get_input(2);
    // const auto spatial_scale = context.get_input(3);
    // const auto sampling_ratio = context.get_input(4);
    // const auto aligned = context.get_input(5);
    // const auto batch_size = input.get_partial_shape()[0];
    // const auto batch_indices = opset8::Constant::create(element::i32, Shape{1}, {2});
    // std::cout << batch_indices << std::endl;
    // const auto roi_align = std::make_shared<opset9::ROIAlign>(input_tensor,
    //                                                           boxes_tensor,
    //                                                           batch_indices,
    //                                                           7,
    //                                                           7,
    //                                                           2.0,
    //                                                           2,
    //                                                           ov::op::v9::ROIAlign::PoolingMode::MAX,
    //                                                           ov::op::v9::ROIAlign::AlignedMode::ASYMMETRIC);

    // return {roi_align};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov