// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
    std::shared_ptr<Node> get_im2col_indices_along_dim(NodeContext& context,
                                                    std::shared_ptr<Node> input_d,
                                                    int64_t kernel_size_d,
                                                    int64_t dilation_d,
                                                    int64_t padding_d,
                                                    int64_t stride_d) {
        auto zero = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
        auto minus_one = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {-1}));
        auto kernel_size = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {kernel_size_d}));
        auto padding_2 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {padding_d * 2}));
        auto stride = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {stride_d}));
        auto blocks_d = context.mark_node(std::make_shared<opset8::Add>(input_d, padding_2));
        auto subtrahend =
            context.mark_node(opset8::Constant::create(element::i64, Shape{}, {dilation_d * (kernel_size_d - 1)}));
        blocks_d = context.mark_node(std::make_shared<opset8::Subtract>(blocks_d, subtrahend));
        auto blocks_d_indices = context.mark_node(std::make_shared<opset8::Range>(zero, blocks_d, stride, element::i64));
        blocks_d_indices = context.mark_node(std::make_shared<opset8::Unsqueeze>(blocks_d_indices, zero));
        std::vector<int64_t> rng;
        for (int64_t i = 0; i < kernel_size_d * dilation_d; i += dilation_d) {
            rng.push_back(i);
        }

        auto kernel_grid = context.mark_node(opset8::Constant::create(element::i64, Shape{rng.size()}, rng));
        auto kernel_mask = context.mark_node(std::make_shared<opset8::Unsqueeze>(kernel_grid, minus_one));
        return context.mark_node(std::make_shared<opset8::Add>(blocks_d_indices, kernel_mask));
    }
namespace op {

OutputVector translate_im2col(NodeContext& context) {
    auto input = context.get_input(0);
    auto kernel_size = context.const_input<std::vector<int64_t>>(1);
    FRONT_END_OP_CONVERSION_CHECK(kernel_size.size() == 2, "kernel size should contains 2 elements");
    auto dilation = context.const_input<std::vector<int64_t>>(2);
    FRONT_END_OP_CONVERSION_CHECK(kernel_size.size() == 2, "dilation should contains 2 elements");
    auto padding = context.const_input<std::vector<int64_t>>(3);
    FRONT_END_OP_CONVERSION_CHECK(kernel_size.size() == 2, "padding should contains 2 elements");
    auto stride = context.const_input<std::vector<int64_t>>(4);
    FRONT_END_OP_CONVERSION_CHECK(kernel_size.size() == 2, "stride should contains 2 elements");
    auto zero = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
    auto input_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
    auto zero_f = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {0}));
    auto minus_one = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {-1}));
    auto two = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {2}));
    auto four = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {4}));
    auto b_dim = zero;
    auto c_dim = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {1}));
    auto h_dim = two;
    auto w_dim = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {3}));
    auto input_c = context.mark_node(std::make_shared<opset8::Gather>(input_shape, c_dim, zero));
    auto input_b = context.mark_node(std::make_shared<opset8::Gather>(input_shape, b_dim, zero));
    auto input_h = context.mark_node(std::make_shared<opset8::Gather>(input_shape, h_dim, zero));
    auto input_w = context.mark_node(std::make_shared<opset8::Gather>(input_shape, w_dim, zero));
    auto stride_h = stride[0];
    auto stride_w = stride[1];
    auto padding_h = padding[0];
    auto padding_w = padding[1];
    auto dilation_h = dilation[0];
    auto dilation_w = dilation[1];
    auto kernel_h = kernel_size[0];
    auto kernel_w = kernel_size[1];
    auto blocks_row_indices = get_im2col_indices_along_dim(context, input_h, kernel_h, dilation_h, padding_h, stride_h);
    auto blocks_col_indices = get_im2col_indices_along_dim(context, input_w, kernel_w, dilation_w, padding_w, stride_w);
    auto kernel_window = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {kernel_h * kernel_w}));
    auto channel_unfolded = context.mark_node(std::make_shared<opset8::Multiply>(input_c, kernel_window));
    auto input_b_unsqueezed = context.mark_node(std::make_shared<opset8::Unsqueeze>(input_b, zero));
    auto channel_unfolded_unsqueezed = context.mark_node(std::make_shared<opset8::Unsqueeze>(channel_unfolded, zero));
    auto output_shape = context.mark_node(
        std::make_shared<opset8::Concat>(NodeVector{input_b_unsqueezed, channel_unfolded_unsqueezed, minus_one}, 0));
    auto pads = context.mark_node(
        opset8::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 0, padding_h, padding_w}));
    auto padded_input =
        context.mark_node(std::make_shared<opset8::Pad>(input, pads, pads, zero_f, ov::op::PadMode::CONSTANT));
    auto output = context.mark_node(std::make_shared<opset8::Gather>(padded_input, blocks_row_indices, two));
    output = context.mark_node(std::make_shared<opset8::Gather>(output, blocks_col_indices, four));
    auto permutation_dims =
        context.mark_node(opset8::Constant::create(element::i64, Shape{6}, std::vector<int64_t>{0, 1, 2, 4, 3, 5}));
    output = context.mark_node(std::make_shared<opset8::Transpose>(output, permutation_dims));
    return {context.mark_node(std::make_shared<opset8::Reshape>(output, output_shape, false))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov