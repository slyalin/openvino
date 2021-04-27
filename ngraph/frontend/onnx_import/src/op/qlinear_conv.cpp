// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include <cstddef>
#include <memory>
#include <vector>

#include "exceptions.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "op/qlinear_conv.hpp"
#include "core/null_node.hpp"
#include "conv.hpp"
#include "dequantize_linear.hpp"
#include "quantize_linear.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector qlinear_conv(const Node& node)
                {
                    const OutputVector& inputs = node.get_ng_inputs();

                    auto x = inputs[0];
                    auto x_scale = inputs[1];
                    auto x_zero_point = inputs[2];
                    auto w = inputs[3];
                    auto w_scale = inputs[4];
                    auto w_zero_point = inputs[5];
                    auto y_scale = inputs[6];
                    auto y_zero_point = inputs[7];
                    Output<ngraph::Node> B = inputs.size() > 8 ? inputs[8] : std::make_shared<NullNode>()->output(0);

                    x = set_13::detail::dequantize_linear(x, x_scale, std::make_shared<opset6::Convert>(x_zero_point, element::f32), 1, node)[0];
                    w = set_13::detail::dequantize_linear(w, w_scale, std::make_shared<opset6::Convert>(w_zero_point, element::f32), 1, node)[0];

                    if(!ngraph::op::is_null(B))
                    {
                        B = std::make_shared<opset6::Multiply>(
                                std::make_shared<opset6::Convert>(B, x_scale.get_element_type()),
                                std::make_shared<opset6::Multiply>(x_scale, w_scale))->output(0);
                    }

                    auto result = detail::conv(node, x, w, B)[0];

                    // TODO: Quantize result instead of just Convert -- the code IS NOT CORRECT, IT IS A STUB
                    //result = std::make_shared<opset6::Convert>(result, inputs[0].get_element_type());
                    result = set_13::detail::quantize_linear(result, y_scale, y_zero_point, 1, node)[0];

                    return {result};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
