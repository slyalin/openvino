//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <map>

#include <ngraph/opsets/opset6.hpp>
#include "elementwise_ops.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

template <typename T>
OutputVector elementwise_ops (const NodeContext& node, const T& def = T()) { 
    auto x = node.get_ng_input("X");
    auto y = node.get_ng_input("Y");    

    auto axis = node.get_attribute<int>("axis");

    auto x_shape = x.get_shape();
    auto y_shape = y.get_shape();

    if ((axis == -1) || (axis == x_shape.size() - 1) || (x_shape.size() == y_shape.size())) {
        return {std::make_shared<T>(x, y)};
    }
    else {
        auto broadcast_shape = Shape(x_shape.size(), 1);
        int32_t idx = 0;
        for(auto it = y_shape.begin(); it != y_shape.end(); ++it, idx++)
            broadcast_shape[axis+idx] = *it;

        std::cout << broadcast_shape << std::endl;

        auto broadcast_shape_node = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{broadcast_shape.size()}, broadcast_shape);    
        auto y_node = std::make_shared<ngraph::opset6::Reshape>(y, broadcast_shape_node, false);    
        return {std::make_shared<T>(x, y_node)};             
    }   
}

//
OutputVector elementwise_add (const NodeContext& node_context) {
    return elementwise_ops(node_context, ngraph::opset6::Add());
}

OutputVector elementwise_div (const NodeContext& node_context) {
    return elementwise_ops(node_context, ngraph::opset6::Divide());
}

}}}}