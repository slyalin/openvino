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

#include "op/conv2d.hpp"
#include "op/conv2d.hpp"
#include "op/batch_norm.hpp"
#include "op/relu.hpp"
#include "op/pool2d.hpp"
#include "op/elementwise_ops.hpp"
#include "op/mul.hpp"
#include "op/scale.hpp"
#include "op/leakyrelu.hpp"
#include "op/interp.hpp"
#include "op/concat.hpp"
#include "op/cast.hpp"

#include "op_table.hpp"


namespace ngraph {
namespace frontend {
namespace pdpd {

std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
            {"conv2d", op::conv2d},
            {"batch_norm", op::batch_norm},
            {"relu", op::relu},
            {"pool2d", op::pool2d},
            {"elementwise_add", op::elementwise_add},
            {"mul", op::mul},
            {"scale", op::scale},
            {"leaky_relu", op::leakyrelu},
            {"nearest_interp_v2", op::interp},
            {"concat", op::concat},
            {"elementwise_div", op::elementwise_div},
            {"cast", op::cast}
    };
};

}}}
