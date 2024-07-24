// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <openvino/op/relu.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "binary_eltwise.hpp"

namespace {

using namespace ov;
using namespace ov::mlir;
using ::mlir::ValueRange;

class ConvertBinaryEltwise {

    BinaryEltwisePatternBase::Builder m_op_builder;

public:

    ConvertBinaryEltwise(BinaryEltwisePatternBase::Builder op_builder) : m_op_builder(op_builder) {}

    void operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto inputs = context.getInputs(node);
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        const int output_rank = ov_output_shape.rank().get_length();

        SmallVector<Value> dynamicSizes;
        for (auto [idx, dim] : llvm::enumerate(ov_output_shape)) {
            if (!dim.is_dynamic())
                continue;
            dynamicSizes.push_back(context.get_dimension_value(dim));
        }

        SmallVector<Value> broadcasted_inputs;
        for(size_t i = 0; i < inputs.size(); ++i) {
            auto dimensions = broadcast_dimensions(node->get_input_partial_shape(i), ov_output_shape);
            if(!dimensions.empty()) {
                // FIXME: Find a way to avoid dimension squeezing before applying linalg.broadcast

                // Step 1: Squeeze input shape to eliminate broadcasted dimensions
                SmallVector<ReassociationIndices, 6> squeeze_map;
                ReassociationIndices ri_cur;
                size_t output_idx = 0; // index in ov_output_shape
                bool group_open = true;
                for(auto [_, dim]: llvm::enumerate(dimensions)) {
                    for(; output_idx < dim; ++output_idx) {
                        if(!ri_cur.empty() && !group_open) {
                            squeeze_map.emplace_back(ri_cur);
                            ri_cur = ReassociationIndices();
                        }
                        ri_cur.push_back(output_idx);
                        group_open = false;
                    }
                    assert(dim == output_idx);
                    ri_cur.push_back(dim);
                    ++output_idx;
                }
                for(; output_idx < output_rank; ++output_idx) {
                    if(group_open) {
                        ri_cur.push_back(output_idx);
                        squeeze_map.push_back(ri_cur);
                        group_open = false;
                    } else {
                        squeeze_map.push_back({output_idx});
                    }
                }

                auto squeezed = builder.create<tensor::CollapseShapeOp>(loc, inputs[i], squeeze_map);

                // Step 2: Broadcast squeezed shape to the target shape
                auto empty = builder.create<tensor::EmptyOp>(loc, outType, dynamicSizes);
                auto op = builder.create<linalg::BroadcastOp>(loc, squeezed, empty, dimensions);
                broadcasted_inputs.push_back(op.getResult()[0]);
            } else {
                broadcasted_inputs.push_back(inputs[i]);
            }
        }

        auto empty = builder.create<tensor::EmptyOp>(loc, outType, dynamicSizes);
        auto op = m_op_builder(builder, loc, ValueRange(broadcasted_inputs), ValueRange{empty});
        context.addOutputs(node, op);
    }
};

}  // namespace

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;

BinaryEltwisePatternBase::BinaryEltwisePatternBase(NodeTypeInfo wrapped_type, Builder op_builder)
    : MarkPattern(
        std::make_shared<pass::pattern::op::WrapType>(
            wrapped_type,
            [](const Output<Node>& output) {
                auto node = output.get_node_shared_ptr();
                for(const auto& input: node->inputs()) {
                    if(!statically_broadcastable(input.get_partial_shape(), output.get_partial_shape())) {
                        return false;
                    }
                }
                return true;
            },
            OutputVector{any_input(), any_input()}),
        ConvertBinaryEltwise(op_builder))
    {}

}  // namespace mlir
}  // namespace ov
