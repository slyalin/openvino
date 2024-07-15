// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <openvino/op/matmul.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "matmul.hpp"
#include "../convert_common.hpp"


namespace {

using namespace ov::mlir;

struct ConvertMatMul {
    void operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        // TODO: Support broadcasts
        const auto inputs = context.getInputs(node);
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type); // Instead of this (WRONG): cast<mlir::ShapedType>(inputs[0].getType());
        // Named binary ops directly overwrite data in `outs` buffer so, there is no need to provide non-empty
        // destination at the tensor-level.
        // Use `tensor.empty` to avoid temporary buffer allocation and memcpy after bufferization.
        llvm::SmallVector<Value> dynamicSizes;
        for (auto [idx, dim] : llvm::enumerate(outType.getShape())) {
            if (!mlir::ShapedType::isDynamic(dim))
                continue;
            auto dimSize = builder.create<tensor::DimOp>(loc, idx == 0 ? inputs[0] : inputs[1], idx);  // TODO: Use symbols instead of taking dims directly from inputs
            dynamicSizes.push_back(dimSize);
        }
        auto empty = builder.create<tensor::EmptyOp>(loc, outType, dynamicSizes);
        auto matmul_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        assert(matmul_node);

        // FIXME: current code limitation
        assert(!matmul_node->get_transpose_a() && matmul_node->get_transpose_b());

        mlir::Operation* op = nullptr;
        // TODO: Add other variants of transpose_a/transpose_b
        op = builder.create<linalg::MatmulTransposeBOp>(loc, mlir::ValueRange{inputs[0], inputs[1]}, mlir::ValueRange{empty});
        context.addOutputs(node, op);
    }
};

}

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;
using namespace ov::op;

MatMulPattern::MatMulPattern() : MarkPattern(
    wrap_type<v0::MatMul>({any_input(), any_input()}),
    ConvertMatMul()) {
    }


} // namespace mlir
} // namespace ov