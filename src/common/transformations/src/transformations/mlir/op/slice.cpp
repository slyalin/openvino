// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <openvino/op/slice.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "slice.hpp"
#include "../convert_common.hpp"

namespace {

using namespace ov::mlir;

struct ConvertSlice {
    void operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];
        const auto start = context.getInputs(node)[1];
        const auto stop = context.getInputs(node)[2];
        const auto step = context.getInputs(node)[3];
        // const auto axes = context.getInputs(node)[4];

        const auto ov_index_shape = node->get_input_partial_shape(1);
        const auto ov_index_element_type = node->get_input_element_type(1);
        auto dynamic_index_dims = context.get_dynamic_dimension_values(ov_index_shape);

        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);
        auto out_type = importTensor(context.context, ov_output_shape, ov_output_element_type);

        auto const_start = std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
        auto const_stop = std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(2));
        auto const_step = std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(3));
        if (const_start && const_stop && const_step) {
            ov::Coordinate coord_start = const_start->get_coordinate_val();
            ov::Coordinate coord_stop = const_stop->get_coordinate_val();
            ov::Coordinate coord_step = const_step->get_coordinate_val();
            SmallVector<int64_t> static_start(coord_start.begin(), coord_start.end());
            SmallVector<int64_t> static_stop(coord_stop.begin(), coord_stop.end());
            SmallVector<int64_t> static_step(coord_step.begin(), coord_step.end());
            SmallVector<int64_t> static_sizes;
            // std::cerr << "static_start.size() == " << static_start.size() << std::endl;
            // std::cerr << "static_stop.size() == " << static_stop.size() << std::endl;
            // std::cerr << "static_step.size() == " << static_step.size() << std::endl;
            assert(static_start.size() == static_stop.size() && static_stop.size() == static_step.size());
            for (size_t i = 0; i < static_start.size(); ++i) {
                static_sizes.push_back(static_stop[i] - static_start[i]);
                // std::cerr << "static_start[" << i << "] = " << static_start[i] << std::endl;
                // std::cerr << "static_sizes[" << i << "] = " << static_sizes[i] << std::endl;
                // std::cerr << "static_step[" << i << "] = " << static_step[i] << std::endl;
                // std::cerr << "static_stop[" << i << "] = " << static_stop[i] << std::endl;
                assert((static_step[i] > 0) && (static_sizes[i] > 0) && (static_start[i] >= 0));
            }
            auto slice = builder.create<tensor::ExtractSliceOp>(loc, out_type, input, ValueRange{}, ValueRange{}, ValueRange{}, static_start, static_sizes, static_step);
            context.addOutputs(node, slice);
            return;
        }
        
        auto index_type = importTensor(context.context, ov_index_shape, ov_index_element_type);
        auto empty = builder.create<tensor::EmptyOp>(loc, index_type, dynamic_index_dims);

        // TODO: this only works for the all-positive numbers case.
        auto sizes = builder.create<linalg::SubOp>(loc, mlir::ValueRange{stop, start}, mlir::ValueRange{empty});

        auto casted_type = RankedTensorType::get(importShape(ov_index_shape), builder.getIndexType()); 
        auto casted_start = builder.create<arith::IndexCastOp>(loc, casted_type, mlir::ValueRange{start});
        auto casted_sizes = builder.create<arith::IndexCastOp>(loc, casted_type, mlir::ValueRange{sizes.getResults()});
        auto casted_steps = builder.create<arith::IndexCastOp>(loc, casted_type, mlir::ValueRange{step});

        auto slice = builder.create<tensor::ExtractSliceOp>(loc, out_type, input, mlir::ValueRange{casted_start.getResult()}, mlir::ValueRange{casted_sizes.getResult()}, mlir::ValueRange{casted_steps.getResult()});
        //  /home/jovyan/graph-compiler/externals/llvm-project/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp:2038: static mlir::RankedTensorType mlir::tensor::ExtractSliceOp::inferResultType(mlir::RankedTensorType, llvm::ArrayRef<long int>, llvm::ArrayRef<long int>, llvm::ArrayRef<long int>): Assertion `static_cast<int64_t>(staticSizes.size()) == sourceTensorType.getRank() && "unexpected staticSizes not equal to rank of source"' failed.
        // auto slice = builder.create<tensor::ExtractSliceOp>(loc, input, mlir::ValueRange{casted_start.getResult()}, mlir::ValueRange{casted_sizes.getResult()}, mlir::ValueRange{casted_steps.getResult()});
        
        // auto slice = builder.create<tensor::ExtractSliceOp>(loc, input, mlir::ValueRange{start}, mlir::ValueRange{sizes.getResults()}, mlir::ValueRange{step});
        context.addOutputs(node, slice);
    }
};

}  // namespace

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;
using namespace ov::op;

SlicePattern::SlicePattern() : MarkPattern(wrap_type<v8::Slice>({any_input(), any_input(), any_input(), any_input(), any_input()}), ConvertSlice()) {}

}  // namespace mlir
}  // namespace ov
