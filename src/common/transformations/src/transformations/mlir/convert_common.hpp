// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// #include "llvm/MC/TargetRegistry.h"
// #include "llvm/Support/Casting.h"
// #include "llvm/Support/InitLLVM.h"
// #include "llvm/Support/MemoryBuffer.h"
// #include "llvm/Support/SourceMgr.h"
// #include "llvm/Support/TargetSelect.h"
// #include "llvm/Target/TargetMachine.h"
// #include "llvm/Target/TargetOptions.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Arith/Transforms/Passes.h"
// #include "mlir/Dialect/Bufferization/Transforms/Passes.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/Dialect/Linalg/Passes.h"
// #include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/Tensor/IR/Tensor.h"
// #include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
// #include "mlir/Dialect/Vector/IR/VectorOps.h"
// #include "mlir/ExecutionEngine/ExecutionEngine.h"
// #include "mlir/ExecutionEngine/JitRunner.h"
// #include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/Diagnostics.h"
// #include "mlir/IR/Dialect.h"
// #include "mlir/IR/Location.h"
// #include "mlir/IR/ValueRange.h"
// #include "mlir/InitAllDialects.h"
// #include "mlir/InitAllExtensions.h"
// #include "mlir/InitAllPasses.h"
// #include "mlir/Parser/Parser.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Pass/PassManager.h"
// #include "mlir/Support/LLVM.h"
// #include "mlir/Target/LLVMIR/Dialect/All.h"
// #include "mlir/Target/LLVMIR/Export.h"
// #include "mlir/Target/LLVMIR/ModuleTranslation.h"
// #include "openvino/core/dimension.hpp"
// #include "openvino/core/rt_info.hpp"
// #include "openvino/pass/pattern/op/wrap_type.hpp"
// #include "transformations_visibility.hpp"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"


#include "openvino/core/node.hpp"
#include "openvino/core/symbol.hpp"

#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace ov {
namespace mlir {

using namespace ::mlir;

using NodePtr = std::shared_ptr<ov::Node>;
using SymbolPtr = std::shared_ptr<ov::Symbol>;
using OVOutputTypes = std::vector<std::tuple<ov::element::Type, ov::PartialShape>>;


Location createLayerLocation(MLIRContext* ctx, const std::string& layerName, const std::string& layerType);

SmallVector<int64_t> importShape(const ov::PartialShape& shape);

Type importPrecision(MLIRContext* ctx, const ov::element::Type& precision);

RankedTensorType importTensor(MLIRContext* ctx,
                                    const ov::PartialShape& shape,
                                    const ov::element::Type& elemType);

Location createLocation(MLIRContext* ctx, NodePtr node);

} // namespace mlir
} // namespace ov