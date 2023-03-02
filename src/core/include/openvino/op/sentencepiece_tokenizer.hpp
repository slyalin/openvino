// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <cassert>

#include "openvino/opsets/opset10.hpp"
#include "openvino/core/type/non_tensor_type.hpp"

// For some helper structures
#include "str_ops.hpp"

namespace ov {

// This is a temporary extension op that consumes multiple operations from TF graph:
// SentencepieceOp + SentencepieceTokenizeOp + RaggedTensorToSparse
// It supports both structural type Str as a single input and decomposed Str Tensor
// represented as regular 3 OV tensors: indices of begins, indices of ends and
// all strings concatenated as U8 1D tensor
class OPENVINO_API SentencepieceTokenizerExtensionOp : public frontend::tensorflow::StructuralTypedOp {
public:
    OPENVINO_OP("SentencepieceTokenizerExtensionOp", "0",  frontend::tensorflow::StructuralTypedOp);

    SentencepieceTokenizerExtensionOp(
        const OutputVector& arguments,
        // TODO: Add necessary attribute parameters or extra constant inputs based on TF graph nodes
        const  frontend::tensorflow::StructuralTypeProxy::BindInputs& bind_inputs = {}
    )
    : StructuralTypedOp(arguments, bind_inputs) {
        constructor_validate_and_infer_types();
    }

    const int max_size = 10;
    const std::vector<int32_t> stub_indices = {0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6};
    const std::vector<int32_t> stub_elements = {1, 37967, 12, 433, 9215, 2, 1, 6, 106425, 12, 1400, 9215, 2};
    const std::vector<int32_t> stub_dims = {2, 7};

    void validate_and_infer_types() override {

        // Handle validation model and evaluatation mode due to CPU bug (see other ops)

        // TODO: Move to cpp file

        if(all_inputs_are_constants(this)) {
            // Fake outputs
            #if 0
            set_output_type(0, element::i32, PartialShape{Dimension(max_size), Dimension(2)});
            set_output_type(1, element::i32, PartialShape{max_size});
            set_output_type(2, element::i32, PartialShape{2});
            #else
            set_output_type(0, element::i32, PartialShape{Dimension(stub_elements.size()), Dimension(2)});
            set_output_type(1, element::i32, PartialShape{Dimension(stub_elements.size())});
            set_output_type(2, element::i32, PartialShape{2});
            #endif
        } else {
            set_output_type(0, element::i32, PartialShape{Dimension(), Dimension(2)});
            set_output_type(1, element::i32, PartialShape{Dimension()});
            set_output_type(2, element::i32, PartialShape{2});
        }
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<SentencepieceTokenizerExtensionOp>(
            inputs,
             frontend::tensorflow::StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info()));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // Add necessary attributes if any
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        // inputs should have at least 3 tensors for input strings
        // [0] i32 tensor of begin indices, indices are offsets in [2]
        // [1] i32 tensor of end indices, indices are offsets in [2]
        // [2] 1D u8 tensor of bytes where all strings are concatenated

        // TODO: Implement the kernel
        // TODO: Move to cpp file

        // Now generate some stub data
        // TODO: Remove this code from final version

        auto data = (const char*)inputs[2].data<uint8_t>();
        size_t len = inputs[2].get_shape()[0];
        std::string symbols(data, data + len);
        std::cerr << "symbols at the input in tokenizer: " << symbols << "\n";
        size_t batch_size = inputs[0].get_shape()[0];
        std::cerr << "batch size = " << batch_size << "\n";

        //int offset = 0;
        //int i;
        #if 0
        outputs[2].data<int32_t>()[0] = 0;
        outputs[2].data<int32_t>()[1] = 0;
        for(int i = 0, offset = 0; offset < max_size; ++i) {
            for(int batch = 0; batch < batch_size && offset < max_size; ++batch, ++offset) {
                outputs[0].data<int32_t>()[2*i + 0] = batch;
                outputs[0].data<int32_t>()[2*i + 1] = i;
                outputs[1].data<int32_t>()[offset] = offset;
                outputs[2].data<int32_t>()[0] = std::max(batch, outputs[2].data<int32_t>()[0]);
                outputs[2].data<int32_t>()[1] = std::max(i, outputs[2].data<int32_t>()[1]);
                std::cerr << outputs[0].data<int32_t>()[2*i + 0] << '\n';
                std::cerr << outputs[0].data<int32_t>()[2*i + 1] << '\n';
                std::cerr << outputs[1].data<int32_t>()[offset] << '\n';
                std::cerr << outputs[2].data<int32_t>()[0] << '\n';
                std::cerr << outputs[2].data<int32_t>()[1] << '\n';
            }
        }
        outputs[2].data<int32_t>()[0]++;
        outputs[2].data<int32_t>()[1]++;
        #else
        std::copy(stub_indices.begin(), stub_indices.end(), outputs[0].data<int32_t>());
        std::copy(stub_elements.begin(), stub_elements.end(), outputs[1].data<int32_t>());
        std::copy(stub_dims.begin(), stub_dims.end(), outputs[2].data<int32_t>());
        #endif



        return true;
    }

    bool has_evaluate() const {
        return true;
    }
};


}  // namespace ov
