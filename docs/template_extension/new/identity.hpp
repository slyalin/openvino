// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include <openvino/op/op.hpp>
#include <frontend_manager/extension.hpp>

//! [op:header]
namespace TemplateExtension {



class Identity : public ov::op::Op {
public:
    OPENVINO_OP("Identity");
    //OPENVINO_FRAMEWORK_MAP(onnx, {}, {});
    //OPENVINO_FRAMEWORK_MAP(onnx);

    Identity() = default;
    Identity(const ov::Output<ov::Node>& arg);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::runtime::TensorVector& outputs, const ov::runtime::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
//! [op:header]

}  // namespace TemplateExtension
