// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"

namespace ov {

/**
 * @brief The base interface for OpenVINO operation extensions
 */
class OPENVINO_API BaseOpExtension : public Extension {
public:
    using Ptr = std::shared_ptr<BaseOpExtension>;
    /**
     * @brief Returns the type info of operation
     *
     * @return ov::DiscreteTypeInfo
     */
    virtual const ov::DiscreteTypeInfo& get_type_info() const = 0;
    /**
     * @brief Method creates an OpenVINO operation
     *
     * @param inputs vector of input ports
     * @param visitor attribute visitor which allows to read necessaty arguments
     *
     * @return vector of output ports
     */
    virtual ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) const = 0;

    /**
     * @brief Returns extensions that should be registered together with this extension class object.
     *
     * Attached extensions may include frontend extensions that OpenVINO op to framework ops or necessary
     * transformations that should be applied to the network which consist of target op.
     *
     * @return
     */
    virtual std::vector<ov::Extension::Ptr> get_attached_extensions() const { return {}; }

    /**
     * @brief Destructor
     */
    ~BaseOpExtension() override;
};

namespace detail {

#define _OV_DETAIL_COLLECT_ATTACHED_EXTENSIONS(FRAMEWORK)   \
    template <class T>\
    struct Has##FRAMEWORK##Map {\
        template <class U>\
        static auto test(U*) -> decltype(std::declval<U>().__openvino_framework_map_##FRAMEWORK(), std::true_type()) {\
            return {};\
        }\
        template <typename>\
        static auto test(...) -> std::false_type {\
            return {};\
        }\
        constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;\
    };\
\
    template <typename T>   \
    typename std::enable_if<!Has##FRAMEWORK##Map<T>::value, void>::type collect_attached_extensions_##FRAMEWORK (std::vector<ov::Extension::Ptr>&) {   \
        /* nothing for generic T */   \
    }   \
    \
    template <typename T>   \
    typename std::enable_if<Has##FRAMEWORK##Map<T>::value, void>::type collect_attached_extensions_##FRAMEWORK(std::vector<ov::Extension::Ptr>& v) { \
        v.emplace_back(typename T::template __openvino_framework_map_helper_##FRAMEWORK<T>().get());   \
    }

    _OV_DETAIL_COLLECT_ATTACHED_EXTENSIONS(onnx)
    _OV_DETAIL_COLLECT_ATTACHED_EXTENSIONS(paddlepaddle)
}

/**
 * @brief The default implementation of OpenVINO operation extensions
 */
template <class T>
class OpExtension : public BaseOpExtension {
public:
    /**
     * @brief Default constructor
     */
    OpExtension() {
        const auto& ext_type = get_type_info();
        OPENVINO_ASSERT(ext_type.name != nullptr && ext_type.version_id != nullptr,
                        "Extension type should have information about operation set and operation type.");
    }

    const ov::DiscreteTypeInfo& get_type_info() const override {
        return T::get_type_info_static();
    }

    ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) const override {
        std::shared_ptr<ov::Node> node = std::make_shared<T>();

        node->set_arguments(inputs);
        if (node->visit_attributes(visitor)) {
            node->constructor_validate_and_infer_types();
        }
        return node->outputs();
    }

    std::vector<ov::Extension::Ptr> get_attached_extensions() const override {
        std::vector<ov::Extension::Ptr> res;
        std::cout << "member function: " << detail::HasonnxMap<T>::value << "\n";
        detail::collect_attached_extensions_onnx<T>(res);
        return res;
    }
};

}  // namespace ov
