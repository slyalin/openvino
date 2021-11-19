// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <fstream>
#include <type_traits>
#include "frontend_manager_defs.hpp"
#include <openvino/pass/pass.hpp>
#include <openvino/core/extension.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/manager.hpp>
#include "../../../../thirdparty/nlohmann/json/json.hpp"


namespace ngraph {
namespace frontend {

/// Holds a transformation that is applied just after the original model graph is decoded.
/** This class is a holder for transformation. The transformation can be specified as
 *  FunctionPass or MathcerPass derivatives or as a function that can be used to build corresponding
 *  FunctionPass or MatcherPass object. The type of the extension is determined in the moment of creation by
 *  calling corresponding ctor.
 */
class FRONTEND_API DecoderTransformationExtension : public ov::Extension {
public:

    DecoderTransformationExtension () {}

    // Create a custom functional pass where code of the pass is implemented as a function.
    DecoderTransformationExtension (std::function<bool(std::shared_ptr<ov::Function>)> function_pass);

    // Create a custom matcher pass where the code of matcher pass initialization is a given function.
    DecoderTransformationExtension (std::function<void(ov::pass::MatcherPass*)> matcher_pass_initializer);

    // Register existing transformation object which will be copied and kept for further registration.
    template <typename Transformation,
            typename std::enable_if<std::is_base_of<ov::pass::PassBase, Transformation>::value, bool>::type = true>
    DecoderTransformationExtension (const Transformation& transformation) :
        m_registration([transformation](ov::pass::Manager& manager) {
            manager.register_pass<Transformation>(transformation);
        }) {}

    // Register pass from this object in a given pass manager object
    void register_pass (ov::pass::Manager& manager) const;

protected:

    void set_registration (std::function<void(ov::pass::Manager&)> registration) {
        m_registration = registration;
    }

private:

    std::function<void(ov::pass::Manager&)> m_registration;
};


/// \brief Describes transformation that JsonConfigExtension can use as a target transformation spedified by ID.
/** JsonTransformationExtension passes JSON parsed object and mathed points in the graph to JsonTransformationExtension
 *  instance, which is derived from DecoderTransformationExtension. DecoderTransformationExtension itself cannot be
 *  used for this purpose because we need to pass those additional objects which are the result of JSON parsing and
 *  graph matching that JsonTransformationExtension performs.
 *
 *  This class is left for backward compatibility only. In the future we would like to get rid off this class as well
 *  as from JsonConfigExtension, and users will use DecoderTransformationExtension only to match sub-graph and modify it
 *  like we do in any other transformations.
 *
 *  Unlike DecoderTransformationExtension, which is initialized by some ready-to-use transformation code and is not used
 *  to derive new classes by regular users, this class is intended to be derived from and it doesn't have convenient
 *  ctos to be initialized. So it is intended for more advanced, internal users inside such components like Model Optimizer.
 */
class FRONTEND_API JsonTransformationExtension : public DecoderTransformationExtension {
public:

    JsonTransformationExtension (const std::string& id) : m_id(id) {}

    // The name of the transformation to identify it from JSON file field 'id'
    const std::string& id () const { return m_id; }

    virtual bool transform (
            std::shared_ptr<ov::Function>& function,
            const nlohmann::json& replacement_descriptions
    ) const = 0;

private:

    std::string m_id;
};


/// Reads MO config file and delegate transformation functionality to specified transformation ID
/// specified in the config.
class FRONTEND_API JsonConfigExtension : public DecoderTransformationExtension {
public:
    JsonConfigExtension (const std::string& config_path);
    ~JsonConfigExtension ();
private:
    std::vector<Extension::Ptr> m_loaded_extensions;
    std::shared_ptr<DecoderTransformationExtension> m_target_extension;
    const nlohmann::json m_replacement_descriptions;
};


/// \brief Provides callback to report telemetry information back to Python code
class FRONTEND_API TelemetryExtension : public ov::Extension {
public:

    TelemetryExtension (std::function<void(const std::string& message)> callback) : m_callback(callback) {}
    void send (const std::string& message) { m_callback(message); }

private:

    std::function<void(const std::string& message)> m_callback;
};


class FRONTEND_API NodeContext {
public:
    NodeContext (const std::string& _op_type, OutputVector _ng_inputs) : m_op_type(_op_type), m_ng_inputs(_ng_inputs) {}
    OutputVector get_ng_inputs() const { return m_ng_inputs; }
    const std::string& op_type() const { return m_op_type; }

    template <typename T>
    T get_attribute (const std::string& name) {
        // TODO: Replace this stub by a real implementation here
        return T();
    }

private:

    OutputVector m_ng_inputs;
    std::string m_op_type;
};


class FRONTEND_API ConversionExtension : public ov::Extension {
public:

    ConversionExtension (const std::string& optype, std::function<OutputVector(std::shared_ptr<NodeContext>)> converter) :
        m_optype(optype), m_converter(converter) {}

    std::string m_optype;
    std::function<OutputVector(std::shared_ptr<NodeContext>)> m_converter;
};

// One-to-one operation mapping for OVOpType != void which means OV type is specified by OVOpType
// See a specialization for OVOptype = void
template <typename OVOpType=void>
class FRONTEND_API OpExtension : public ConversionExtension { // TODO: Consider deriving from base Extension class
public:

    // All attributes come from OVOpType definition, op type in FW and OV match, available for OVOpType != void only
    // Attributes mapping can be modified with optional parameters
    OpExtension (const std::map<std::string, std::string>& attr_names_map = {},
                 const std::map<std::string, std::string>& attr_values_map = {}) :
            OpExtension(OVOpType::get_type_info_static().name, attr_names_map, attr_values_map) {}

    // Maps op with a given type in FW and OV type given in template parameter
    OpExtension (
            const std::string& fw_type_name,
            const std::map<std::string, std::string>& attr_names_map = {},
            const std::map<std::string, std::string>& attr_values_map = {});

};

class FWVisitor : public ov::AttributeVisitor {
public:
    explicit FWVisitor(
            std::shared_ptr<NodeContext> context,
            const std::map<std::string, std::string> &attr_names_map = {},
            const std::map<std::string, std::string> &attr_values_map = {}) :
            m_context(context), m_attr_names_map(attr_names_map), m_attr_values_map(attr_values_map) {}

    void on_adapter (const std::string& name, ValueAccessor<void>& a) override {
        if(auto adapter = ngraph::as_type<ngraph::AttributeAdapter<std::string>>(&a)) {
            auto p_value = m_attr_values_map.find(name);
            if (p_value != m_attr_values_map.end()) {
                adapter->set(p_value->second);
            } else {
                p_value = m_attr_names_map.find(name);
                const std::string &target_name = p_value != m_attr_names_map.end() ? p_value->second : name;
                adapter->set(m_context->get_attribute<std::string>(target_name));
            }
        }
    }

private:
    std::shared_ptr<NodeContext> m_context;
    const std::map<std::string, std::string> &m_attr_names_map;
    const std::map<std::string, std::string> &m_attr_values_map;
};

template <typename OVOpType>
OpExtension<OVOpType>::OpExtension (const std::string& fw_type_name,
                                    const std::map<std::string, std::string>& attr_names_map,
                                    const std::map<std::string, std::string>& attr_values_map) :
    ConversionExtension(
            fw_type_name,
            [attr_names_map, attr_values_map](std::shared_ptr<NodeContext> context) {
                std::cerr << "[ INFO ] Activated OpExtension!\n";
                auto node = std::make_shared<OVOpType>();
                node->set_arguments(context->get_ng_inputs());
                FWVisitor fwvisitor(context, attr_names_map, attr_values_map);
                node->visit_attributes(fwvisitor);
                node->validate_and_infer_types();
                return node->outputs();
            }
    )
{
        std::cerr << "[ INFO ] Registered OpExtension\n";
}

template <>
class FRONTEND_API OpExtension<void> : public ConversionExtension { // TODO: Consider deriving from base Extension class
public:

    // Default ctor is not available, you need to specify OV type with another ctor
    OpExtension () = delete;

    // Maps op with a given type in FW and matching OV type given in template parameter
    OpExtension (
            const std::string& fw_ov_type_name,
            const std::map<std::string, std::string>& attr_names_map = {},
            const std::map<std::string, std::string>& attr_values_map = {});

    // Maps op with a given type in FW and specified OV type given in template parameter
    OpExtension (
            const std::string& ov_type_name,
            const std::string& fw_type_name,
            const std::map<std::string, std::string>& attr_names_map = {},
            const std::map<std::string, std::string>& attr_values_map = {});
};

}  // namespace frontend

}  // namespace ngraph
