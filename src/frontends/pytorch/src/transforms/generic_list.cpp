// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_getitem_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/list.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"
#include "generic_list.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {


FWNodeMatcherPass::FWNodeMatcherPass(const std::string& fw_type) {
    //std::cerr << "Here\n";
    auto fw_node = ov::pass::pattern::wrap_type<PtFrameworkNode>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        //std::cerr << "[ FWNodeMatcherPass ] " << m.get_match_root() << "\n";
        auto fw_node = std::dynamic_pointer_cast<PtFrameworkNode>(
            cast_fw_node(m.get_match_root(), fw_type));
        if (!fw_node)
            return false;
        return replacer(fw_node);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        fw_node,
        "ov::frontend::pytorch::pass::FWNodeMatcherPass(" + fw_type + ")");
    this->register_matcher(m, callback);
}

bool GenericListConstruct::replacer (std::shared_ptr<PtFrameworkNode> fw_node) const {
    // Put all the inputs to the list,
    // Assume each input has a type of a tensor and they all have the same rank
    std::cerr << "[ GENERIC LIST ] Detected prim::ListConstruct " << fw_node << "\n";
    auto list = ov::pass::decompose_list_construct(fw_node->input_values());
    replace_node(fw_node, list.get_node_shared_ptr());
    return true;
}

bool GenericListAppend::replacer (std::shared_ptr<PtFrameworkNode> fw_node) const {

    std::cerr << "[ GENERIC LIST ] Detected prim::append " << fw_node << "\n";
    auto new_list = ov::pass::decompose_list_append(fw_node->get_input_node_shared_ptr(0), fw_node->input_value(1));
    replace_node(fw_node, {new_list, new_list});
    return true;
}

bool GenericListGetItem::replacer (std::shared_ptr<PtFrameworkNode> fw_node) const {

    std::cerr << "[ GENERIC LIST ] Detected aten::__getitem__ " << fw_node << "\n";
    auto item = ov::pass::decompose_list_get_item(fw_node->get_input_node_shared_ptr(0), fw_node->input_value(1));
    replace_node(fw_node, item.get_node_shared_ptr());
    return true;
}

bool GenericListSetItem::replacer (std::shared_ptr<PtFrameworkNode> fw_node) const {
    std::cerr << "[ GENERIC LIST ] Detected aten::_set_item " << fw_node << "\n";
    auto new_list = ov::pass::decompose_list_set_item(fw_node->get_input_node_shared_ptr(0), fw_node->input_value(1), fw_node->input_value(2));
    replace_node(fw_node, {new_list, new_list});
    return true;
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
