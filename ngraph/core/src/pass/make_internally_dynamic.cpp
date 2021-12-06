// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/make_internally_dynamic.hpp"

#include <memory>
#include <ngraph/log.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/pass/pass.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/builder/make_constant.hpp>

//NGRAPH_SUPPRESS_DEPRECATED_START
//NGRAPH_RTTI_DEFINITION(ov::pass::MakeInternallyDynamic, "MakeInternallyDynamic", 0);

using namespace std;

namespace {

using ov::pass::MatcherPass;

class CutOutput : public MatcherPass {
public:

    CutOutput () {
        using namespace ov;
        auto pattern = ov::pass::pattern::wrap_type<opset7::DetectionOutput, opset7::Proposal>();
        ov::matcher_pass_callback callback = [](pass::pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            std::cout << "[ DEBUG ] Found op for MakeInternallyDynamic: " << node->get_friendly_name() << std::endl;

            auto output_shape = node->get_output_partial_shape(0);
            if (output_shape.rank().is_dynamic()) {
                return false;
            }

            auto input_shape = node->get_input_partial_shape(0);
            if (input_shape.rank().is_dynamic() || input_shape[0].get_length() != 1) {
                // input batch should be equal to 1 to simplify calculations for specific case of DetectionOutput
                // TODO: Eliminate this limitation
                return false;
            }

            const auto rank = output_shape.rank().get_length();
            const auto output_batch_axis = rank - 2;
            const auto attr_axis = rank - 1;
            const size_t input_batch_pos = 0;
            const std::vector<int32_t> begin(rank);
            const std::vector<int32_t> end1(output_batch_axis);
            const std::vector<int32_t> end2(rank - output_batch_axis - 1);
            std::vector<int64_t> end_mask(rank, 1);
            end_mask[output_batch_axis] = 0;

            auto consumers = node->output(0).get_target_inputs();

            ov::OutputVector concat_inputs = {make_shared<opset7::Reshape>(
                    make_shared<opset7::TopK>(
                            make_shared<opset7::Equal>(
                                    make_shared<opset7::Gather>(
                                            node->output(0),
                                            make_shared<opset7::Constant>(ov::element::i32, Shape{1}, input_batch_pos)->output(0),
                                            make_shared<opset7::Constant>(ov::element::i32, Shape{}, attr_axis)->output(0)),
                                    make_shared<opset7::Constant>(node->get_output_element_type(0), Shape{}, -1)),
                            make_shared<opset7::Constant>(ov::element::i32, Shape{}, 1),
                            output_batch_axis,
                            opset7::TopK::Mode::MAX,
                            opset7::TopK::SortType::SORT_INDICES)->output(1),
                    make_shared<opset7::Constant>(ov::element::i32, Shape{1}, 1)->output(0), false)->output(0),
                    make_shared<opset7::Constant>(ov::element::i32, Shape{end2.size()}, end2)->output(0)};

            if (end1.size()) {
                concat_inputs.insert(concat_inputs.begin(), make_shared<opset7::Constant>(ov::element::i32, Shape{end1.size()}, end1)->output(0));
            }

            auto postprocessing = make_shared<opset7::StridedSlice>(
                node->output(0),
                make_shared<opset7::Constant>(ov::element::i32, Shape{begin.size()}, begin),
                make_shared<opset7::Concat>(concat_inputs, 0),
                std::vector<int64_t>(rank, 1),
                end_mask);

            for (auto input : consumers) {
                input.replace_source_output(postprocessing);
            }

            std::cout << "[ DEBUG ] Still here\n";

            return true;
        };
        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "ReshapeFullyConnectedFusion");
        register_matcher(m, callback);
    }
};

}  // namespace


bool ov::pass::MakeInternallyDynamic::run_on_function (std::shared_ptr<ov::Function> f) {
    ov::pass::Manager manager;
    manager.register_pass<CutOutput>();
    manager.run_passes(f);
    return true;
}

//NGRAPH_SUPPRESS_DEPRECATED_END
