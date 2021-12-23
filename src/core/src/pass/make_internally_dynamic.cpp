// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/make_internally_dynamic.hpp"

#include <memory>
#include <ngraph/log.hpp>
#include <openvino/opsets/opset8.hpp>
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

// Find all Results and saves the original shape; will be used after dynamic transformation is applied to restore the shape
class SaveResultShapes : public MatcherPass {
public:

    static constexpr const char* shape_rt_tag = "MakeInternallyDynamic::orig_shape";

    SaveResultShapes() {
        using namespace ov;
        auto pattern = ov::pass::pattern::wrap_type<opset8::Result>();
        ov::matcher_pass_callback callback = [](pass::pattern::Matcher &m) {
            const auto &node = m.get_match_root();
            node->get_rt_info()[shape_rt_tag] = node->get_input_partial_shape(0);
            // Always return false, because we don't do real compute-affecting changes
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "SaveResultShapes");
        register_matcher(m, callback);
    }
};

class CutOutput : public MatcherPass {
public:

    CutOutput () {
        using namespace ov;
        auto pattern = ov::pass::pattern::wrap_type<opset8::DetectionOutput, opset8::Proposal>();
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

            auto flatten = make_shared<opset8::Reshape>(
                    make_shared<opset8::Gather>(
                            node->output(0),
                            make_shared<opset8::Constant>(ov::element::i32, Shape{1}, input_batch_pos),
                            make_shared<opset8::Constant>(ov::element::i32, Shape{}, attr_axis)),
                    make_shared<opset8::Constant>(ov::element::i32, Shape{1}, -1)->output(0), false);

            auto topk = make_shared<opset8::TopK>(
                    make_shared<opset8::Equal>(
                            flatten,
                            make_shared<opset8::Constant>(node->get_output_element_type(0), Shape{}, -1)),
                    make_shared<opset8::Constant>(ov::element::i32, Shape{}, 1),
                    0,
                    opset8::TopK::Mode::MAX,
                    opset8::TopK::SortType::SORT_INDICES);

            auto bound = make_shared<opset8::Select>(topk->output(0), topk->output(1), make_shared<opset8::Convert>(make_shared<opset8::ShapeOf>(flatten), ov::element::i32));

            ov::OutputVector concat_inputs = {
                make_shared<opset8::Maximum>(
                        bound,
                        make_shared<opset8::Constant>(ov::element::i32, Shape{1}, 1)),
                    make_shared<opset8::Constant>(ov::element::i32, Shape{end2.size()}, end2)};

            if (end1.size()) {
                concat_inputs.insert(concat_inputs.begin(), make_shared<opset8::Constant>(ov::element::i32, Shape{end1.size()}, end1));
            }

            auto postprocessing = make_shared<opset8::StridedSlice>(
                node->output(0),
                make_shared<opset8::Constant>(ov::element::i32, Shape{begin.size()}, begin),
                make_shared<opset8::Concat>(concat_inputs, 0),
                std::vector<int64_t>(rank, 1),
                end_mask);

            for (auto input : consumers) {
                input.replace_source_output(postprocessing);
            }

            std::cout << "[ DEBUG ] Still here\n";

            return true;
        };
        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "CutOutput");
        register_matcher(m, callback);
    }
};

class RestoreResult : public MatcherPass {
public:

    RestoreResult() {
        using namespace ov;
        auto pattern = ov::pass::pattern::wrap_type<opset8::Result>();
        ov::matcher_pass_callback callback = [](pass::pattern::Matcher &m) {
            const auto &node = m.get_match_root();

            // check applicability of this pass: result was marked by SaveResultShapes and differs from the original one
            // TODO: Learn how PS are compared and if upperbounds are compared as well, exclude them from comparison
            auto& rt_info = node->get_rt_info();
            auto rt = rt_info.find(SaveResultShapes::shape_rt_tag);
            if (rt == rt_info.end()) {
                // Why can we get here? New results? It cannot be.
                // TODO: trigger exception instead of silent return
                return false;
            }

            auto orig_shape = rt->second.as<PartialShape>();
            rt_info.erase(rt);
            auto cur_shape = node->get_input_partial_shape(0);

            if (orig_shape == cur_shape) {
                // Clean return, this result is turned to be not affected by our dynamic transform.
                return false;
            }

            if (orig_shape.rank().is_dynamic() || cur_shape.rank().is_dynamic() || orig_shape.rank() != cur_shape.rank()) {
                // TODO: Should revert the entire dynamic transformation, because we cannot track dimensions upto
                // the end but indeed dimensions are affected. Need to use more advanced dimension tracking to fix it.
                // Even the advanced dynamic tracking is used, there is a chance that we couldn't restore the result
                // anyway.
                return false;
            }

            int32_t rank = orig_shape.rank().get_length();

            int32_t axis = -1;  // axis for apdding; -1 means not found (yet)
            int64_t orig_static = 0;  // original static dimension that we have to pad the tensor
            std::vector<int32_t> left_part, right_part;
            // Identify dimensions that were static and become dynamic.
            for (int32_t i = 0; i < rank; ++i) {
                if (orig_shape[i].is_static() && cur_shape[i].is_dynamic()) {
                    // Expect a single dimension of such kind, if more -- we cannot restore orig shape -- fail
                    if (axis != -1) {
                        // Found two dimensions of the kind, cannot continue
                        // TODO: Should revert the entire dynamic transformation
                        return false;
                    }

                    axis = i;
                    orig_static = orig_shape[i].get_length();
                } else if (axis == -1) {
                    // axis is not yet found, on the left side: ...<here>..., axis
                    left_part.push_back(i);
                } else {
                    // axis has been found, on the right side from it: axis, ...<here>...
                    right_part.push_back(i);
                }
            }

            if (axis == -1) {
                // The differentce in dynamic upperbound only.
                // TODO: check if it is reachable
                return false;
            }

            // Parts for concat to build the final repeats parameter for Tile
            // Depending on position of axis in the shape it may contain different number of parts:
            // [..., axis, ...], [axis, ...], [..., axis] or [axis]
            OutputVector tile_shape_parts;

            auto input = node->get_input_node_shared_ptr(0);
            auto shape = make_shared<opset8::ShapeOf>(input);
            auto zero = make_shared<opset8::Constant>(ov::element::i64, Shape{}, 0);

            if (!left_part.empty()) {
                // fill the left part
                tile_shape_parts.push_back(
                        make_shared<opset8::Gather>(
                                shape,
                                make_shared<opset8::Constant>(ov::element::i64, Shape{left_part.size()}, left_part),
                                zero));
            }

            tile_shape_parts.push_back(
                    make_shared<opset8::Subtract>(
                            make_shared<opset8::Constant>(ov::element::i64, Shape{1}, orig_static),
                            make_shared<opset8::Gather>(
                                    shape,
                                    make_shared<opset8::Constant>(ov::element::i64, Shape{1}, axis),
                                    zero)));

            if (!right_part.empty()) {
                // fill the left part
                tile_shape_parts.push_back(
                        make_shared<opset8::Gather>(
                                shape,
                                make_shared<opset8::Constant>(ov::element::i64, Shape{right_part.size()}, right_part),
                                zero));
            }

            auto pad_shape = make_shared<opset8::Concat>(tile_shape_parts, 0);

/*


            // TODO: Analyse all side effects of such modification. It is not always safe.


            std::cerr << "[ INFO ] Dynamic Result " << node->get_input_partial_shape(0) << "\n";
            std::cerr << node->get_input_partial_shape(0)[0].get_max_length() << "\n";
            auto dyn_shape = node->get_input_partial_shape(0);
            const int max_size = 100;
            std::vector<size_t> final_size(dyn_shape.rank().get_max_length());
            final_size[0] = max_size;
            for(int i = 1; i < final_size.size(); ++i) {
                // TODO: make better than 2* for those dimensions that shouldn't be changed
                // 2* is needed because later we subtract the same value from this doubled value to have original value
                // for those dimensions that are static
                final_size[i] = 2*dyn_shape[i].get_length();
            }

            auto final_size_const = make_shared<opset8::Constant>(ov::element::i64, Shape{final_size.size()}, final_size);

            auto pad_shape = make_shared<opset8::Subtract>(final_size_const, make_shared<opset8::ShapeOf>(node->get_input_node_shared_ptr(0)));
*/
            auto pad = make_shared<opset8::Tile>(make_shared<opset8::Constant>(ov::element::f32, Shape{1}, -1), pad_shape);

            auto concat = make_shared<opset8::Concat>(ov::OutputVector{node->get_input_node_shared_ptr(0), pad}, 0);

            auto old_friendly_name = node->get_input_node_shared_ptr(0)->get_friendly_name();
            auto new_friendly_name = old_friendly_name + "/synthetic_123";
            node->get_input_node_shared_ptr(0)->set_friendly_name(new_friendly_name);
            concat->set_friendly_name(old_friendly_name);

            copy_runtime_info(node, concat);
            node->input(0).replace_source_output(concat);
            node->validate_and_infer_types();

            return true;
        };
        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "RestoreResult");
        register_matcher(m, callback);
    }
};

}  // namespace


bool ov::pass::MakeInternallyDynamic::run_on_function (std::shared_ptr<ov::Model> f) {
    ov::pass::Manager manager;

    manager.register_pass<SaveResultShapes>();

    //auto group = manager.register_pass<ov::pass::GraphRewrite>();
    manager.register_pass<CutOutput>();
    manager.register_pass<RestoreResult>();

    manager.run_passes(f);

    return true;
}

//NGRAPH_SUPPRESS_DEPRECATED_END
