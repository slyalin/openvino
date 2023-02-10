// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>

#include <openvino/op/str_ops.hpp>
#include <ngraph/opsets/opset10.hpp>

using std::make_shared;
using std::shared_ptr;

namespace ov {
namespace pass {


bool decompose_list_set_item (
    shared_ptr<Node> set_item,
    shared_ptr<Node> list,
    Output<Node> index_scalar,
    Output<Node> item
) {
    using namespace opset10;
    using ov::frontend::tensorflow::StructPack;

    std::cerr << "[ LIST ] decompose_list_set_item\n";

    auto sp = std::dynamic_pointer_cast<StructPack>(list);
    if(!sp) {
        std::cerr
            << "[ ERROR ] Coudn't decode StructPack in the list input: " << list << "\n";
        return false;
    }

    auto shapes =   sp->get_input_source_output(0);
    auto begins =   sp->get_input_source_output(1);
    auto ends =     sp->get_input_source_output(2);
    auto elements = sp->get_input_source_output(3);

    auto zero_1d = const_value(0, 1, begins.get_element_type());
    auto zero = const_value(0);
    typedef std::vector<int64_t> V;

    auto index = make_shared<Unsqueeze>(index_scalar, zero);

    auto begin = make_shared<Gather>(begins, index, zero);
    auto end = make_shared<Gather>(ends, index, zero);
    auto len = make_shared<ShapeOf>(begins, begins.get_element_type());

    // Get two parts of elements: before and after a target item area
    auto before = make_shared<StridedSlice>(elements, zero_1d, begin, V{1}, V{0});
    auto after = make_shared<StridedSlice>(elements, end, zero_1d, V{0}, V{1});
    auto flat = make_shared<Reshape>(item, const_value(-1, 1), false);
    auto new_elements = make_shared<Concat>(OutputVector{before, flat, after}, 0);
    auto new_end = make_shared<Add>(begin, make_shared<ShapeOf>(flat, begins.get_element_type()));
    auto shift = make_shared<Subtract>(new_end, end);

    auto one_1d = const_value(1, 1, begins.get_element_type());

    // TODO: begins_shift/ends_shift don't look very efficient, try StridedSplice, Add and Concat instead

    auto begins_shift = make_shared<Concat>(OutputVector{
        make_shared<Tile>(zero_1d, make_shared<Add>(index, one_1d)),
        make_shared<Tile>(shift, make_shared<Subtract>(make_shared<Subtract>(len, index), one_1d))},
        0);

    auto ends_shift = make_shared<Concat>(OutputVector{
        make_shared<Tile>(zero_1d, index),
        make_shared<Tile>(shift, make_shared<Subtract>(len, index))},
        0);

    auto new_begins = make_shared<Add>(begins, begins_shift);
    auto new_ends = make_shared<Add>(ends, ends_shift);

    //auto shape = make_shared<SpyOp>(OutputVector{make_shared<ShapeOf>(item, shapes.get_element_type())});
    auto shape = make_shared<ShapeOf>(item, shapes.get_element_type());

    #if 0 // this part has an issue in CPU: ranks mismatch presumably due to scalar index
    auto new_shapes = make_shared<ScatterUpdate>(shapes, index_scalar, shape, zero);
    #else
    #if 1
    //auto new_shapes = make_shared<SpyOp>(OutputVector{make_shared<ScatterUpdate>(shapes, index, make_shared<Unsqueeze>(shape, zero), zero)});
    auto new_shapes = make_shared<ScatterUpdate>(shapes, index, make_shared<Unsqueeze>(shape, zero), zero);
    #else
    auto index_2d = make_shared<Unsqueeze>(index, zero);
    make_shared<StridedSlice>(shapes, const_value(0, 2), index_2d, V{1, 1}, V{0, 1})
    make_shared<StridedSlice>(shapes, make_index_2d) ...
    #endif
    #endif

    auto new_sp = sp->clone_with_new_inputs({new_shapes, new_begins, new_ends, new_elements});

    replace_node(set_item, new_sp);

    return true;
}


bool decompose_list_get_item (
    std::shared_ptr<Node> get_item,
    std::shared_ptr<Node> list,
    Output<Node> index
) {
    using namespace opset10;
    using ov::frontend::tensorflow::StructPack;

    std::cerr << "[ LIST ] decompose_list_set_item\n";

    auto sp = std::dynamic_pointer_cast<StructPack>(list);
    if(!sp) {
        std::cerr << "[ ERROR ] Coudn't decode StructPack as list at get_item\n";
        std::cerr << list << "\n";
        return false;
    }

    auto shapes =   sp->get_input_source_output(0);
    auto begins =   sp->get_input_source_output(1);
    auto ends =     sp->get_input_source_output(2);
    auto elements = sp->get_input_source_output(3);

    auto zero = const_value(0);
    typedef std::vector<int64_t> V;

    // Get part of elements which correspond to a required item tensor
    auto flat = make_shared<StridedSlice>(
        elements,
        make_shared<Unsqueeze>(make_shared<Gather>(begins, index, zero), zero),
        make_shared<Unsqueeze>(make_shared<Gather>(ends,   index, zero), zero),
        V{0}, V{0});

    // auto flat = make_shared<StridedSlice>(
    //     elements,
    //     make_shared<SpyOp>(OutputVector{make_shared<Unsqueeze>(make_shared<Gather>(begins, index, zero), zero)}),
    //     make_shared<SpyOp>(OutputVector{make_shared<Unsqueeze>(make_shared<Gather>(ends,   index, zero), zero)}),
    //     V{0}, V{0});

    // Get shape that belongs to that area
    auto shape = make_shared<Gather>(shapes, index, zero);

    // Shape `flat` to obtained `shape`
    // TODO: In case of TF learn how to use node->input(2) which contains item shape (double Reshape to fix rank? not sure it adds value...)
    //auto item = make_shared<SpyOp>(OutputVector{make_shared<Reshape>(flat, shape, false)});
    auto item = make_shared<Reshape>(flat, shape, false);
    replace_node(get_item, item);
    return true;
}

bool decompose_list_construct (
    std::shared_ptr<Node> list_construct,
    OutputVector inputs
) {
    using namespace opset10;
    using ov::frontend::tensorflow::StructPack;

    std::cerr << "[ LIST ] decompose_list_construct\n";

    // Take the first tensor (if any) element type as a final list tensor element type
    element::Type element_type = element::dynamic;
    if(!inputs.empty()) {
        element_type = inputs[0].get_element_type();
    }

    if(element_type == element::undefined || element_type == element::dynamic) {
        element_type = element::u8; // as universal placeholder, it implies having ConvertLike each time when we putting something in the list
    }

    // Suppose we have at least one input and all inputs have the same rank

    OutputVector elements_for_concat;
    OutputVector shapes_for_concat;

    for(size_t i = 1; i < inputs.size(); ++i) {
        // Port existing code from working branch with lists but in more generic way
    }

    auto tensor_shape = make_shared<ShapeOf>(tensor, shape_type);
    //zero_1d = const_value(0, 1, shape_type);
    auto one_1d = const_value(1, 1, shape_type);
    typedef std::vector<int64_t> V;
    auto num_elements = make_shared<StridedSlice>(tensor_shape, one_1d, one_1d, V{1}, V{0});
    auto real_element_shape = make_shared<StridedSlice>(tensor_shape, one_1d, one_1d, V{0}, V{1});

    auto shapes = make_shared<opset10::Tile>(
        real_element_shape, make_shared<Concat>(
            OutputVector{num_elements, const_value(1, 1, shape_type)}, 0));

    auto total_element_size = make_shared<ReduceProd>(real_element_shape, const_value(0));
    auto num_elements_scalar = make_shared<Squeeze>(num_elements);

    // auto begins = make_shared<SpyOp>(OutputVector{make_shared<Range>(
    //     const_value(0),
    //     make_shared<Multiply>(num_elements_scalar, total_element_size),
    //     total_element_size,
    //     shape_type)});

    // auto ends = make_shared<SpyOp>(OutputVector{make_shared<Range>(
    //     total_element_size,
    //     make_shared<Multiply>(
    //         make_shared<Add>(num_elements_scalar, const_value(1, 0, shape_type)),
    //         total_element_size),
    //     total_element_size,
    //     shape_type)});

    auto begins = make_shared<Range>(
        const_value(0),
        make_shared<Multiply>(num_elements_scalar, total_element_size),
        total_element_size,
        shape_type);

    auto ends = make_shared<Range>(
        total_element_size,
        make_shared<Multiply>(
            make_shared<Add>(num_elements_scalar, const_value(1, 0, shape_type)),
            total_element_size),
        total_element_size,
        shape_type);

    auto elements = make_shared<Reshape>(tensor, const_value(-1, 1), true);

    return make_shared<StructPack>(
        OutputVector{shapes, begins, ends, elements},
        element::StructuralType::TensorListWithRank(element_type, element_rank),
        PartialShape::dynamic())->outputs();
}

}
}
