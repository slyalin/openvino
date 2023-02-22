// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>

#include <openvino/pass/list.hpp>
#include <openvino/op/str_ops.hpp>
#include <ngraph/opsets/opset10.hpp>

using std::make_shared;
using std::shared_ptr;

namespace ov {
namespace pass {

using ov::frontend::tensorflow::StructPack;

struct Diagnostics {
    Diagnostics(const std::string& _title) : title(_title) {
        std::cerr << "[ LIST ] Start " << title << "\n";
    }
    ~Diagnostics() {
        if(std::uncaught_exception()) {
            std::cerr << "Exception was thrown\n";
        }
        std::cerr << "[ LIST ] End " << title << "\n";
    }
    std::string title;
};

Output<Node> decompose_list_set_item (
    shared_ptr<Node> list,
    Output<Node> index_scalar,
    Output<Node> item
) {
    using namespace opset10;
    using ov::frontend::tensorflow::StructPack;

    Diagnostics _diag("decompose_list_set_item");

    auto sp = std::dynamic_pointer_cast<StructPack>(list);
    if(!sp) {
        throw "[ ERROR ] Coudn't decode StructPack in the list input";
    }

    auto shapes =   sp->get_input_source_output(0);
    auto begins =   sp->get_input_source_output(1);
    auto ends =     sp->get_input_source_output(2);
    auto elements = sp->get_input_source_output(3);

    auto zero_1d = const_value(0, 1, begins.get_element_type());
    auto zero = const_value(0);
    typedef std::vector<int64_t> V;

    // TODO: What should be the target index element type?
    auto index = make_shared<Convert>(make_shared<Unsqueeze>(index_scalar, zero), begins.get_element_type());
    std::cerr << "***\n";

    auto begin = make_shared<Gather>(begins, index, zero);
    auto end = make_shared<Gather>(ends, index, zero);
    auto len = make_shared<ShapeOf>(begins, begins.get_element_type());
    std::cerr << len << "\n";

    std::cerr << "***\n";
    // Get two parts of elements: before and after a target item area
    auto before = make_shared<StridedSlice>(elements, zero_1d, begin, V{1}, V{0});
    auto after = make_shared<StridedSlice>(elements, end, zero_1d, V{0}, V{1});
    auto flat = make_shared<Reshape>(item, const_value(-1, 1), false);
    auto new_elements = make_shared<Concat>(OutputVector{before, flat, after}, 0);
    auto new_end = make_shared<Add>(begin, make_shared<ShapeOf>(flat, begins.get_element_type()));
    auto shift = make_shared<Subtract>(new_end, end);

    auto one_1d = const_value(1, 1, begins.get_element_type());

    // TODO: begins_shift/ends_shift don't look very efficient, try StridedSplice, Add and Concat instead

    std::cerr << index << "\n";
    std::cerr << "***\n";
    auto p3 = make_shared<Add>(index, one_1d);
    std::cerr << "///\n" << p3 << "\n";
    auto p1 = make_shared<Tile>(zero_1d, p3);
    std::cerr << "---\n" << p1 << "\n";
    auto p2 = make_shared<Tile>(shift, make_shared<Subtract>(make_shared<Subtract>(len, index), one_1d));
    std::cerr << "+++\n" << p2 << "\n";
    auto begins_shift = make_shared<Concat>(OutputVector{
        p1,
        p2},
        0);

    std::cerr << "***\n";
    auto ends_shift = make_shared<Concat>(OutputVector{
        make_shared<Tile>(zero_1d, index),
        make_shared<Tile>(shift, make_shared<Subtract>(len, index))},
        0);

    auto new_begins = make_shared<Add>(begins, begins_shift);
    auto new_ends = make_shared<Add>(ends, ends_shift);

    std::cerr << "***\n";
    auto shape_type = shapes.get_element_type();

    //auto shape = make_shared<SpyOp>(OutputVector{make_shared<ShapeOf>(item, shapes.get_element_type())});
    auto shape = make_shared<ShapeOf>(item, shape_type);
    auto rank = make_shared<ShapeOf>(shape, shape_type);
    // Reshape initial shapes in case if it was created without knowing the shape that means it is empty with wrong shape
    auto preshape = make_shared<Concat>(OutputVector{const_value(-1, 1, shape_type), rank}, 0);
    shapes = make_shared<Reshape>(shapes, preshape, false);
    std::cerr << "***\n";

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

    return sp->clone_with_new_inputs({new_shapes, new_begins, new_ends, new_elements});

    //replace_node(set_item, new_sp);

    //return true;
}


Output<Node> decompose_list_append (
    shared_ptr<Node> list,
    Output<Node> item
) {
    using namespace opset10;
    using ov::frontend::tensorflow::StructPack;

    Diagnostics _diag("decompose_list_append");

    auto sp = std::dynamic_pointer_cast<StructPack>(list);
    if(!sp) {
        throw "[ ERROR ] Coudn't decode StructPack in the list input";
    }

    auto shapes =   sp->get_input_source_output(0);
    auto begins =   sp->get_input_source_output(1);
    auto ends =     sp->get_input_source_output(2);
    auto elements = sp->get_input_source_output(3);

    //auto zero_1d = const_value(0, 1, begins.get_element_type());
    auto zero = const_value(0);

    // Append begin offset
    auto new_begins = make_shared<Concat>(OutputVector{begins, make_shared<ShapeOf>(elements, begins.get_element_type())}, 0);

    // Cast elements to type from item, because elements can be initialized without knowing real element type (in case of empty list)
    auto elements_casted = make_shared<ConvertLike>(elements, item);
    auto flat = make_shared<Reshape>(item, const_value(-1, 1), false);
    auto new_elements = make_shared<Concat>(OutputVector{elements_casted, flat}, 0);
    auto new_ends = make_shared<Concat>(OutputVector{ends, make_shared<ShapeOf>(new_elements, ends.get_element_type())}, 0);

    auto shape_type = shapes.get_element_type();
    auto num_elements = make_shared<ShapeOf>(begins, shape_type);  // the same as for ends

    auto shape = make_shared<ShapeOf>(item, shape_type);
    auto rank = make_shared<ShapeOf>(shape, shape_type);
    // Reshape initial shapes in case if it was created without knowing the shape that means it is empty with wrong shape
    auto preshape = make_shared<Concat>(OutputVector{num_elements, rank}, 0);   // do not use -1 because it doesn't work always correctly for empty input tensors
    std::cerr << "HERE\n";
    shapes = make_shared<Reshape>(shapes, preshape, false);
    auto new_shapes = make_shared<Concat>(OutputVector{shapes, make_shared<Unsqueeze>(shape, zero)}, 0);

    std::cerr << "[ LIST ] Near the end of decompose_list_append\n";
    std::cerr << "Shapes after decompose_list_append: " << new_shapes << "\n";


    return sp->clone_with_new_inputs({new_shapes, new_begins, new_ends, new_elements});
}

Output<Node> decompose_list_get_item (
    std::shared_ptr<Node> list,
    Output<Node> index
) {
    using namespace opset10;
    using ov::frontend::tensorflow::StructPack;

    Diagnostics _diag("decompose_list_get_item");

    auto sp = std::dynamic_pointer_cast<StructPack>(list);
    if(!sp) {
        throw  "[ ERROR ] Coudn't decode StructPack as list at get_item\n";
    }

    auto shapes =   sp->get_input_source_output(0);
    auto begins =   sp->get_input_source_output(1);
    auto ends =     sp->get_input_source_output(2);
    auto elements = sp->get_input_source_output(3);

    auto zero = const_value(0);
    typedef std::vector<int64_t> V;

    std::cerr << "index = " << index << "\n";
    std::cerr << "begins = " << make_shared<Unsqueeze>(make_shared<Gather>(begins, index, zero), zero) << "\n";

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
    return make_shared<Reshape>(flat, shape, false);
}

Output<Node> decompose_list_reserve (
    Output<Node> element_shape,
    Output<Node> num_elements_scalar,
    element::Type element_type,
    element::Type shape_type
) {
    Diagnostics _diag("decompose_list_reserve");
    auto num_elements = std::make_shared<opset10::Reshape>(num_elements_scalar, const_value(1, 1), false);

    // known rank of elements implies element_shape has static shape
    OPENVINO_ASSERT(element_shape.get_partial_shape().is_static(), "element_shape is not static");
    OPENVINO_ASSERT(element_shape.get_shape().size() == 1, "element_shape is not 1D tensor");
    auto element_rank = element_shape.get_shape()[0];
    std::cerr << "[ TF FE INFO ] Element rank = " << element_rank << "\n";

    // Form concatenated shapes tensor as zeros of [num_elements, element_rank] shape

    auto shape_shape = std::make_shared<opset10::Concat>(
        OutputVector{num_elements, std::make_shared<opset10::ShapeOf>(element_shape, shape_type)}, 0);

    auto shapes = std::make_shared<opset10::Tile>(const_value(0, 2, shape_type), shape_shape);

    // Use one tensor with zeros for both begins and ends as there are no real element in tensors
    auto indices = std::make_shared<opset10::Tile>(const_value(0, 1, shape_type), num_elements);

    // An empty tensor
    // FIXME: This should be an empty tensor but it breaks transformation flow which improperly over-optimize loop bodies
    // FIXME: That's why a padding in one element is used to keep it not empty. In all other operations this element is ignored
    // FIXME: due to nature of index operations. The only exception is in the operation which turns a list to a tensor,
    // FIXME: there will be an extra StridedSlice to cut off this padding.
    auto elements = opset10::Constant::create(element_type, {1}, {0});

    std::cerr << "Shapes after decompose_list_reserve: " << shapes << "\n";

    return make_shared<StructPack>(
        OutputVector{shapes, indices, indices, elements},
        element::StructuralType::TensorListWithRank(element_type, element_rank),
        PartialShape::dynamic());
}


Output<Node> decompose_list_construct (
    OutputVector inputs,
    element::Type element_type,
    element::Type shape_type
) {
    Diagnostics _diag("decompose_list_construct");
    Output<Node> sp = decompose_list_reserve(
        make_shared<opset10::Constant>(shape_type, Shape{0}),
        const_value(0, 0),
        element_type,
        shape_type);

    for(auto input: inputs) {
        sp = decompose_list_append(sp.get_node_shared_ptr(), input);
    }

    return sp;

    #if 0
    using namespace opset10;
    using ov::frontend::tensorflow::StructPack;

    std::cerr << "[ LIST ] decompose_list_construct\n";

    // Take the first tensor (if any) element type as a final list tensor element type
    element::Type element_type = element::dynamic;
    size_t element_rank = 0;
    auto shape_type = element::i32;
    if(!inputs.empty()) {
        element_type = inputs[0].get_element_type();
        element_rank = inputs[0].get_partial_shape().rank().get_length();
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
    #endif
}

Output<Node> decompose_tensor_to_list (
    Output<Node> tensor,
    element::Type shape_type
) {
    using namespace opset10;
    using ov::frontend::tensorflow::StructPack;

    Diagnostics _diag("decompose_tensor_to_list");

    // Take the first tensor (if any) element type as a final list tensor element type
    element::Type element_type = tensor.get_element_type();
    size_t element_rank = tensor.get_partial_shape().rank().get_length();

    // Slice input tensor along 0th dimension. Each slice is a new list item placed in order.
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
        PartialShape::dynamic());
}


}
}
