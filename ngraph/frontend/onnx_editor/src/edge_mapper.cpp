// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <onnx/onnx_pb.h>

#include "ngraph/except.hpp"
#include "onnx_editor/edge_mapper.hpp"

using namespace ngraph;
using namespace ngraph::onnx_editor;

onnx_editor::EdgeMapper::EdgeMapper(const ONNX_NAMESPACE::GraphProto& graph_proto)
{
    update(graph_proto);
}

void onnx_editor::EdgeMapper::update(const ONNX_NAMESPACE::GraphProto& graph_proto)
{
    // reset state
    m_node_inputs.clear();
    m_node_outputs.clear();
    m_node_name_to_index.clear();

    int topological_index = 0;
    m_node_inputs.resize(graph_proto.node().size());
    m_node_outputs.resize(graph_proto.node().size());
    for (const auto& node_proto : graph_proto.node())
    {
        for (const auto& out_name : node_proto.output())
        {
            // node output name is unique
            m_node_name_to_index.emplace(out_name, topological_index);
            m_node_outputs[topological_index].push_back(out_name);
        }
        for (const auto& in_name : node_proto.input())
        {
            m_node_inputs[topological_index].push_back(in_name);
        }
        if (!node_proto.name().empty())
        {
            // node name can identify node, but it can be ambiguous
            m_node_name_to_index.emplace(node_proto.name(), topological_index);
        }
        ++topological_index;
    }
}

std::vector<int> onnx_editor::EdgeMapper::find_node_indexes(const std::string& node_name,
                                                            const std::string& output_name) const
{
    if (!output_name.empty())
    {
        const auto& index_iter = m_node_name_to_index.find(output_name);
        if (index_iter != std::end(m_node_name_to_index))
        {
            return std::vector<int>{index_iter->second};
        }
    }
    if (!node_name.empty())
    {
        const auto matched_nodes_range = m_node_name_to_index.equal_range(node_name);
        std::vector<int> result;
        std::transform(matched_nodes_range.first,
                       matched_nodes_range.second,
                       std::back_inserter(result),
                       [](const std::pair<std::string, int>& iter) { return iter.second; });
        if (!result.empty())
        {
            return result;
        }
    }
    throw ngraph_error("Node with name: " + (node_name.empty() ? "not_given" : node_name) +
                       " and output_name: " + (output_name.empty() ? "not_given" : output_name) +
                       " was not found");
};

std::string onnx_editor::EdgeMapper::get_node_output_name(int node_index, int output_index) const
{
    if (node_index >= m_node_outputs.size())
    {
        throw ngraph_error("Node with index: " + std::to_string(node_index) +
                           "is out of scope outputs list");
    }
    if (output_index >= m_node_outputs[node_index].size())
    {
        throw ngraph_error("Node with index: " + std::to_string(node_index) +
                           " has not output with index: " + std::to_string(output_index));
    }
    const auto output_name = m_node_outputs[node_index][output_index];
    return output_name;
}

std::string onnx_editor::EdgeMapper::get_node_input_name(int node_index, int input_index) const
{
    if (node_index >= m_node_inputs.size())
    {
        throw ngraph_error("Node with index: " + std::to_string(node_index) +
                           "is out of scope inputs list");
    }
    if (input_index >= m_node_inputs[node_index].size())
    {
        throw ngraph_error("Node with index: " + std::to_string(node_index) +
                           " has not input with index: " + std::to_string(input_index));
    }
    const auto input_name = m_node_inputs[node_index][input_index];
    return input_name;
}

InputEdge onnx_editor::EdgeMapper::find_input_edge(const EditorNode& node,
                                                   const EditorInput& in) const
{
    // identification can be both based on node name and output name
    const auto& node_indexes = find_node_indexes(node.m_node_name, node.m_output_name);
    int node_index = -1;
    if (node_indexes.size() == 1)
    {
        node_index = node_indexes[0];
    }
    else if (!in.m_input_name
                  .empty()) // input indexes are not deterministic if a node name is ambiguous
    {
        // many nodes with the same name
        // check if some of found index matches input name
        int matched_inputs_number = 0;
        for (const auto& index : node_indexes)
        {
            if (std::count(std::begin(m_node_inputs[index]),
                           std::end(m_node_inputs[index]),
                           in.m_input_name) > 0)
            {
                node_index = index;
                ++matched_inputs_number;
            }
        }
        if (matched_inputs_number == 0)
        {
            throw ngraph_error("Input edge described by: " + node.m_node_name +
                               " and input name: " + in.m_input_name + " was not found");
        }
        if (matched_inputs_number > 1)
        {
            throw ngraph_error("Given node name: " + node.m_node_name + " and input name: " +
                               in.m_input_name + " are ambiguous to determine input edge");
        }
    }
    else
    {
        throw ngraph_error("Given node name: " + node.m_node_name +
                           " and input index: " + std::to_string(in.m_input_index) +
                           " are ambiguous to determine input edge");
    }
    if (!in.m_input_name.empty())
    {
        return InputEdge{node_index, in.m_input_name};
    }
    else if (in.m_input_index != -1) // input index is set
    {
        const auto& input_name = get_node_input_name(node_index, in.m_input_index);
        return InputEdge{node_index, input_name};
    }
    else
    {
        throw ngraph_error("Not enough information to determine input edge");
    }
}

OutputEdge onnx_editor::EdgeMapper::find_output_edge(const EditorNode& node,
                                                     const EditorOutput& out) const
{
    // identification can be both based on node name and output name
    const auto& node_indexes = find_node_indexes(node.m_node_name, node.m_output_name);
    int node_index = -1;
    if (node_indexes.size() == 1)
    {
        node_index = node_indexes[0];
    }
    else if (!out.m_output_name
                  .empty()) // output indexes are not deterministic if a node name is ambiguous
    {
        // many nodes with the same name
        // check if some of found index matches output name
        int matched_outputs_number = 0;
        for (const auto& index : node_indexes)
        {
            if (std::count(std::begin(m_node_outputs[index]),
                           std::end(m_node_outputs[index]),
                           out.m_output_name) > 0)
            {
                node_index = index;
                ++matched_outputs_number;
            }
        }
        if (matched_outputs_number == 0)
        {
            throw ngraph_error("Output edge described by: " + node.m_node_name +
                               " and output name: " + out.m_output_name + " was not found");
        }
    }
    else
    {
        throw ngraph_error("Given node name: " + node.m_node_name +
                           " and output index: " + std::to_string(out.m_output_index) +
                           " are ambiguous to determine output edge");
    }
    if (!out.m_output_name.empty())
    {
        return OutputEdge{node_index, out.m_output_name};
    }
    else if (out.m_output_index != -1) // output index is set
    {
        const auto& output_name = get_node_output_name(node_index, out.m_output_index);
        return OutputEdge{node_index, output_name};
    }
    else
    {
        throw ngraph_error("Not enough information to determine output edge");
    }
}

OutputEdge onnx_editor::EdgeMapper::find_output_edge(const std::string& output_name) const
{
    return find_output_edge(EditorNode{EditorOutput{output_name}}, EditorOutput{output_name});
}
