"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging as log

import os
import numpy as np

import paddle
import paddle.fluid as fluid
#from mo.front.pdpd.loader import load_pdpd_model, protobuf2nx

from extensions.load.loader import Loader
from mo.front.common.register_custom_ops import update_extractors_with_extensions, check_for_duplicates
from mo.front.extractor import extract_node_attrs
from mo.front.pdpd.extractor import pdpd_op_extractor, pdpd_op_extractors
from mo.graph.graph import Graph
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg

def protobuf2nx(graph, program, feed_var_names, fetch_vars):

    parameters_dict = {}
    variables = program.global_block().vars

    ### collect weight node
    for name in variables:
        var = program.global_block().var(name)
        print("Process weight node", name)
        if name.endswith('feed') or name.endswith('fetch'):
            continue
        if not var.persistable:
            continue
        parameters_dict[name] = {
            'data': np.array(fluid.global_scope().var(name).get_tensor()),
            'dtype': var.dtype,
            'shape': var.shape
        }

    block = program.global_block()

    ## intermediate data tensor
    data_nodes_map = {}

    ## add input nodes
    for ipt in feed_var_names:
        layer_name = ipt
        var = block.var(ipt)
        attrs = {}
        attrs['shape'] = var.shape
        attrs['dtype'] = var.dtype
        graph.add_node(layer_name, kind='op', op='Parameter', pb=block)
        data_nodes_map[layer_name] = (layer_name, 0)

    ## build graph
    ### add weigth node
    for name, param in parameters_dict.items():
        graph.add_node(name, kind='op', op='Const', pb_init=param)
        data_nodes_map[name] = (name, 0)


    op_type_count = dict()
    ### add op nodes
    for block in program.blocks:
        for i, op in enumerate(block.ops):
            if op.type in ['feed', 'fetch']:
                continue
            else:
                if op.type in op_type_count:
                    op_type_count[op.type] += 1
                else:
                    op_type_count[op.type] = 1
                ## Get Input/Output
                layer_name = op.type + '_' + str(op_type_count[op.type] - 1)
                print("Process op node ",layer_name)
                inputs = {}
                outputs = {}
                graph.add_node(layer_name, kind='op', pb=op)
                input_port = 0
                for ipt_type in op.input_names:
                    input_names = op.input(ipt_type)
                    if not input_names :
                        continue
                    for input_name in input_names:
                        if input_name not in data_nodes_map:
                            print("Missing node ",input_name)
                        src_id, src_port = data_nodes_map[input_name]
                        assert(graph.has_node(src_id))
                        edge_attrs = {
                            'out' : src_port,
                            'in' : input_port,
                            'name' : input_name,
                            'fw_tensor_debug_info' : [(input_name, input_name)],
                            'in_attrs' : ['in', 'name'],
                            'out_attrs' : ['out', 'name'],
                            'data_attrs' : ['fw_tensor_debug_info']
                        }
                        graph.add_edge(src_id, layer_name, **edge_attrs)

                        input_port += 1
                
                for src_port, out_name in enumerate(op.output_names):
                
                    output_nodes = op.output(out_name)
                    for out_node_name in output_nodes:
                        if out_node_name in data_nodes_map:
                            print("Detected reuse of blob {}.".format(out_node_name))    
                        data_nodes_map[out_node_name] = (layer_name, src_port)
                
    ### add output nodes
    for port, opt in enumerate(fetch_vars):
        layer_name = opt.name
        var = block.var(layer_name)
        attrs = {}
        attrs['shape'] = var.shape
        attrs['dtype'] = var.dtype
        graph.add_node(layer_name, kind='op', op='Result', pb=block)
        src_id, src_port = data_nodes_map[layer_name]
        assert(graph.has_node(src_id))
        edge_attrs = {
            'out' : src_port,
            'in' : port,
            'name' : src_id,
            'fw_tensor_debug_info' : [(src_id, src_id)],
            'in_attrs' : ['in', 'name'],
            'out_attrs' : ['out', 'name'],
            'data_attrs' : ['fw_tensor_debug_info']
        }
        graph.add_edge(src_id, layer_name, **edge_attrs)

def load_pdpd_model(proto_path: str, model_path: [str, None] = None):
    paddle.enable_static()
    exe = fluid.Executor(fluid.CPUPlace())

    dir_path = os.path.dirname(os.path.realpath(model_path))
    model_name = os.path.basename(model_path)
    model_name = os.path.splitext(model_name)[0]
    params_name = os.path.splitext(model_name)[0]
    [program, feed_var_names, fetch_vars] = paddle.static.load_inference_model(dir_path, exe, model_filename=model_name, params_filename=params_name)

    return [program, feed_var_names, fetch_vars]

class PDPDLoader(Loader):
    enabled = True

    def load(self, graph: Graph):
        argv = graph.graph['cmd_params']
        [program, feed_var_names, fetch_vars] = load_pdpd_model(argv.input_proto, argv.input_model)
        # model_graph = model_proto.graph  # pylint: disable=no-member
        # print(model_graph)
        # assert len(model_graph) == 1, "An ONNX model contains more than 1 graph: unsupported"
        # log.debug("Number of nodes in graph_def: {}".format(len(model_graph.node)))
        # log.debug("Number of all input ports (not true inputs) in graph_def: {}".format(len(feed_var_names)))
        # log.debug("Number of initializers in graph_def: {}".format(len(model_graph.initializer)))
        # log.debug(
        #     "Number of real inputs in graph_def: {}".format(len(model_graph.input) - len(model_graph.initializer)))
        update_extractors_with_extensions(pdpd_op_extractors)

        try:
            protobuf2nx(graph, program, feed_var_names, fetch_vars)
        except Exception as e:
            raise Error(
                'Cannot pre-process ONNX graph after reading from model file "{}". ' \
                'File is corrupt or has unsupported format. Details: {}. ' +
                refer_to_faq_msg(44),
                argv.input_model,
                str(e)
            ) from e
        log.debug("Number of nodes in NX graph: {}".format(graph.number_of_nodes()))

        graph.__setattr__('name',
                          argv.model_name if argv.model_name else 'model_proto.graph.name')  # pylint: disable=no-member
        graph.graph['layout'] = 'NCHW'
        graph.graph['fw'] = 'pdpd'
        graph.graph['feature_dim'] = 1

        graph.check_empty_graph('protobuf2nx. It may happen due to problems with loaded model')
        extract_node_attrs(graph, lambda node: pdpd_op_extractor(node, check_for_duplicates(pdpd_op_extractors)))
