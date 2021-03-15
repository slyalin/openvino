import numpy as np
import os

import paddle
import paddle.fluid as fluid
from mo.graph.graph import Graph
pdpd_io_dict = {
    'conv2d': {
        'inputs': ['Input','Filter'], 
        'outputs':['Output']
    },
    'pool2d': {
        'inputs' : ['X'], 
        'outputs' : ['Out']
    },
    'batch_norm':{
        'inputs' : ['X','Scale', 'Bias', 'Mean', 'Variance'],
        'outputs' : ['Y']
    },
    'relu': {
        'inputs' : ['X'],
        'outputs' : ['Out']
    },
    'reshape2' : {
        'inputs' : ['X'],
        'outputs' : ['Out']
    },
    'matmul' : {
        'inputs' : ['X', 'Y'],
        'outputs' : ['Out']
    },
    'elementwise_add' : {
        'inputs' : ['X', 'Y'],
        'outputs' : ['Out']
    },
    'scale' : {
        'inputs' : ['X'],
        'outputs' : ['Out']
    },
    'softmax' : {
        'inputs' : ['X'],
        'outputs' : ['Out']
    }
}

def get_io_ports(op_type):
    return pdpd_io_dict[op_type]

def protobuf2nx(graph, program, feed_var_names, fetch_vars):

    parameters_dict = {}
    variables = program.global_block().vars

    ### collect weight node
    for name in variables:
        var = program.global_block().var(name)
        # print("Process weight node", name)
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
                # print("Process op node ",layer_name)
                graph.add_node(layer_name, kind='op', pb={'op': op, 'layer_name': layer_name, 'block': block})
                io_names = get_io_ports(op.type)
                for input_port, input_type in enumerate(io_names['inputs']):
                    input_names = op.input(input_type)
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

                for src_port, out_name in enumerate(io_names['outputs']):
                
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
    
    