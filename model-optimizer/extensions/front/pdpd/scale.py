"""
 Copyright (C) 2018-2020 Intel Corporation

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

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ScaleFrontExtractor(FrontExtractorOp):
    # Affine operation will be transformed to ImageScalar and further will be converted to Mul->Add seq
    op = 'scale'
    enabled = True

    @classmethod
    def extract(cls, node):
        scale = node.pb['op'].attr('scale')
        bias = node.pb['op'].attr('bias')

        node['scale'] = scale
        node['bias'] = bias
        node['op'] = 'ImageScaler'

        return cls.enabled

'''
from extensions.ops.elementwise import Add
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.scale_shift import ScaleShiftOp

class ScaleToMulAdd(FrontReplacementOp):
    """
    Replaces Scale layer with ScaleShift and Add.
    """
    op = "scale"
    enabled = False

    def replace_op(self, graph: Graph, node: Node):
        scale = node.pb.attr('scale')
        bias = node.pb.attr('bias')

        #TODO
        #ss = ScaleShiftOp(graph, {'name': node.id + "/ScaleShift_", 'axis': 0})
        #scale_shift = ss.create_node(inputs=[in_node_1, in_node_0])

        #el = Add(graph, {'name': node.id + "/Add_"})
        #el_node = el.create_node(inputs=[scale_shift, in_node_2])

        #return [el_node.id]
        return [node.id]
'''