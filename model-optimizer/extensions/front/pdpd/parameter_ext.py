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
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from extensions.ops.parameter import Parameter
from mo.front.extractor import FrontExtractorOp


class PlaceholderFrontExtractor(FrontExtractorOp):
    op = 'Parameter'
    enabled = True

    @classmethod
    def extract(cls, node):
        block = node.pb
        var = block.var(node.id)
        t_type = var.dtype
        t_shape = var.shape  #tuple
        attrs = {
            'shape': np.array([t_shape[d] for d in range(len(t_shape))], dtype=np.int64),
            'data_type': np.float32 #TODO cast TENSOR_TYPE_TO_NP_TYPE[t_type]
        }
        Parameter.update_node_stat(node, attrs)
        return cls.enabled
