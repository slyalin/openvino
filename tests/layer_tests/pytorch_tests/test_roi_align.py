# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

from typing import List


class TestRoiAlign(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.rand(*self.input_shape).astype(np.float32), self.boxes)

    def create_model(self, output_size, spatial_scale, sampling_ratio, aligned):
        
        import torch
        import torchvision.ops as ops

        class torchvision_roi_align(torch.nn.Module):
            def __init__(self, output_size, spatial_scale, sampling_ratio, aligned):
                super(torchvision_roi_align, self).__init__()
                self.output_size = output_size
                self.spatial_scale = spatial_scale
                self.sampling_ratio = sampling_ratio
                self.aligned = aligned

            def forward(self, input_tensor, boxes: List[torch.Tensor]):
                return ops.roi_align(input_tensor, boxes, self.output_size)
                # return torchvision.ops.roi_align(input_tensor,
                #                              rois=boxes,
                #                              output_size=self.output_size,
                #                              spatial_scale=self.spatial_scale,
                #                              sampling_ratio=self.sampling_ratio,
                #                              aligned=self.aligned)

        ref_net = None

        return torchvision_roi_align(output_size, spatial_scale, sampling_ratio, aligned), ref_net, "torchvision::roi_align"

    @pytest.mark.parametrize(("input_shape"), [
        (1, 1, 3, 10),
    ])
    @pytest.mark.parametrize(("boxes"), [
        # np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).astype(np.float32)
        np.array([0, 1, 2, 3, 4]).astype(np.float32)
    ])
    @pytest.mark.parametrize('output_size', ([
        (7, 7),
    ]))
    @pytest.mark.parametrize('spatial_scale', ([
        (2.0),
    ]))
    @pytest.mark.parametrize('sampling_ratio', ([
        (2),
    ]))
    @pytest.mark.parametrize('aligned', ([
        True,
    ]))


    @pytest.mark.nightly
    def test_roi_align(self, input_shape, boxes, output_size, spatial_scale, sampling_ratio, aligned, ie_device, precision, ir_version):
        self.input_shape = input_shape
        self.boxes = boxes
        if ie_device == "CPU":
            self._test(*self.create_model(output_size, spatial_scale, sampling_ratio, aligned), ie_device, precision, ir_version)

