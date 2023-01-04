# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestLog(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 10).astype(np.float32),)

    def create_model(self, op):
        import torch

        ops = {
            "log": torch.log,
            "log_": torch.log_,
            "log2": torch.log2,
            "log2_": torch.log2_
        }

        op_fn = ops[op]

        class aten_log(torch.nn.Module):
            def __init__(self, op):
                super(aten_log, self).__init__()
                self.op = op

            def forward(self, x):
                return self.op(x)

        ref_net = None

        return aten_log(op_fn), ref_net, f"aten::{op}"

    @pytest.mark.nightly
    @pytest.mark.parametrize("op", ["log", "log_", "log2", "log2_"])
    def test_log(self, op, ie_device, precision, ir_version):
        self._test(*self.create_model(op), ie_device, precision, ir_version)
