import torch
import openvino as ov
from openvino_devtools.ov2py import ov2py
from torch import onnx

class My(torch.nn.Module):
    def forward(self, x):
        d = torch.unsqueeze(x, 1) 
        return d

input = torch.randn((4, 4), dtype=torch.float32)

# onnx.export(My(), (input,), 'model.onnx')
# core = ov.Core()
# model = core.read_model('model.onnx')
# print(model)
model = ov.convert_model(My(), example_input=(input,))
print(ov2py(model))
compiled_model = ov.compile_model(model, "CPU")
result = compiled_model([input])[0]
print("OV result: ", result)

print("Expected result: ", My()(input))
