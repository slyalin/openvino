import torch
import openvino as ov
from openvino_devtools.ov2py import ov2py
from torch import onnx

class My(torch.nn.Module):
    def forward(self, x):
        d = torch.transpose(x, 0, 1) 
        return d

input = torch.randn((4, 4), dtype=torch.float32)

model = ov.convert_model(My(), example_input=(input,))
print(ov2py(model))
compiled_model = ov.compile_model(model, "CPU")
result = compiled_model([input])[0]
print("Input: ", input)
print("OV result: ", result)

print("Expected result: ", My()(input))
