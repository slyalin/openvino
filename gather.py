import torch
import openvino as ov
from openvino_devtools.ov2py import ov2py

class My(torch.nn.Module):
    def forward(self, x):
        d = x[0]
        return d


input = torch.randn(16, dtype=torch.float32)

ov_model = ov.convert_model(My(), example_input=(input,))
print(ov2py(ov_model))
compiled_model = ov.compile_model(ov_model, "GPU")
result = compiled_model([input])[0]
print("OV result: ", result)

print("Expected result: ", My()(input))
