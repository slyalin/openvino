import torch
import openvino as ov
from openvino_devtools.ov2py import ov2py

class My(torch.nn.Module):
    def forward(self, x, params):
        d = x[params[0] : params[1] : params[2]]
        return d


input = torch.randn(16, dtype=torch.float32)
params = torch.tensor([1, 2, 1])

# jit_model = torch.jit.trace(My(), (torch.randn(16).to(torch.float32), torch.tensor([1, 2, 1]).to(torch.int32)))
# ov_model = ov.convert_model(jit_model)
ov_model = ov.convert_model(My(), example_input=(input,params,))
print(ov2py(ov_model))
compiled_model = ov.compile_model(ov_model, "GPU")
result = compiled_model([input, params])[0]
print("OV result: ", result)

print("Expected result: ", My()(input, params))
