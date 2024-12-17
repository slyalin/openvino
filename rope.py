import openvino as ov
from openvino.runtime import op, opset1, opset8
from openvino_devtools.ov2py import ov2py

import numpy as np


def build_model(bs=2, seq_len=7, num_head=32, head_dim=128, max_position_embeddings=2048, dtype=ov.Type.f16):
    input_shape = [bs, seq_len, num_head, head_dim]
    output_shape = [bs, num_head, seq_len, head_dim]
    cache_shape = [1, 1, max_position_embeddings, head_dim]
    input = op.Parameter(dtype, ov.Shape(input_shape))
    pos_id_end = op.Parameter(ov.Type.i64, ov.Shape())
    pos_ids = op.Parameter(ov.Type.i64, ov.Shape([1, seq_len]))
    cos_cache = op.Parameter(dtype, ov.Shape(cache_shape))
    sin_cache = op.Parameter(dtype, ov.Shape(cache_shape))

    def apply(input, cache):
        cache = opset8.slice(cache, [0, 0, 0, 0], [1, 1, seq_len, head_dim], [1, 1, 1, 1])
        cache = opset1.reshape(cache, [1, seq_len, head_dim], special_zero=False)
        cache = opset1.reshape(cache, [1, 1, seq_len, head_dim], special_zero=False)
        cache = opset1.broadcast(cache, output_shape, [0, 1])
        return opset1.multiply(input, cache)

    transposed_input = opset1.transpose(input, [0, 2, 1, 3])
    apply_cos = apply(transposed_input, cos_cache)

    half_head_dim = head_dim // 2
    half_head_dim_shape = [bs, num_head, seq_len, half_head_dim]
    transposed_input_first_half = opset8.slice(transposed_input, [0, 0, 0, 0], half_head_dim_shape, [1, 1, 1, 1])
    transposed_input_second_half = opset8.slice(transposed_input, [0, 0, 0, half_head_dim], output_shape,
                                                [1, 1, 1, 1])
    minus1 = op.Constant(dtype, ov.Shape(half_head_dim_shape), [-1.0])
    transposed_input_second_half = opset1.multiply(transposed_input_second_half, minus1)
    transformed_input = opset1.concat([transposed_input_second_half, transposed_input_first_half], axis=-1)

    apply_sin = apply(transformed_input, sin_cache)

    result = opset1.add(apply_cos, apply_sin)
    return ov.Model(result, [input, pos_id_end, pos_ids, cos_cache, sin_cache], 'RoPE')


if __name__ == "__main__":
    model = build_model()
    print(ov2py(model))

    core = ov.Core()
    compiled_model = core.compile_model(model, "GPU")
    infer_req = compiled_model.create_infer_request()

    input = np.full((2, 7, 32, 128), 3.0, dtype=np.float16)
    pos_id_end = np.array(1, dtype=np.int64)
    pos_ids = np.full((1, 7), 1, dtype=np.int64)
    cos_cache = np.full((1, 1, 2048, 128), 3.0, dtype=np.float16)
    sin_cache = np.full((1, 1, 2048, 128), 2.0, dtype=np.float16)

    infer_req.set_input_tensor(0, ov.Tensor(input))
    infer_req.set_input_tensor(1, ov.Tensor(pos_id_end))
    infer_req.set_input_tensor(2, ov.Tensor(pos_ids))
    infer_req.set_input_tensor(3, ov.Tensor(cos_cache))
    infer_req.set_input_tensor(4, ov.Tensor(sin_cache))

    infer_req.infer()
    output = infer_req.get_output_tensor(0)

    print("Output:\n", output.data)
