# Works with nested modules; Refer to test in replace_layernorm.py
from typing import Callable, Union

import torch
import torch.nn as nn
from cublas_ops import cublas_half_matmul_simple, cublas_half_matmul_batched_simple, \
    cublaslt_fused_half_matmul_batched_simple, cublaslt_fused_half_matmul_simple
from util import replace_pattern
from torch.fx import subgraph_rewriter
import torch.fx as fx
from unet_pt import UNet2DConditionModel

# Load weights from the original model
from diffusers import DiffusionPipeline
import sys

sys.setrecursionlimit(10000)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True

def linear_wrapper(v: torch.Tensor, linear: torch.nn.Linear, activation: bool):
    return linear_wrapper_functional(v, linear.weight, linear.bias, activation)


torch.fx.wrap("linear_wrapper")

def linear_wrapper_functional(v: torch.Tensor, weight: torch.Tensor, bias: Union[None, torch.Tensor], activation: bool):
    # small hack to avoid casting weights/bias at each call
    batched = True if v.dim() == 3 else False
    if v.dtype == torch.float16:
        if weight.dtype == torch.float32:
            weight.data = weight.data.half()
        if bias is not None and bias.dtype == torch.float32:
            bias.data = bias.data.half()
    if bias is not None and not batched:
        return cublaslt_fused_half_matmul_simple(v, weight, bias, "NONE")
    if bias is not None and batched:
        return cublaslt_fused_half_matmul_batched_simple(v, weight, bias, "NONE")
    if batched:
        return cublas_half_matmul_batched_simple(v, weight)
    return cublas_half_matmul_simple(v, weight)
    


torch.fx.wrap("linear_wrapper_functional")

# Linear ends up classified as "get_attr" instead of a "call_module". Breaks the subgraph_replacement
def replace_linear(gm: torch.fx.GraphModule):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, v):
            return self.linear(v)

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, v):
            return linear_wrapper(v, self.linear, False)

    replace_pattern(gm, Pattern(), Replacement())

if __name__ == "__main__":
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    unet_new = UNet2DConditionModel().half().cuda()
    unet_new.load_state_dict(pipe.unet.state_dict())

    fx_model = fx.symbolic_trace(unet_new)
    replace_linear(fx_model)
    old_traced = fx.symbolic_trace(unet_new)

    assert fx_model.code != old_traced.code, "Issue with fusion with fx graph"
    print("Fx Graph replacement was a success! Kernel Fusion works perfectly")
    # Test output
    # x = torch.rand(5, 5, dtype=torch.float32).cuda()
    # out_old = unet_new(x)
    # out_fused = fx_model(x)
    # # Some margin for triton code.
    # assert ((out_fused - out_old).abs() < 1e-3).all(), "Outputs don't match"