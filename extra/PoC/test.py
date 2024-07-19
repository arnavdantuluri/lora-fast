import torch

F = torch.nn.functional
aten = torch.ops.aten
import torch
from functorch.compile import aot_function
import torch.nn.functional as F
from hgemm import CublasLinear
from torch.profiler import profile, record_function, ProfilerActivity
from gpt2 import matmul

hgemm_ = CublasLinear.apply

def matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, weight.T)

x = torch.rand([1024, 2048], device="cuda", requires_grad=True)
w = torch.rand([1024, 2048], device="cuda", requires_grad=True)

fc = torch.nn.Linear(2048, 1024, False, "cuda")
w = fc.weight

# def compiler_fn(fx_module: torch.fx.GraphModule, _):
#     print(fx_module.code)
#     return fx_module

# aot_pytorch = aot_function(matmul, fw_compiler=compiler_fn, bw_compiler=compiler_fn)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
out = fc(x)
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))


start.record()
out_cu = matmul(x, w)
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))

# dout = torch.empty_like(out)
# out.backward(dout)

# x_grad_pt = x.grad.clone()
# w_grad_pt = w.grad.clone()

# x.grad, w.grad = None, None
# out_cu.backward(dout)
# x_grad_cu = x.grad.clone()
# w_grad_cu = w.grad.clone()

# x.grad, w.grad = None, None

# assert torch.all(torch.abs(out_cu - out) / out) < 0.99, "Not within 99 percent error"
# assert torch.all(torch.nan_to_num(torch.abs(x_grad_cu - x_grad_pt) / x_grad_pt)) < 0.99, "Not within 99 percent error"
# assert torch.all(torch.nan_to_num(torch.abs(w_grad_cu - w_grad_pt) / w_grad_pt)) < 0.99, "Not within 99 percent error"