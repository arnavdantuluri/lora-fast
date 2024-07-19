import torch
from functorch.compile import aot_function
import torch.nn.functional as F

def ff_pytorch(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    a = F.silu(torch.matmul(x, w1)) * torch.matmul(x, w2)
    return a

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return fx_module

x = (torch.rand([2048, 4096], dtype=torch.float16, device="cuda")).requires_grad_(True)
w1 = (torch.rand([4096, 4096], dtype=torch.float16, device="cuda") * 0.02).requires_grad_(True)
w2 = (torch.rand([4096, 4096], dtype=torch.float16, device="cuda")* 0.02).requires_grad_(True)
w3 = (torch.rand([4096, 2048], dtype=torch.float16, device="cuda")* 0.02).requires_grad_(True)

aot_pytorch = aot_function(ff_pytorch, fw_compiler=compiler_fn, bw_compiler=compiler_fn)
out = aot_pytorch(x, w1, w2, w3)
dout = torch.empty_like(out)
out.backward(dout)