import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import FunctionCtx
import torch.nn.functional as F
from axolotl.llama_ffn.triton_backward import grad_input_kernel_wrapper
from axolotl.llama_ffn.triton_forward import kernel_ff as tri_kernel_ff

def ff_pytorch(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    a = F.silu(torch.matmul(x, w1)) * torch.matmul(x, w3)
    return a

class llama_ffn(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, x, w1, w3):
        dtype = x.dtype
        batch, seq_len, dim = x.shape
        M, K = batch * seq_len, dim

        x, w1, w3 = x.to(torch.float16), w1.to(torch.float16), w3.to(torch.float16)
        x, w1, w3 = x.reshape(M, K), w1.reshape(K, -1), w3.reshape(K, -1)

        out, act_in, mm_1 = tri_kernel_ff(x, w1, w3)
        ctx.save_for_backward(x, w1, w3, act_in, mm_1)
        return out.reshape(batch, seq_len, -1).to(dtype)

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs):
        grad_output = grad_outputs[0]
        dtype = grad_output.dtype
        grad_output = grad_output.to(torch.float16)

        batch, seq_len, dim = grad_output.shape
        M, K = batch * seq_len, dim
        grad_output = grad_output.reshape(M, K)

        x, w1, w3, act_in, mm_1 = ctx.saved_tensors

        grad_input, w1_grad, w3_grad = grad_input_kernel_wrapper(grad_output, x, w1, w3, act_in, mm_1)
        return grad_input.reshape(batch, seq_len, -1).to(dtype), w1_grad.to(dtype).t(), w3_grad.to(dtype).t()

kernel_ff = llama_ffn.apply

if __name__ == "__main__":
    x = (torch.rand([2048, 4096], dtype=torch.float16, device="cuda")).requires_grad_(True)
    w1 = (torch.rand([4096, 2048], dtype=torch.float16, device="cuda") * 0.02).requires_grad_(True)
    w3 = (torch.rand([4096, 2048], dtype=torch.float16, device="cuda")* 0.02).requires_grad_(True)

    output_pytorch = ff_pytorch(x, w1, w3)

    dout = torch.rand_like(output_pytorch)
    output_pytorch.backward(dout)

    x_pt_grad, x.grad = x.grad.clone(), None

    output_triton = kernel_ff(x, w1, w3)
    output_triton.backward(dout)

    x_tri_grad, x.grad = x.grad.clone(), None

    x_pt_grad = torch.nan_to_num(x_pt_grad, 0)
    x_tri_grad = torch.nan_to_num(x_tri_grad, 0)

    assert torch.all(torch.abs(output_triton - output_pytorch) / output_pytorch) < 0.99, "Not within 99 percent error"
    assert torch.all(torch.abs(x_tri_grad - x_pt_grad) / x_pt_grad) < 0.99, "Not within 99 percent error"