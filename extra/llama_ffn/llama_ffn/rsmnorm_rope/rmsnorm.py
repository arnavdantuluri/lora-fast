# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Credit to unsloth team :)
#Modified from https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rms_layernorm.py
import triton
import triton.language as tl
import torch

MAX_FUSED_SIZE = 65536
next_power_of_2 = triton.next_power_of_2

def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: 
        num_warps = 32
    elif BLOCK_SIZE >=  8192: 
        num_warps = 16
    elif BLOCK_SIZE >=  2048: 
        num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.jit
def _rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr
):
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)#.to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    inv_var = 1.0 / tl.sqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype) # Exact copy from HF
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask = mask)

# Modified to compute gradient of the weight which is grad_output * normed 
@triton.jit
def _rms_layernorm_backward(
    dY, dY_row_stride,
    X,   X_row_stride,
    W,   W_row_stride,
    r,   r_row_stride,
    dW, dW_row_stride,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr,
):
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    dW += row_idx * dW_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx *  r_row_stride

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask = mask, other = 0).to(tl.float32)

    # Get saved row variance
    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    w_grad = dY_row * normed

    dY_W = dY_row * W_row
    rowsum_dY_normed = tl.sum(dY_W * normed, axis = 0)
    output = inv_var/n_cols * (n_cols*dY_W - normed*rowsum_dY_normed)
    tl.store(dY + col_offsets, output, mask = mask)
    tl.store(dW + col_offsets, w_grad, mask=mask)


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = "cuda")
        r = torch.empty(n_rows, dtype = torch.float32, device = "cuda")

        _rms_layernorm_forward[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            n_cols, eps,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        dW = torch.empty_like(dY)
        _rms_layernorm_backward[(n_rows,)](
            dY, dY.stride(0),
            X,  X .stride(0),
            W,  W .stride(0),
            r,  r .stride(0),
            dW, dW.stride(0),
            n_cols, ctx.eps,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dX = dY.view(*shape)
        dW = dW.view(*shape)
        return dX, dW, None


def fast_rms_layernorm(X, W, eps):
    out = Fast_RMS_Layernorm.apply(X, W, eps)
    return out

def rms_norm_pytorch(x: torch.Tensor, rms_w: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms_w

if __name__ == "__main__":
    x = torch.rand()
    w = torch.rand()

    triton_out = fast_rms_layernorm(x, w, 1e-6)
    pytorch_out = rms_norm_pytorch(x, w, 1e-6)

    d_out = torch.empty_like(pytorch_out)

    triton_out.backward(d_out)
    x_tri_grad, x.grad = x.grad.clone(), None
    w_tri_grad, w.grad = w.grad.clone(), None

    pytorch_out.backward(d_out)
    x_pt_grad, x.grad = x.grad.clone(), None
    w_pt_grad, w.grad = w.grad.clone(), None

    assert torch.all(torch.abs(triton_out - pytorch_out) / pytorch_out) < 0.99, "Not within 99 percent error"
    assert torch.all(torch.abs(x_tri_grad - x_pt_grad) / x_pt_grad) < 0.99, "Not within 99 percent error"
    assert torch.all(torch.abs(w_tri_grad - w_pt_grad) / x_pt_grad) < 0.99, "Not within 99 percent error"