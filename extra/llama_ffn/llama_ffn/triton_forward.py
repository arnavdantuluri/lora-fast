import torch

import triton
import triton.language as tl
from triton import Config, autotune
from triton.ops.matmul import get_configs_io_bound
# from triton.ops.matmul_perf_model import estimate_matmul_time, early_config_prune

# Need to save acc1 (act_in) and acc2 (mm_1) for backward pass
# act_out is recomputed in backward pass to save memory
@autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=4),

        # good for int8
        Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
    ], #+ get_configs_io_bound(),
    key=['M', 'N', 'K'],
    # prune_configs_by={
    #     'early_config_prune': early_config_prune,
    #     'perf_model': estimate_matmul_time,
    #     'top_k': 10,
    # },
)
@triton.jit
def ff_llama(
    a_ptr, w1_ptr, w3_ptr, 
    out_ptr, act_in_ptr, mm_1_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_outm, stride_outn,
    stride_inm, stride_inn,
    stride_mmm, stride_mmn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """
    w1 and w3 are weights (linear layers)
    F.silu(w1(x)) * w3(x)
    """
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(w1_ptrs)
        acc1 += tl.dot(a, b)

        c = tl.load(w3_ptrs)
        acc2 += tl.dot(a, c)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k

    accumulator = (acc1 * tl.sigmoid(acc1)) * acc2

    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :])
    act_in_ptrs = act_in_ptr + (stride_inm * offs_outm[:, None] + stride_inn * offs_outn[None, :])
    mm_1_ptrs = mm_1_ptr + (stride_mmm * offs_outm[:, None] + stride_mmn * offs_outn[None, :])

    out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)

    tl.store(out_ptrs, accumulator, mask=out_mask)
    tl.store(act_in_ptrs, acc1, mask=out_mask)
    tl.store(mm_1_ptrs, acc2, mask=out_mask)

def kernel_ff(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor,) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert w1.dtype == w3.dtype
    assert w1.dtype
    assert w1.shape == w3.shape

    M, K = x.shape

    N = w1.shape[1]
    assert K == w1.shape[0]
    assert w1.shape == w3.shape
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    act_in = torch.empty((M, N), dtype=x.dtype, device=x.device)
    mm_1 = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = lambda META: (triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"]),)
    # a_ptr, w1_ptr, w3_ptr, 
    # out_ptr, act_in_ptr, mm_1_ptr, act_out_ptr,
    # M, N, K,
    # stride_am, stride_ak,
    # stride_w1k, stride_w1n,
    # stride_w3k, stride_w3n,
    # stride_outm, stride_outn,
    # stride_inm, stride_inn,
    # stride_mmm, stride_mmn,
    # stride_act_outm, stride_act_outn,
    ff_llama[grid](
        x, w1, w3, 
        out, act_in, mm_1,
        M, N, K,
        *x.stride(),
        *w1.stride(),
        *w3.stride(),
        *out.stride(),
        *act_in.stride(),
        *mm_1.stride(),
    )
    return out, act_in, mm_1


x = torch.randn([1, 16, 4096], dtype=torch.float16, device="cuda")
w1_w = torch.randn([11008, 4096], dtype=torch.float16, device="cuda") * 0.2
w3_w = torch.randn([11008, 4096], dtype=torch.float16, device="cuda") * 0.2

def ff_pytorch(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    act_in = torch.matmul(x, w1.t())
    a = torch.nn.functional.silu(act_in)
    b = torch.matmul(x, w3.t())
    return a * b, act_in, b, a

if __name__ == "__main__":
    output_tri, act_in_tri, mm_1_tri = kernel_ff(x=x, w1=w1_w, w3=w3_w)
    output_pt, act_in_pt, mm_1_pt, act_out_pt = ff_pytorch(x=x, w1=w1_w, w3=w3_w)

    assert torch.allclose(output_tri, output_pt, atol=1e-1), f"max diff: {torch.max(torch.abs(output_tri - output_pt))}"
    assert torch.allclose(act_in_tri, act_in_pt, atol=1e-1), f"max diff: {torch.max(torch.abs(act_in_tri - act_in_pt))}"
    assert torch.allclose(mm_1_tri, mm_1_pt, atol=1e-1), f"max diff: {torch.max(torch.abs(mm_1_tri - mm_1_pt))}"

    print("rms matmul silu mul triton", triton.testing.do_bench(lambda: kernel_ff(x=x, w1=w1_w, w3=w3_w)))
    print("rms matmul silu mul pytorch", triton.testing.do_bench(lambda: ff_pytorch(x=x, w1=w1_w, w3=w3_w)))