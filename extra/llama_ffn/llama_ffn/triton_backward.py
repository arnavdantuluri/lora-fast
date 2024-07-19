# Some meta-parameters are not necessary within the kernel but are there to align with what triton get_configs_io_bound config returns
import torch

import triton
import triton.language as tl
from triton import Config, autotune
from triton.ops.matmul import get_configs_io_bound
from triton.ops.matmul_perf_model import estimate_matmul_time, early_config_prune

import time

def map_dtype(input):
    if input == torch.float32:
        return tl.float32
    elif input == torch.float16:
        return tl.float16
    elif input == torch.bfloat16:
        return tl.bfloat16
    elif input == torch.int64:
        return tl.int64
    else:
        raise ValueError(f"Unable to convert the given input: '{input}'.")

@triton.jit
def silu(input):
    return input * tl.sigmoid(input)

@triton.jit
def silu_grad(grad_output, input):
    sigma = 1 / (1 + tl.math.fast_expf(-input.to(tl.float32)))
    grad_input = grad_output * (sigma + input * sigma * (1 - sigma))
    return grad_input

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
def backward_intermediate_kernel(
    grad_output_ptr, mm_1_ptr, act_in_ptr, 
    intermediate_1_ptr, intermediate_2_ptr,
    M, N, K,
    stride_gdm, stride_gdk,
    stride_mmm, stride_mmk,
    stride_aim, stride_aik,
    stride_i1m, stride_i1k,
    stride_i2m, stride_i2k,
    dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_k = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))

    grad_output_ptrs = grad_output_ptr + (offs_m[:, None] * stride_gdm + offs_k[None, :] * stride_gdk)
    mm_1_ptrs = mm_1_ptr + (offs_m[:, None] * stride_mmm + offs_k[None, :] * stride_mmk)
    act_in_ptrs = act_in_ptr + (offs_m[:, None] * stride_aim + offs_k[None, :] * stride_aik)

    grad_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    mm_1_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    act_in_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    grad_output = tl.load(grad_output_ptrs, mask=grad_mask)
    mm_1 = tl.load(mm_1_ptrs, mask=mm_1_mask)
    act_in = tl.load(act_in_ptrs, mask=act_in_mask)
    act_out = silu(act_in.to(tl.float32)).to(act_in.dtype)

    intermediate_1 = (grad_output * act_out).to(dtype)
    intermediate_2 = (silu_grad(grad_output * mm_1, act_in)).to(dtype)

    offs_i1m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_i1k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    offs_i2m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_i2k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    intermediate_1_ptrs = intermediate_1_ptr + stride_i1m * offs_i1m[:, None] + stride_i1k * offs_i1k[None, :]
    intermediate_2_ptrs = intermediate_2_ptr + stride_i2m * offs_i2m[:, None] + stride_i2k * offs_i2k[None, :]

    i1_mask = (offs_i1m[:, None] < M) & (offs_i1k[None, :] < K)
    i2_mask = (offs_i2m[:, None] < M) & (offs_i2k[None, :] < K)

    tl.store(intermediate_1_ptrs, intermediate_1, mask=i1_mask)
    tl.store(intermediate_2_ptrs, intermediate_2, mask=i2_mask)


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
def backward_weight_kernel(
        x_ptr,
        w3_ptr, w1_ptr,
        w3_grad_ptr, w1_grad_ptr,
        M, N, K,
        stride_gdm, stride_gdk,
        stride_w3k, stride_w3n,
        stride_w1k, stride_w1n,
        stride_cm, stride_cn,
        stride_dm, stride_dn,
        dtype: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_gdm + offs_k[None, :] * stride_gdk)

    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_n[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_n[None, :] * stride_w3n)

    accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)

        w1 = tl.load(w1_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        w3 = tl.load(w3_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator1 += tl.dot(x, w1)
        accumulator2 += tl.dot(x, w3)

        x_ptrs += BLOCK_SIZE_K * stride_gdk
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k

    c = accumulator1.to(dtype)
    d = accumulator2.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    w1_grad_ptrs = w1_grad_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    w3_grad_ptrs = w3_grad_ptr + stride_dm * offs_cm[:, None] + stride_dn * offs_cn[None, :]

    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(w1_grad_ptrs, c, mask=c_mask)
    tl.store(w3_grad_ptrs, d, mask=c_mask)


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
def backward_grad_input_kernel(
        grad_output_ptr, grad1_output_ptr,
        w3_ptr, w1_ptr,
        c_ptr,
        M, N, K,
        stride_gdm, stride_gdk,
        stride_gd1m, stride_gd1k,
        stride_w3k, stride_w3n,
        stride_w1k, stride_w1n,
        stride_cm, stride_cn,
        dtype: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    grad_output_ptrs = grad_output_ptr + (offs_m[:, None] * stride_gdm + offs_k[None, :] * stride_gdk)
    grad_output1_ptrs = grad1_output_ptr + (offs_m[:, None] * stride_gd1m + offs_k[None, :] * stride_gd1k)

    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_n[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_n[None, :] * stride_w3n)

    accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        grad_output = tl.load(grad_output_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        grad1_output = tl.load(grad_output1_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)

        w1 = tl.load(w1_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        w3 = tl.load(w3_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator1 += tl.dot(grad_output, w3)
        accumulator2 += tl.dot(grad1_output, w1)

        grad_output_ptrs += BLOCK_SIZE_K * stride_gdk
        grad_output1_ptrs += BLOCK_SIZE_K * stride_gd1k

        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k

    c = (accumulator1 + accumulator2).to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def grad_input_kernel_wrapper(grad_output, x, w1, w3, act_in, mm_1):
    assert grad_output.is_contiguous(), "Grad Output must be contiguous"
    assert grad_output.is_contiguous(), "Grad Output must be contiguous"
    assert w1.is_contiguous(), "First set of weights must be contiguous"
    assert w3.is_contiguous(), "Second set of weights must be contiguous"

    M, K = grad_output.shape
    N = 1

    intermediate_cache_1 = torch.empty((M, K), device=grad_output.device, dtype=grad_output.dtype)
    intermediate_cache_2 = torch.empty((M, K), device=grad_output.device, dtype=grad_output.dtype)

    # First kernel calculates the intermediate caches necessary for the final input_grad calculation
    # Recomputes act_out to save memory
    # mul_1 = grad_output * act_out
    # mul_5 = silu_grad(grad_output * mm_1, act_inputs)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(K, META['BLOCK_SIZE_K']))
    backward_intermediate_kernel[grid](
        grad_output, mm_1, act_in,
        intermediate_cache_1, intermediate_cache_2,
        M, N, K,
        grad_output.stride(0), grad_output.stride(1),
        mm_1.stride(0), mm_1.stride(1),
        act_in.stride(0), act_in.stride(1),
        intermediate_cache_1.stride(0), intermediate_cache_1.stride(1),
        intermediate_cache_2.stride(0), intermediate_cache_2.stride(1),
        map_dtype(grad_output.dtype),
        )

    M, K = intermediate_cache_1.shape
    K, N = w1.t().shape

    grad_input = torch.empty((M, N), device=grad_output.device, dtype=grad_output.dtype)

    # Second kernel takes the intermediate caches and cacluates the input grad
    # grad_input = intermediate_1 @ w3.T + intermediate_2 @ w1.T
    grid = lambda META: ((triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N'])), )
    w3, w1 = w3.t().contiguous(), w1.t().contiguous()
    backward_grad_input_kernel[grid](
        intermediate_cache_1, intermediate_cache_2,
        w3, w1, grad_input,  #
        M, N, K,  #
        *intermediate_cache_1.stride(),
        *intermediate_cache_2.stride(),
        *w3.stride(),
        *w1.stride(),
        *grad_input.stride(),
        map_dtype(grad_output.dtype),
    )

    # Third and final kernel calculates the gradient of the weights using the intermediate caches we have saved
    # w2_grad = x.T @ mul_1
    # w1_grad = x.T @ mul_5
    M, K = x.t().shape
    K, N = intermediate_cache_1.shape

    w1_grad = torch.empty((M, N), device=grad_output.device, dtype=grad_output.dtype)
    w3_grad = torch.empty((M, N), device=grad_output.device, dtype=grad_output.dtype)

    # Second kernel takes the intermediate caches and cacluates the input grad
    # grad_input = intermediate_1 @ w3.T + intermediate_2 @ w1.T
    grid = lambda META: ((triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N'])), )
    x = x.t().contiguous()
    backward_weight_kernel[grid](
        x, intermediate_cache_1, intermediate_cache_2,
        w3_grad, w1_grad,
        M, N, K,  #
        *x.stride(),
        *intermediate_cache_1.stride(),
        *intermediate_cache_2.stride(),
        *w3_grad.stride(),
        *w1_grad.stride(),
        map_dtype(grad_output.dtype),
    )

    return grad_input, w1_grad, w3_grad

def silu_pt(input):
    return input * torch.sigmoid(input)

def silu_grad_pt(grad_output, act_in):
    sigma = 1 / (1 + torch.exp(-act_in.to(torch.float32)))
    grad_input = grad_output * (sigma + act_in * sigma * (1 - sigma))
    return grad_input.to(grad_output.dtype)

def torch_reference(grad_output, x, w1, w3, act_out, act_in, mm_1):
    w3, w1 = w3.t().contiguous(), w1.t().contiguous()
    mul_1 = grad_output * act_out
    mul_5 = silu_grad_pt(grad_output * mm_1, act_in)
    grad_input = mul_1 @ w3 + mul_5 @ w1

    w1_grad = x.T @ mul_5
    w3_grad = x.T @ mul_1
    return grad_input, w1_grad, w3_grad

if __name__ == "__main__":
    grad_output = torch.rand([2048, 11008], device='cuda', dtype=torch.float16)
    x = torch.rand([2048, 4096], device='cuda', dtype=torch.float16)
    w1 = (torch.rand([4096, 11008], device='cuda', dtype=torch.float16) * 0.02)
    w3 = (torch.rand([4096, 11008], device='cuda', dtype=torch.float16)* 0.02)
    act_in = torch.matmul(x, w1)
    mm_1 = torch.matmul(x, w3)
    act_out = silu_pt(act_in)

    grad_input, w1_grad, w3_grad = grad_input_kernel_wrapper(grad_output, x, w1, w3, act_in, mm_1)

    grad_input_pt, w1_grad_pt, w3_grad_pt = torch_reference(grad_output, x, w1, w3, act_out, act_in, mm_1)

    print(grad_input - grad_input_pt)
    print(w1_grad - w1_grad_pt)
    print(w3_grad - w3_grad_pt)

    print("llama mlp backward triton", triton.testing.do_bench(lambda: grad_input_kernel_wrapper(grad_output, x, w1, w3, act_in, mm_1)))
    print("llama mlp backward pytorch", triton.testing.do_bench(lambda: torch_reference(grad_output, x, w1, w3, act_out, act_in, mm_1)))