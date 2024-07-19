#Taken from stable-fast: https://github.com/chengzeyi/stable-fast/tree/main

import torch
from torch._prims_common import suggest_memory_format
import triton
import triton.language as tl

#Taken from stable-fast: https://github.com/chengzeyi/stable-fast/tree/main

import triton
import triton.language as tl


@triton.jit
def identity(x):
    return x


@triton.jit
def silu(x):
    return x * tl.sigmoid(x.to(tl.float32)).to(x.dtype)


@triton.jit
def relu(x):
    return tl.max(x, 0.0)


@triton.jit
def gelu(x):
    return 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 *
                                    (x + 0.044715 * x * x * x)))

act = identity

#Taken from stable-fast: https://github.com/chengzeyi/stable-fast/tree/main

import triton
import triton.language as tl
import copy
import types
import functools

@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    # w2_over_w = weight_2 / new_weight
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


def copy_func(f, globals=None, module=None, name=None):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    if name is None:
        name = f.__name__
    g = types.FunctionType(f.__code__,
                           globals,
                           name=name,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    if module is not None:
        g.__module__ = module
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    g.__name__ = name
    return g
# Stupid: https://github.com/openai/triton/issues/1589
@eval('''triton.heuristics({
    'ROW_SIZE':
    lambda kwargs: triton.next_power_of_2(kwargs['C'] // kwargs['groups']),
    'BLOCK_SIZE':
    lambda kwargs: max(
        1, min(triton.next_power_of_2(kwargs['HxW']),
               4096 // (triton.next_power_of_2(kwargs['C'] // kwargs['groups']))
               )),
})''')
@eval('''triton.heuristics({
    'num_warps':
    lambda kwargs: max(1, min(16, kwargs['ROW_SIZE'] * kwargs['BLOCK_SIZE'] // 128)),
    'C_G': lambda kwargs: kwargs['C'] // kwargs['groups'],
})''')
@triton.jit
def group_norm_4d_channels_last_forward_collect_stats_kernel(
    input_ptr,
    N,
    C,
    HxW,
    groups,
    eps,
    mean_ptr,
    rstd_ptr,
    C_G,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    pid_batch = tl.program_id(1)

    offset = pid_batch * C * HxW + group * C_G
    X = input_ptr + offset
    _mean = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _m2 = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _weight = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    row = tl.arange(0, ROW_SIZE)
    for off in range(0, HxW, BLOCK_SIZE):
        r = off + tl.arange(0, BLOCK_SIZE)
        m2_ = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
        mask = (r < HxW)[:, None] & (row[None, :] < C_G)
        weight_ = mask.to(tl.float32)
        x = tl.load(X + (r * C)[:, None] + row[None, :],
                    mask=mask).to(tl.float32)
        _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x, m2_,
                                              weight_)
    _mean = tl.view(_mean, (BLOCK_SIZE * ROW_SIZE, ))
    _m2 = tl.view(_m2, (BLOCK_SIZE * ROW_SIZE, ))
    _weight = tl.view(_weight, (BLOCK_SIZE * ROW_SIZE, ))
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    var = m2 / weight
    rstd = 1. / tl.sqrt(var + eps)
    offset = pid_batch * groups + group
    tl.store(mean_ptr + offset, mean)
    tl.store(rstd_ptr + offset, rstd)

def group_norm_4d_channels_last_forward_apply_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    C,
    HxW,
    groups,
    eps,
    output_ptr,
    C_G,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hw = tl.program_id(0) * BLOCK_SIZE
    pid_batch = tl.program_id(1)

    offset = pid_batch * C * HxW
    X = input_ptr + offset
    Y = output_ptr + offset
    group_row = tl.arange(0, ROW_SIZE)
    group_row = group_row // C_G
    group_mask = group_row < groups
    mean = tl.load(mean_ptr + pid_batch * groups + group_row, mask=group_mask)
    rstd = tl.load(rstd_ptr + pid_batch * groups + group_row, mask=group_mask)
    row = tl.arange(0, ROW_SIZE)
    mask = row < C
    if gamma_ptr is None:
        gamma = tl.full((ROW_SIZE, ), 1., dtype=mean.dtype)
    else:
        gamma = tl.load(gamma_ptr + row, mask=mask)
    if beta_ptr is None:
        beta = tl.zeros((ROW_SIZE, ), dtype=mean.dtype)
    else:
        beta = tl.load(beta_ptr + row, mask=mask)
    a = rstd * gamma
    b = beta - a * mean
    a = a[None, :]
    b = b[None, :]
    r = hw + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + (r * C)[:, None] + row[None, :],
                mask=(r < HxW)[:, None] & mask[None, :])
    x = a * x + b
    x = act(x)
    tl.store(Y + (r * C)[:, None] + row[None, :],
             x,
             mask=(r < HxW)[:, None] & mask[None, :])


def create_group_norm_4d_channels_last_forward_apply_kernel(
        act=identity):
    kernel = group_norm_4d_channels_last_forward_apply_kernel
    kernel = copy_func(kernel,
                       globals={
                           **globals(),
                           **{
                               'act': act
                           }
                       },
                       name=f'{kernel.__name__}_{act.__name__}')
    kernel = triton.heuristics({
        'ROW_SIZE':
        lambda kwargs: triton.next_power_of_2(kwargs['C']),
        'BLOCK_SIZE':
        lambda kwargs: max(
            1,
            min(triton.next_power_of_2(kwargs['HxW']), 4096 // triton.
                next_power_of_2(kwargs['C']))),
    })(triton.heuristics({
        'num_warps':
        lambda kwargs: max(
            1, min(16, kwargs['ROW_SIZE'] * kwargs['BLOCK_SIZE'] // 128)),
        'C_G':
        lambda kwargs: kwargs['C'] // kwargs['groups'],
    })(triton.jit(kernel)))
    return kernel


def create_group_norm_forward(act=identity):
    group_norm_4d_channels_last_forward_apply_kernel = create_group_norm_4d_channels_last_forward_apply_kernel(
        act=act)

    def group_norm_forward(input,
                           num_groups,
                           weight=None,
                           bias=None,
                           eps=1e-05,
                           output_mean=True,
                           output_rstd=True):
        assert input.device.type == 'cuda'
        assert 2 <= input.ndim <= 4
        dim_padding = 0
        while input.ndim < 4:
            input = input.unsqueeze(-1)
            dim_padding += 1
        shape = input.shape
        N, C, H, W = shape
        assert C % num_groups == 0
        assert weight is None or weight.shape == (C, )
        assert bias is None or bias.shape == (C, )
        if weight is not None:
            assert weight.device.type == 'cuda'
            weight = weight.contiguous()
        if bias is not None:
            assert bias.device.type == 'cuda'
            bias = bias.contiguous()
        memory_format = suggest_memory_format(input)
        if memory_format == torch.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

            mean = torch.empty((
                N,
                num_groups,
            ),
                dtype=input.dtype,
                device=input.device)
            rstd = torch.empty((
                N,
                num_groups,
            ),
                dtype=input.dtype,
                device=input.device)

            def grid(meta):
                return (num_groups, N)

            group_norm_4d_channels_last_forward_collect_stats_kernel[grid](
                input, N, C, H * W, num_groups, eps, mean, rstd)

            output = torch.empty_like(input)

            def grid(meta):
                return (triton.cdiv(H * W, meta['BLOCK_SIZE']), N)

            group_norm_4d_channels_last_forward_apply_kernel[grid](
                input, weight, bias, mean, rstd, N, C, H * W, num_groups, eps,
                output)

            if not output_mean:
                mean = None
            if not output_rstd:
                rstd = None
        else:
            raise RuntimeError("No Tensor Cores found, please disable Group Norm optimization in optimization config.")

    return group_norm_forward


group_norm_forward = create_group_norm_forward()
group_norm_silu_forward = create_group_norm_forward(act=silu)