"""
Layer Normalization
====================
In this tutorial, you will write a high-performance layer normalization
kernel that runs faster than the PyTorch implementation.

In doing so, you will learn about:

* Implementing backward pass in Triton.

* Implementing parallel reduction in Triton.

"""

# %%
# Motivations
# -----------
#
# The *LayerNorm* operator was first introduced in [BA2016]_ as a way to improve the performance
# of sequential models (e.g., Transformers) or neural networks with small batch size.
# It takes a vector :math:`x` as input and produces a vector :math:`y` of the same shape as output.
# The normalization is performed by subtracting the mean and dividing by the standard deviation of :math:`x`.
# After the normalization, a learnable linear transformation with weights :math:`w` and biases :math:`b` is applied.
# The forward pass can be expressed as follows:
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# where :math:`\epsilon` is a small constant added to the denominator for numerical stability.
# Let’s first take a look at the forward pass implementation.

import functools
import operator
import torch
import triton
import triton.language as tl
from utils import welford_combine

# try:
#     # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
#     # should not be added to extras_require in setup.py.
#     import apex
#     HAS_APEX = True
# except ModuleNotFoundError:
#     HAS_APEX = False
HAS_APEX = False

aten = torch.ops.aten


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride: tl.
    constexpr,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    if BLOCK_SIZE >= N:
        cols = tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N).to(tl.float32)
        m2_ = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        weight_ = (cols < N).to(tl.float32)
        _mean, _m2, _weight = x, m2_, weight_
    else:
        # Compute mean
        _mean = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        _m2 = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        _weight = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N).to(tl.float32)
            m2_ = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
            weight_ = (cols < N).to(tl.float32)
            if off == 0:
                _mean, _m2, _weight = x, m2_, weight_
            else:
                _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x,
                                                      m2_, weight_)
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    var = m2 / weight
    rstd = 1 / tl.sqrt(var + eps)
    mean = mean.to(x.dtype)
    rstd = rstd.to(x.dtype)
    # Write mean / rstd
    if Mean is not None:
        tl.store(Mean + row, mean)
    if Rstd is not None:
        tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    if BLOCK_SIZE >= N:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        if W is None:
            w = tl.full((BLOCK_SIZE, ), 1.0, dtype=x.dtype)
        else:
            w = tl.load(W + cols, mask=mask)
        if B is None:
            b = tl.zeros((BLOCK_SIZE, ), dtype=x.dtype)
        else:
            b = tl.load(B + cols, mask=mask)
        # x = tl.load(X + cols, mask=mask).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)
    else:
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            if W is None:
                w = tl.full((BLOCK_SIZE, ), 1.0, dtype=x.dtype)
            else:
                w = tl.load(W + cols, mask=mask)
            if B is None:
                b = tl.zeros((BLOCK_SIZE, ), dtype=x.dtype)
            else:
                b = tl.load(B + cols, mask=mask)
            x = tl.load(X + cols, mask=mask)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            # Write output
            tl.store(Y + cols, y, mask=mask)