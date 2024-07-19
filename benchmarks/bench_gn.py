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

import torch

import triton
import triton.language as tl

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

import apex.contrib.group_norm.group_norm as groupnorm
from triton_.ops.groupnorm_fwd import group_norm_forward

group_norm = groupnorm.GroupNormNHWC.apply


def test_layer_norm(M, N, dtype, group_size=32, eps=1e-5, device='cuda'):
    M = 512
    # create data
    x_shape = (2, M, N, 4)
    w_shape = (M, )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    x = x.to(memory_format=torch.channels_last)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)

    y_tri = group_norm(x, group_size, weight, bias, eps)
    y_ref = torch.nn.functional.group_norm(x, group_size, weight, bias, eps).to(dtype)

    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]

    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    # assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 10)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='layer-norm-forward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'forward'},
    ))
def bench_layer_norm(M, N, dtype, provider, group_size=32, mode='backward', eps=1e-5, device='cuda'):
    # create data
    x_shape = (1, M, N, 4)  
    w_shape = (x_shape[1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    if provider == "apex" or provider == "triton":
        x = x.to(memory_format=torch.channels_last).requires_grad_(True) 
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return group_norm_forward(x, group_size, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.group_norm(x, group_size, weight, bias, eps)  # noqa: F811, E704

        if provider == "apex":
            return (group_norm(x, group_size, weight, bias, eps))
        
    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return ms, max_ms, min_ms


# test_layer_norm(1151, 8192, torch.float16)
bench_layer_norm.run(save_path='.', print_data=True)

# %%
# References
# ----------
#
# .. [BA2016] Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton, "Layer Normalization", Arxiv 2016