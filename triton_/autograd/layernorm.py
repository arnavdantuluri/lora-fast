import functools
import operator
import torch
import triton
import triton.language as tl
from ops.layernorm_fwd import _layer_norm_fwd_fused
from ops.layernorm_bwd import _layer_norm_bwd_dx_fused, _layer_norm_bwd_dwdb

try:
    import apex
    HAS_APEX = True
except:
    HAS_APEX = False

aten = torch.ops.aten
# %%
# Benchmark
# ---------
#
# We can now compare the performance of our kernel against that of PyTorch.
# Here we focus on inputs that have Less than 64KB per feature.
# Specifically, one can set :code:`'mode': 'backward'` to benchmark the backward pass.


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        x = x.contiguous()
        weight = weight.contiguous() if weight is not None else None
        bias = bias.contiguous() if bias is not None else None
        # allocate output
        y = torch.empty_like(x)

        N = functools.reduce(operator.mul, normalized_shape, 1)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, N)
        M, N = x_arg.shape
        needs_backward = any(x is not None and x.requires_grad
                             for x in [x, weight, bias])
        if needs_backward:
            mean = torch.empty((M, ), dtype=x.dtype, device=x.device)
            rstd = torch.empty((M, ), dtype=x.dtype, device=x.device)
        else:
            mean, rstd = None, None
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError(
                "This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 16)
        # enqueue kernel
        _layer_norm_fwd_fused[(M, )](  #
            x_arg,
            y,
            weight,
            bias,
            mean,
            rstd,  #
            x_arg.stride(0),
            N,
            eps,  #
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            # num_ctas=1,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.normalized_shape = normalized_shape
        return y

    @staticmethod
    def backward(ctx, dy):
        dy.contiguous()
        x, w, b, m, v = ctx.saved_tensors
        x = x.contiguous()
        w = w.contiguous() if w is not None else None
        b = b.contiguous() if b is not None else None
        m = m.contiguous()
        v = v.contiguous()

        # grad_input_mask = (ctx.needs_input_grad[0], ctx.needs_input_grad[2],
        #                    ctx.needs_input_grad[3])
        # grad_inputs = aten.native_layer_norm_backward(dy, x,
        #                                               ctx.normalized_shape, m,
        #                                               v, w, b, grad_input_mask)
        # dx, dw, db = grad_inputs
        # return dx, None, dw, db, None

        M = m.numel()
        N = x.numel() // M
        # heuristics for amount of parallel reduction stream for DW/DB
        # N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192:
            GROUP_SIZE_M = 96
        if N <= 4096:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device='cuda')
        _dw = torch.empty((GROUP_SIZE_M, w.shape[0]),
                          dtype=x.dtype,
                          device=w.device)
        _db = torch.empty((GROUP_SIZE_M, w.shape[0]),
                          dtype=x.dtype,
                          device=w.device)
        dw = torch.empty((w.shape[0], ), dtype=w.dtype, device=w.device)
        db = torch.empty((w.shape[0], ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M, )](  #
            dx,
            dy,
            _dw,
            _db,
            x,
            w,
            b,
            m,
            v,
            locks,  #
            x_arg.stride(0),
            N,
            ctx.eps,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            num_warps=ctx.num_warps)

        def grid(meta):
            return [triton.cdiv(N, meta['BLOCK_SIZE_N'])]

        # accumulate partial sums in separate kernel
        _layer_norm_bwd_dwdb[grid](
            _dw,
            _db,
            dw,
            db,
            GROUP_SIZE_M,
            N,  #
            BLOCK_SIZE_M=32,  #
            BLOCK_SIZE_N=128,
            # num_ctas=1,
        )
        return dx, None, dw, db, None


layer_norm = LayerNorm.apply

if __name__ == '__main__':

    def test_layer_norm(M, N, dtype, eps=1e-5, device='cuda'):
    # create data
        x_shape = (M, N)
        w_shape = (x_shape[-1], )
        weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
        dy = .1 * torch.randn_like(x)
        x.requires_grad_(True)
        # forward pass
        y_tri = layer_norm(x, w_shape, weight, bias, eps)
        y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
        # backward pass (triton)
        y_tri.backward(dy, retain_graph=True)
        dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
        x.grad, weight.grad, bias.grad = None, None, None
        # backward pass (torch)
        y_ref.backward(dy, retain_graph=True)
        dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
        # compare
        assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
        assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
        assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
        assert torch.allclose(dw_tri, dw_ref, atol=1e-1, rtol=0)


    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[512 * i for i in range(2, 32)],
            line_arg='provider',
            line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
            line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
            styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
            ylabel='GB/s',
            plot_name='layer-norm-backward',
            args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
        ))
    def bench_layer_norm(M, N, dtype, provider, mode='forward', eps=1e-5, device='cuda'):
        mode = "backward"
        # create data
        x_shape = (M, N)
        w_shape = (x_shape[-1], )
        weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
        dy = .1 * torch.randn_like(x)
        x.requires_grad_(True)
        quantiles = [0.5, 0.2, 0.8]

        def y_fwd():

            if provider == "triton":
                return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

            if provider == "torch":
                return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

            if provider == "apex":
                apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
                return apex_layer_norm(x)  # noqa: F811, E704

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


    test_layer_norm(1151, 8192, torch.float16)
    bench_layer_norm.run(save_path='.', print_data=True)
