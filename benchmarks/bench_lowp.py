import sfast #sfast doesn't have python interface only cpp
from cublas_ops import CublasLinear, cublas_half_matmul_simple, cublaslt_fused_half_matmul_batched_simple, \
                                        cublas_half_matmul_batched_simple, cublaslt_fused_half_matmul_simple

import torch

import triton

# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = torch.matmul(a, b.T)
torch_output = cublaslt_fused_half_matmul_simple(a, b, None, "NONE")
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")

rtol =  0
if torch.allclose(triton_output, torch_output, atol=1e-0, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


ref_lib = 'cuBLAS'

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 36)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib.lower(), "triton"],  # Label name for the lines
        line_names= [ref_lib, "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="ms",  # Label name for the y-axis
        plot_name="matmul-performance-" +
        ("fp16" ),  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))

def mm_act(a, b):
    return torch.nn.functional.gelu(torch.matmul(a, b))

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((2, M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((2, K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: cublaslt_fused_half_matmul_batched_simple(a, b, None, 'GELU'), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return ms, max_ms, min_ms


benchmark.run(print_data=True, save_path='.')