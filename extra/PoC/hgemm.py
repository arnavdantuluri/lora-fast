import math
from typing import Literal, Optional

import torch
from cublas_ops_ext import _simt_hgemv
from cublas_ops_ext import cublas_hgemm_axbT as _cublas_hgemm_axbT
from cublas_ops_ext import cublas_hgemm_batched_simple as _cublas_hgemm_batched_simple
from cublas_ops_ext import (
    cublaslt_hgemm_batched_simple as _cublaslt_hgemm_batched_simple,
)
from cublas_ops_ext import cublaslt_hgemm_simple as _cublaslt_hgemm_simple
from torch import nn
from torch.autograd.function import FunctionCtx

global has_moved
has_moved = {idx: False for idx in range(torch.cuda.device_count())}


class StaticState:
    workspace = {
        idx: torch.empty((1024 * 1024 * 8,), dtype=torch.uint8)
        for idx in range(torch.cuda.device_count())
    }
    workspace_size = workspace[0].nelement()
    bias_g = {
        idx: torch.tensor([], dtype=torch.float16)
        for idx in range(torch.cuda.device_count())
    }

    @classmethod
    def get(cls, __name: str, device: torch.device) -> torch.Any:
        global has_moved
        idx = device.index if device.index is not None else 0
        if not has_moved[idx]:
            cls.workspace[idx] = cls.workspace[idx].cuda(idx)
            cls.bias_g[idx] = cls.bias_g[idx].cuda(idx)
            has_moved[idx] = True
        if "bias" in __name:
            return cls.bias_g[idx]
        if "workspace" in __name:
            return cls.workspace[idx]
        if "workspace_size" in __name:
            return cls.workspace_size


@torch.no_grad()
def hgemv_simt(vec: torch.HalfTensor, mat: torch.HalfTensor, block_dim_x: int = 32):
    prev_dims = vec.shape[:-1]
    return _simt_hgemv(mat, vec.view(-1, 1), block_dim_x=block_dim_x).view(
        *prev_dims, -1
    )


@torch.no_grad()
def cublas_half_matmul_batched_simple(a: torch.Tensor, b: torch.Tensor):
    return _cublas_hgemm_batched_simple(a, b)


@torch.no_grad()
def cublas_half_matmul_simple(a: torch.Tensor, b: torch.Tensor):
    return _cublas_hgemm_axbT(b, a)


@torch.no_grad()
def cublaslt_fused_half_matmul_simple(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue_str: Optional[Literal["NONE", "RELU", "GELU"]] = "NONE",
):
    if bias is None:
        bias = StaticState.get("bias", a.device)
    return _cublaslt_hgemm_simple(
        a, b, bias, epilogue_str, StaticState.get("workspace", a.device)
    )


@torch.no_grad()
def cublaslt_fused_half_matmul_batched_simple(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue_str: Optional[Literal["NONE", "RELU", "GELU"]] = "NONE",
):
    if bias is None:
        bias = StaticState.get("bias", a.device)
    return _cublaslt_hgemm_batched_simple(
        a, b, bias, epilogue_str, StaticState.get("workspace", a.device)
    )


class CublasLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx:FunctionCtx, x, weight, bias=None, has_bias=False, _epilogue_str="NONE"):

        bias_ref = None if not has_bias else bias
        x = x.to(torch.float16)
        weight = weight.to(torch.float16)
        bias = bias.to(torch.float16) if has_bias else None
        ctx.save_for_backward(x, weight)
        
        if x.dtype != torch.float16 or weight.device.type != "cuda":
            print("using torch")
            out = torch.nn.functional.linear(x, weight, bias)
            if _epilogue_str == "RELU":
                out = torch.relu(out)
            elif _epilogue_str == "GELU":
                out = torch.nn.functional.gelu(out)
            return out

        use_cublasLt = has_bias or _epilogue_str != "NONE"
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        if not use_cublasLt:
            if x.ndim == 3:
                return cublas_half_matmul_batched_simple(x, weight)
            elif x.ndim == 2:
                return cublas_half_matmul_simple(x, weight)
            leading_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = cublas_half_matmul_simple(x, weight).view(
                *leading_dims, out.shape[-1]
            )
        if use_cublasLt:
            if x.ndim == 3:
                return cublaslt_fused_half_matmul_batched_simple(
                    x, weight, bias=bias_ref, epilogue_str=_epilogue_str
                )
            elif x.ndim == 2:
                return cublaslt_fused_half_matmul_simple(
                    x, weight, bias=bias_ref, epilogue_str=_epilogue_str
                )

            leading_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = cublaslt_fused_half_matmul_simple(
                x, weight, bias=bias_ref, epilogue_str=_epilogue_str
            ).view(*leading_dims, out.shape[-1])
        
        return out
    
    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs):
        # Cannot do fp16 accumulation during backward pass
        x, w = ctx.saved_tensors
        
        grad_output = grad_outputs[0]
        grad_output = grad_output * 128 #scaling factor of 128

        batched = True if x.dim() == 3 else False
        if batched:
            dout = cublas_half_matmul_batched_simple(grad_output, w.T)
            dw = cublas_half_matmul_batched_simple(x, grad_output)
        else:
            dout = cublas_half_matmul_simple(grad_output, w.T)
            dw = cublas_half_matmul_simple(x, grad_output)
        # dout = torch.matmul(grad_output, w)
        # dw = torch.matmul(x.T, grad_output)
        dout = dout * 1/128

        return dout, dw, None, None, None


class CublasLinearGelu(CublasLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=torch.float16,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            epilogue_str="GELU",
        )


class CublasLinearRelu(CublasLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=torch.float16,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            epilogue_str="RELU",
        )


__ALL__ = [
    "CublasLinear",
    "CublasLinearGelu",
    "CublasLinearRelu",
    "cublas_half_matmul_simple",
    "cublas_half_matmul_batched_simple",
    "cublaslt_fused_half_matmul_simple",
    "cublaslt_fused_half_matmul_batched_simple",
]