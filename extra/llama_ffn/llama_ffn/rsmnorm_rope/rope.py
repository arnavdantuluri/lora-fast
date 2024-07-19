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
#Copied from https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rope_embedding.py
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

@triton.heuristics({"BACKWARD_PASS": lambda args: args["BACKWARD_PASS"],})
@triton.jit
def _rope_embedding(
    Q,     Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen, head_dim,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        See our blog post for more info
    """
    row_position  = tl.program_id(0)
    head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    Q1   = tl.load(Q + row_position*Q_row_stride + head_position*head_dim + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)
    Q2   = tl.load(Q + row_position*Q_row_stride + head_position*head_dim + \
                   half_head_dim*1 + col_offsets, mask = mask, other = 0)
    sin1 = tl.load(sin + (row_position % seqlen)*sin_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)

    if BACKWARD_PASS:
        # See our blog post for more info.
        sin1 = -sin1
    pass

    tl.store(Q + row_position*Q_row_stride + head_position*head_dim + \
             half_head_dim*0 + col_offsets, Q1*cos1 - Q2*sin1, mask = mask)
    tl.store(Q + row_position*Q_row_stride + head_position*head_dim + \
             half_head_dim*1 + col_offsets, Q2*cos1 + Q1*sin1, mask = mask)


class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.view(batch*seq_len, n_heads*head_dim)
        n_rows, n_cols = Q.shape
        assert(seq_len <= cos.shape[0])

        # [TODO] Changing blocksize to head_dim//2 seems to have
        # some concurrency / un-deterministic issues.
        BLOCK_SIZE, num_warps = calculate_settings(head_dim) # (head_dim//2)
        _rope_embedding[(n_rows, n_heads,)](
              Q,   Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim,
            BACKWARD_PASS = False,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.cos = cos
        ctx.sin = sin
        return Q.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dY):
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.reshape(batch*seq_len, n_heads*head_dim)
        # Must be reshape not view
        n_rows, n_cols = dY.shape

        cos = ctx.cos
        sin = ctx.sin

        _rope_embedding[(n_rows, n_heads,)](
            dY,  dY .stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim,
            BACKWARD_PASS = True,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dY = dY.view(batch, seq_len, n_heads, head_dim)
        #sin and cos should not need gradients returned. Only Q and K should need gradients
        return dY, None, None,


def fast_rope_embedding(Q, K, cos, sin):
    Q = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    K = Fast_RoPE_Embedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return Q, K