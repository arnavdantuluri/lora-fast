#Taken from unsloth github and slightly modified
import torch

def get_lora_parameters(proj):
    # For DPO or disabled adapters
    base_layer = (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, None, None, None

    active_adapter = proj.active_adapters[0] if \
        hasattr(proj, "active_adapters") else proj.active_adapter
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, A, B, s

def matmul_lora(X, W, A, B, s, out = None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False
    pass

    out = torch.matmul(X, W.t(), out = out)
    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        out += (X @ A.to(dtype)) @ (s * B.to(dtype))
    pass

    return out.view(batch, seq_len, -1) if reshape else out
pass

class LoRA_QKV(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    See our blogpost for more details.

    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)
    We then sum them all find dC/dX

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, X : torch.Tensor,
                QW, QA, QB, QS,
                KW, KA, KB, KS,
                VW, VA, VB, VS,):
        dtype = X.dtype

        Q = matmul_lora(X, QW, QA, QB, QS)
        K = matmul_lora(X, KW, KA, KB, KS)
        V = matmul_lora(X, VW, VA, VB, VS)

        ctx.custom_saved_tensors = (
            QW, QS,
            KW, KS,
            VW, VS,
        )
        ctx.save_for_backward(X, QA, QB, KA, KB, VA, VB,)
        return Q, K, V
    pass

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dQ, dK, dV):
        QW, QS, KW, KS, VW, VS = \
            ctx.custom_saved_tensors
        X, QA, QB, KA, KB, VA, VB, = ctx.saved_tensors

        QA, QB, KA, KB, VA, VB = \
            QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()

        batch, seq_len, hd = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1]) # view doesn't work on K.T
        dV = dV.view(-1, dV.shape[-1])
        X  = X .view(-1, X .shape[-1])
        dtype = X.dtype

        ### Weight projection LoRA weights
        # See our blogpost for more details.

        # Q Projection
        d_QA = X.t() @ (dQ @ QB.t())
        d_QB = (QA.t() @ X.t()) @ dQ
        d_QA *= QS
        d_QB *= QS

        # K Projection
        d_KA = X.t() @ (dK @ KB.t())
        d_KB = (KA.t() @ X.t()) @ dK
        d_KA *= KS
        d_KB *= KS

        # V Projection
        d_VA = X.t() @ (dV @ VB.t())
        d_VB = (VA.t() @ X.t()) @ dV
        d_VA *= VS
        d_VB *= VS

        # Combine derivatives to find dX
        # dQ
        QW = QW.t()
        dX = torch.matmul(dQ, QW.t(), out = X)
        del QW
        dX += (dQ @ QB.to(dtype).t() @ (QS * QA.to(dtype).t()))

        # dK
        KW = KW.t()
        dX += dK @ KW.t()
        del KW
        dX += dK @ KB.to(dtype).t() @ (KS * KA.to(dtype).t())

        # dV
        VW = VW.t()
        dX += dV @ VW.t()
        del VW
        dX += dV @ VB.to(dtype).t() @ (VS * VA.to(dtype).t())

        return dX.view(batch, seq_len, hd), \
            None, None, d_QA.t(), d_QB.t(), None, \
            None, None, d_KA.t(), d_KB.t(), None, \
            None, None, d_VA.t(), d_VB.t(), None
    pass
pass


class LoRA_W(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, X : torch.Tensor,
                W, A, B, S):
        dtype = X.dtype
        XW = matmul_lora(X, W, A, B, S)
        ctx.custom_saved_tensors = (W, S,)
        ctx.save_for_backward(A, B, X)
        return XW
    pass

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY : torch.Tensor):
        W, S = ctx.custom_saved_tensors
        A, B, X = ctx.saved_tensors

        A, B = A.t(), B.t()

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1]) # Must be reshape
        X  = X .reshape(-1, X .shape[-1]) # Must be reshape
        dtype = X.dtype

        ### Weight projection LoRA weights
        # Weight projection
        d_A = X.t() @ (dY @ B.t())
        d_B = (A.t() @ X.t()) @ dY
        d_A *= S
        d_B *= S

        # Get derivative for dX
        W = W.t()
        dX = dY @ W.t()
        del W
        dX += dY @ B.to(dtype).t() @ (S * A.to(dtype).t())

        return dX.view(batch, seq_len, hd), \
            None, d_A.t(), d_B.t(), None, None


def apply_lora_o(layer, X):
    OW, OA, OB, OS = get_lora_parameters(layer)
    O = LoRA_W.apply(X, OW, OA, OB, OS)
    return O

def apply_lora_w(X, OW, OA, OB, OS):
    O = LoRA_W.apply(X, OW, OA, OB, OS)
    return O

def apply_lora_qkv(self, X):
    QW, QA, QB, QS = get_lora_parameters(self.to_q)
    KW, KA, KB, KS = get_lora_parameters(self.to_k)
    VW, VA, VB, VS = get_lora_parameters(self.to_v)
    Q, K, V = LoRA_QKV.apply(X,
        QW, QA, QB, QS,
        KW, KA, KB, KS,
        VW, VA, VB, VS,
    )
    return Q, K, V