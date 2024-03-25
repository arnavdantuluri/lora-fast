# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from peft import LoraConfig
from peft.utils.loftq_utils import loftq_init
from lora_kernel import apply_lora_w


from contextlib import contextmanager

import packaging.version
import torch
import transformers


@contextmanager
def gather_params_ctx(module: torch.nn.Module, modifier_rank: int = 0):
    """Call DeepSpeed GatheredParameters context manager if DeepSpeed is enabled, otherwise do nothing."""
    if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
        from transformers.integrations import is_deepspeed_zero3_enabled
    else:
        from transformers.deepspeed import is_deepspeed_zero3_enabled

    if not is_deepspeed_zero3_enabled():
        yield
        return

    import deepspeed

    params_to_gather = module.parameters()
    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=modifier_rank):
        yield
    return


def dequantize_bnb_weight(weight: torch.nn.Parameter, state=None):
    """
    Helper function to dequantize 4bit or 8bit bnb weights.

    If the weight is not a bnb quantized weight, it will be returned as is.
    """
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError(f"Input weight should be of type nn.Parameter, got {type(weight)} instead")

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    import bitsandbytes as bnb

    if cls_name == "Params4bit":
        return bnb.functional.dequantize_4bit(weight.data, weight.quant_state)

    if state.SCB is None:
        state.SCB = weight.SCB

    im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, "col32")
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    return bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()

class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector: Optional[torch.nn.ParameterDict] = None  # for DoRA
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def loftq_init(self, adapter_name):
        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def dora_init(self, adapter_name: str) -> None:
        lora_A = self.lora_A[adapter_name]
        lora_B = self.lora_B[adapter_name]
        scaling = self.scaling[adapter_name]
        with gather_params_ctx(self.get_base_layer()):
            weight = self.get_base_layer().weight
            quant_state = getattr(self.get_base_layer(), "state", None)
            weight = dequantize_bnb_weight(weight, state=quant_state)  # no-op if not bnb
            if weight.data.ndim == 4:  # For handling LoRAs applied to Conv2Ds.
                lora_weight = torch.mm(lora_B.weight.flatten(start_dim=1), lora_A.weight.flatten(start_dim=1))
                lora_weight = lora_weight.reshape(weight.shape)
            else:
                lora_weight = lora_B.weight @ lora_A.weight
            weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
        self.lora_magnitude_vector = nn.ParameterDict()
        self.lora_magnitude_vector[adapter_name] = nn.Parameter(weight_norm, requires_grad=True)
        # add lora_magnitude_vector to the list of learnable parameters
        self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def _apply_dora(self, x, lora_A, lora_B, scaling, active_adapter):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        lora_weight = lora_B.weight @ lora_A.weight
        magnitude = self.lora_magnitude_vector[active_adapter]
        weight = self.get_base_layer().weight
        quant_state = getattr(self.get_base_layer(), "state", None)
        weight = dequantize_bnb_weight(weight, state=quant_state)  # no-op if not bnb
        weight = weight.to(x.dtype)
        weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        result_dora = (mag_norm_scale - 1) * (
            F.linear(x, transpose(weight, self.fan_in_fan_out))
        ) + mag_norm_scale * lora_B(lora_A(x)) * scaling

        # Note: Computation could potentially be accelerated by using the code below instead of calculating X@W again.
        # This is only correct if dropout=0, otherwise results will differ:
        # https://github.com/huggingface/peft/pull/1474#issuecomment-1964682771
        # bias = self.get_base_layer().bias
        # if bias is not None:
        #     result = result - bias
        # result = mag_norm_scale * result + mag_norm_scale * lora_B(lora_A(x)) * scaling
        # if bias is not None:
        #     result = result + bias

        return result_dora

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        unique_adapters = set(self.active_adapters)
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)

            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(orig_weights, delta_weight, scaling=1).detach()
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        orig_weights = dora_factor.view(-1, 1) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(base_layer.weight, delta_weight, scaling=1).detach()
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        new_weight = dora_factor.view(-1, 1) * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            assert type(dropout) == nn.Indentity, "kernel assumes dropout is not contained in the lora layer"
            scaling = self.scaling[active_adapter]
            result_dtype = x.dtype
            x = x.to(lora_A.weight.dtype)

            result = apply_lora_w(x, self.base_layer.weight, lora_A.weight, lora_B.weight, scaling)

        result = result.to(result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    else:
        warnings.warn(
                "Only Linear LoRa is currently supported"
            )

    return new_module