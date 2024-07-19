from lora.lora_kernel import apply_lora_o
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from diffusers import UNet2DConditionModel
import triton
import triton.language as tl

adapter_config = LoraConfig(
    r=16,
    lora_alpha=16,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet"
)


#Extract attention layer from unet
for layer in unet.down_blocks:
    for x in layer.attentions:
        for y in x.transformer_blocks:
            model = y.attn1
            break
        break
    break

model.requires_grad_(False)
inject_adapter_in_model(adapter_config, model, "test")

def model_foward(model, x):
    q = model.to_q(x)
    k = model.to_k(x)
    v = model.to_v(x)
    return q, k, v

def w_forward(model, x):
    q = apply_lora_o(model.to_q, x)
    k = apply_lora_o(model.to_k, x)
    v = apply_lora_o(model.to_v, x)
    return q, k, v

x = torch.rand(1, 320, 320, dtype=torch.float16, device='cuda').requires_grad_(True)
model = model.cuda().half()

torch_q, torch_k, torch_v = model_foward(model, x)
x_pt_grad, x.grad = x.grad.clone(), None

wq, wk, wv = w_forward(model, x)
x_tri_grad, x.grad = x.grad.clone(), None

assert (torch.stack([torch_q, torch_k, torch_v]) - torch.stack([wq, wk, wv])).all() < 1e-2, "Assertion does not hold, some issue in the triton kernel"
assert (x_pt_grad - x_tri_grad).all() < 1e-2, "Assertion does not hold, some issue in the triton kernel"