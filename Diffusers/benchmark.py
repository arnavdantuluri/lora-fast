from Diffusers.unet_pt import UNet2DConditionModel as UNet2DConditionModelPT
import torch
from collections import namedtuple
from PIL import Image, ImageChops

# Load weights from the original model
from diffusers import DiffusionPipeline
import sys
from datetime import datetime
from Diffusers.term_image import print_image
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)

sys.setrecursionlimit(10000)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True

pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
).to("cuda")

config = CompilationConfig.Default()
#baseline 4.18
config.enable_jit = True #3.86
config.enable_jit_freeze = False #3.66
config.enable_cnn_optimization = False #
config.enable_fused_linear_geglu = False
config.prefer_lowp_gemm = False
config.enable_xformers = True
config.enable_cuda_graph = False
config.enable_triton = False
config.trace_scheduler = False

prompt = "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail, 4k render."

image = pipe(prompt).images[0]

model = compile(pipe, config)

for _ in range(3):
    image = pipe(prompt).images[0]

# unet = pipe.unet

# sample = torch.rand(2, 4, 128, 128).cuda().half().requires_grad_(True)
# timesteps = torch.rand([]).cuda()
# encoder_hidden_states = torch.rand(2, 77, 2048).cuda().half()
# added_cond_kwargs = {
#     'text_embeds': torch.rand(2, 1280).cuda().half(),
#     'time_ids': torch.rand(2, 6).cuda().half(),
# }

# out = unet(sample, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)

# dout = torch.rand_like(out.sample)
# out.sample.backward(dout)

# sample_pt_grad = sample.grad.clone() 
# assert sample_pt_grad is not None

# # compile model and recalculate grads

image.save("img.png")
print_image(image, max_width=32)