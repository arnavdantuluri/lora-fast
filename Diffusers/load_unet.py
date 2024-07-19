from unet_pt import UNet2DConditionModel as UNet2DConditionModelPT
import torch
from collections import namedtuple
from PIL import Image, ImageChops

# Load weights from the original model
from diffusers import DiffusionPipeline
import sys
from datetime import datetime
from dataclasses import dataclass
from term_image import print_image
from hgemm import replace_linear
from sfast.compilers.diffusion_pipeline_compiler import compile_unet

sys.setrecursionlimit(10000)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True

class CompilationConfig:

    @dataclass
    class Default:
        '''
        Default compilation config

        memory_format:
            channels_last if tensor core is available, otherwise contiguous_format.
            On GPUs with tensor core, channels_last is faster
        enable_jit:
            Whether to enable JIT, most optimizations are done with JIT
        enable_jit_freeze:
            Whether to freeze the model after JIT tracing.
            Freezing the model will enable us to optimize the model further.
        preserve_parameters:
            Whether to preserve parameters when freezing the model.
            If True, parameters will be preserved, but the model will be a bit slower.
            If False, parameters will be marked as constants, and the model will be faster.
            However, if parameters are not preserved, LoRA cannot be switched dynamically.
        enable_cnn_optimization:
            Whether to enable CNN optimization by fusion.
        enable_fused_linear_geglu:
            Whether to enable fused Linear-GEGLU kernel.
            It uses fp16 for accumulation, so could cause **quality degradation**.
        prefer_lowp_gemm:
            Whether to prefer low-precision GEMM and a series of fusion optimizations.
            This will make the model faster, but may cause numerical issues.
            These use fp16 for accumulation, so could cause **quality degradation**.
        enable_xformers:
            Whether to enable xformers and hijack it to make it compatible with JIT tracing.
        enable_cuda_graph:
            Whether to enable CUDA graph. CUDA Graph will significantly speed up the model,
            by reducing the overhead of CUDA kernel launch, memory allocation, etc.
            However, it will also increase the memory usage.
            Our implementation of CUDA graph supports dynamic shape by caching graphs of
            different shapes.
        enable_triton:
            Whether to enable Triton generated CUDA kernels.
            Triton generated CUDA kernels are faster than PyTorch's CUDA kernels.
            However, Triton has a lot of bugs, and can increase the CPU overhead,
            though the overhead can be reduced by enabling CUDA graph.
        trace_scheduler:
            Whether to trace the scheduler.
        '''
        memory_format: torch.memory_format = (
            torch.channels_last)
        enable_jit: bool = True
        enable_jit_freeze: bool = True
        preserve_parameters: bool = True
        enable_cnn_optimization: bool = False
        enable_fused_linear_geglu: bool = False
        prefer_lowp_gemm: bool = True
        enable_xformers: bool = False
        enable_cuda_graph: bool = False
        enable_triton: bool = False
        trace_scheduler: bool = False

comp_config = CompilationConfig.Default()

def optimize_model(model: torch.nn.Module):
    return 

pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
).to("cuda")
unet = pipe.unet

unet_new = UNet2DConditionModelPT().half().cuda()
unet_new.load_state_dict(pipe.unet.state_dict())

# 'We need to include some extra unet methodologies since other methods in sdxl pipeline are dependent on it'
unet_new.config = namedtuple(
        "config", "in_channels addition_time_embed_dim sample_size"
        )
unet_new.config.in_channels = 4
unet_new.config.addition_time_embed_dim = 256
unet_new.config.sample_size = 128

prompt = "(masterpiece, best quality),ultra detailed,cinematic lighting,HDR,ilustration,horror,ChineseDragon,a majestic dragon soaring through a vibrant mystical forest, leaves ablaze with autumn colors. Enormous, detailed, fierce, mythical creature, sharp scales, brilliant fiery breath, green foliage, atmospheric lighting."
# pipe.do_classifier_free_guidance = True
# image = pipe(prompt).images[0]

fx_model = torch.fx.symbolic_trace(unet_new)
replace_linear(fx_model)
del unet_new

# pipe.unet = fx_model
# image = pipe(prompt).images[0]
sample = torch.rand(2, 4, 128, 128).cuda().half().requires_grad_(True)
timesteps = torch.rand([]).cuda()
encoder_hidden_states = torch.rand(2, 77, 2048).cuda().half()
added_cond_kwargs = {
    'text_embeds': torch.rand(2, 1280).cuda().half(),
    'time_ids': torch.rand(2, 6).cuda().half(),
}

unet = compile_unet(unet, comp_config)

start = datetime.now()
out = unet(sample, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
print(datetime.now() - start)
del out
torch.cuda.empty_cache()


start = datetime.now()
out = fx_model(sample, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
print(datetime.now() - start)




# dout = torch.rand_like(out.sample)
# out.sample.backward(dout)

# sample_pt_grad = sample.grad.clone() 
# assert sample_pt_grad is not None

# compile model and recalculate grads

# image.save("img.png")
# print_image(image, max_width=32)