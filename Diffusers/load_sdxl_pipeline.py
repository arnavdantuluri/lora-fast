from unet_pt import UNet2DConditionModel as UNet2DConditionModelPT
import torch
from collections import namedtuple
from PIL import Image, ImageChops

# Load weights from the original model
from diffusers import DiffusionPipeline
import sys
from datetime import datetime
from term_image import print_image
from hgemm import replace_linear

sys.setrecursionlimit(10000)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True

def optimize_model(model: torch.nn.Module):
    return 

pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
).to("cuda")
unet = pipe.unet


prompt = "(masterpiece, best quality),ultra detailed,cinematic lighting,HDR,ilustration,horror,ChineseDragon,a majestic dragon soaring through a vibrant mystical forest, leaves ablaze with autumn colors. Enormous, detailed, fierce, mythical creature, sharp scales, brilliant fiery breath, green foliage, atmospheric lighting."
image = pipe(prompt).images[0]


image.save("img.png")
print_image(image, max_width=32)