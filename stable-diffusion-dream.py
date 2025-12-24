import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# StableDiffusionパイプラインの準備
pipe = StableDiffusionPipeline.from_pretrained(
    "./diffusers/examples/dreambooth/model/data/400",
    torch_dtype=torch.float16
).to("cuda")

from torch import autocast

# テキストからの画像生成
prompt = "photo of sks cat" 
with autocast("cuda"):
    images = pipe(prompt, guidance_scale=7.5).images
images[0].save("output/mochimaru.png")