from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import sys

# パイプラインの準備
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    scheduler=EulerDiscreteScheduler.from_pretrained(
        model_id, 
        subfolder="scheduler"
    ), 
    torch_dtype=torch.float16
).to("cuda")
pipe.enable_attention_slicing()

# 日本語テキストから翻訳
from deep_translator import GoogleTranslator
ja_text = "「……どうして……どうして僕ばかり……こんなにも辛いのに……」"
#ja_text = "ラインドローイング、影無し、白黒、教会"

en_text = GoogleTranslator(source='auto',target='en').translate(ja_text)
print(en_text)
file_name = en_text.replace(' ','_')
# 画像生成（入力は英語）
prompt = en_text
image = pipe(prompt, height=768, width=768).images[0]

image.save(f'output/3D/{file_name}.png')
