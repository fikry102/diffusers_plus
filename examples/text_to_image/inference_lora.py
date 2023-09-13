from diffusers import StableDiffusionPipeline
import torch

MODEL_NAME="../../stable-diffusion-v1-4"
model_path = "./sd-pokemon-model-lora/"
pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("generation/pokemon_lora.png")