from diffusers import StableDiffusionPipeline
import torch

# model_path = "path_to_saved_model"
model_path="./sd-pokemon-model/"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="yoda").images[0]
image.save("generation/yoda-pokemon.png")