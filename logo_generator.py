from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model only once
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_logo(business_name, description=""):
    prompt = f"Professional minimalist logo design for a company named '{business_name}'. {description}. Vector, clean, modern, white background"
    image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
    return image
