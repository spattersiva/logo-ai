import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32
)

# Page setup
st.set_page_config(page_title="AI Logo Generator", page_icon="🎨", layout="centered")
st.title("🎨 AI Logo Generator")
st.write("Generate professional business logos instantly using AI!")

# Load model only once
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

pipe = load_model()

# UI inputs
business_name = st.text_input("📝 Enter Business Name:")
description = st.text_area("💡 Describe your business (optional):")

# Style options
style = st.selectbox(
    "🎨 Choose logo style:",
    ["Modern", "Luxury", "Minimalist", "Playful", "Tech", "Vintage"],
)

color = st.text_input("🎨 Preferred colors (optional):", "blue, white")

if st.button("✨ Generate Logo"):
    if not business_name.strip():
        st.warning("Please enter a business name.")
    else:
        # Create prompt
        prompt = (
            f"{style} logo design for a business named '{business_name}'. "
            f"{description}. Use colors: {color}. Vector, clean, modern, white background, high quality"
        )

        st.info("⏳ Generating your logo, please wait...")
        image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]

        # Display result
        st.image(image, caption=f"Logo for {business_name}", use_container_width=True)

        # Save image
        os.makedirs("assets", exist_ok=True)
        file_path = f"assets/{business_name}_logo.png"
        image.save(file_path)
        st.success(f"✅ Logo saved as {file_path}")

        # Download button
        with open(file_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Logo",
                data=f,
                file_name=f"{business_name}_logo.png",
                mime="image/png",
            )
