import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the pre-trained Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v-1-4-original")

# Move the model to GPU if available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

# Streamlit App
st.title("Text-to-Image Generation with Stable Diffusion")
st.write("Enter a text prompt to generate an image.")

# User Input for the text prompt
prompt = st.text_input("Enter your prompt", "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Generate the image from the model
        image = pipe(prompt).images[0]

        # Display the generated image
        st.image(image, caption="Generated Image", use_column_width=True)

        # Save the image
        image.save("generated_image.png")
        st.success("Image generated and saved as 'generated_image.png'.")
