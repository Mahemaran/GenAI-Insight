import torch
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion model
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# Define the prompt
prompt = "A majestic elephant standing beside a peaceful monk, in a misty forest, cinematic lighting"

# Generate the image
image = pipeline(prompt).images[0]

# Save and display the image
# image.save("elephant_with_monk.png")
image.show()