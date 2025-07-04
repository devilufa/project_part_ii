from huggingface_hub import InferenceClient
from PIL import Image
import io
import os

def generate_image(
    prompt: str,
    width: int = 512,
    height: int = 512,
    steps: int = 25,
    cfg_scale: float = 7.0
) -> bytes:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise ValueError("HF_API_KEY not set in environment")
    
    client = InferenceClient(token=api_key)
    
    image = client.text_to_image(
        prompt,
        model="black-forest-labs/FLUX.1-dev",
        width=width,
        height=height,
        guidance_scale=cfg_scale,
        num_inference_steps=steps
    )
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()
