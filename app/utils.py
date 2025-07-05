from huggingface_hub import InferenceClient
from PIL import Image
import io
import os
import logging

logger = logging.getLogger(__name__)

def generate_image(
    prompt: str,
    width: int = 512,
    height: int = 512,
    steps: int = 25,
    cfg_scale: float = 7.0
) -> bytes:
    """Генерация изображения через Hugging Face Hub"""
    try:
        logger.info(f"Generating image for prompt: '{prompt}'")
        api_key = os.getenv("HF_API_KEY")
        
        if not api_key:
            logger.error("HF_API_KEY environment variable is not set!")
            raise ValueError("HF_API_KEY not set in environment")
        
        logger.info("Initializing InferenceClient")
        client = InferenceClient(token=api_key)
        
        logger.info("Sending request to Hugging Face")
        image = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-dev",
            width=width,
            height=height,
            guidance_scale=cfg_scale,
            num_inference_steps=steps
        )
        
        logger.info("Image received, converting to bytes")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    except Exception as e:
        logger.exception("Error in generate_image function")
        raise
