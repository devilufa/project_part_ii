import os
from pathlib import Path
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("HF_API_KEY")

if not api_key:
    print("❌ ERROR: HF_API_KEY environment variable is not set!")
    print("Please create a .env file with HF_API_KEY=your_token")
    exit(1)

print(f"✅ API Key loaded: {api_key[:4]}...{api_key[-4:]}")

try:
    client = InferenceClient(token=api_key, timeout=60)
    print("🚀 Sending request to Hugging Face...")
    
    image = client.text_to_image(
        "a cute cat",
        model="black-forest-labs/FLUX.1-dev",
        width=256,
        height=256,
        guidance_scale=5.0,
        num_inference_steps=15
    )
    
    image.save("test_image.png")
    print("🎉 Image saved successfully as 'test_image.png'")
    print("🖼️ Please check the generated image")
    
except Exception as e:
    print(f"🔥 Error: {str(e)}")
    print("\nTroubleshooting steps:")
    print("1. Check your internet connection")
    print("2. Verify your Hugging Face token is valid")
    print("3. Check model availability: https://huggingface.co/black-forest-labs/FLUX.1-dev")
    print("4. Try smaller image size (e.g. 256x256)")
