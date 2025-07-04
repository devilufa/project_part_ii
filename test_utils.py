from app.utils import generate_image
from unittest.mock import patch
import pytest
import os

@patch("huggingface_hub.InferenceClient.text_to_image")
def test_generate_image_success(mock_text_to_image):
    mock_text_to_image.return_value.save.return_value = None
    
    image_bytes = generate_image("test prompt")
    
    assert isinstance(image_bytes, bytes)
    assert len(image_bytes) > 0

def test_missing_api_key():
    original_key = os.environ.get("HF_API_KEY")
    if "HF_API_KEY" in os.environ:
        del os.environ["HF_API_KEY"]
    
    with pytest.raises(ValueError) as excinfo:
        generate_image("test")
    assert "HF_API_KEY" in str(excinfo.value)
    
    if original_key:
        os.environ["HF_API_KEY"] = original_key
