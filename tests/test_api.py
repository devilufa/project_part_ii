from fastapi.testclient import TestClient
from app.main import app
import os
from unittest.mock import patch

client = TestClient(app)

def mock_generate_image(*args, **kwargs):
    return b"fake_image_data"

@patch("app.utils.generate_image", return_value=mock_generate_image())
def test_generate_image_success(mock_gen):
    payload = {
        "prompt": "test image",
        "width": 512,
        "height": 512,
        "steps": 25,
        "cfg_scale": 7.0
    }
    response = client.post("/generate/", json=payload)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert b"fake_image_data" in response.content

def test_invalid_input():
    payload = {
        "prompt": "",
        "width": 2000, 
        "height": 512,
        "steps": 5,   
        "cfg_scale": 25.0
    }
    response = client.post("/generate/", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("less than or equal to 1024" in e["msg"] for e in errors)
    assert any("greater than or equal to 10" in e["msg"] for e in errors)

def test_healthcheck():
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "FLUX.1-dev"
    }
