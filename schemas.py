from pydantic import BaseModel, Field

class ImageRequest(BaseModel):
    prompt: str = Field(..., example="Astronaut riding a horse")
    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)
    steps: int = Field(25, ge=10, le=100)
    cfg_scale: float = Field(7.0, ge=1.0, le=20.0)
