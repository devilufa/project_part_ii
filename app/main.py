from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas import ImageRequest  
from app.utils import generate_image 
import io

app = FastAPI(
    title="Text-to-Image Generator API",
    description="Генерация изображений по текстовым описаниям с использованием FLUX.1-dev"
)

@app.post("/generate/", response_class=StreamingResponse)
async def generate_image_endpoint(request: ImageRequest):
    try:
        image_bytes = generate_image(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg_scale=request.cfg_scale
        )
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=generated.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "model": "FLUX.1-dev"}
