from functools import lru_cache

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import Settings, get_settings
from app.services.gradio_vqa import GradioVQAClient


app = FastAPI(title="Vietnamese Medicinal Herb VQA Demo")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@lru_cache
def get_vqa_client() -> GradioVQAClient:
    return GradioVQAClient(get_settings())


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, settings: Settings = Depends(get_settings)):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "model_name": settings.model_name,
            "gradio_api_url": settings.gradio_api_url,
        },
    )


@app.get("/health")
async def health(settings: Settings = Depends(get_settings)):
    return {
        "ok": True,
        "model": settings.model_name,
        "gradio_configured": bool(settings.gradio_api_url),
    }


@app.post("/api/v1/predict")
async def predict(
    image: UploadFile = File(...),
    question: str = Form(...),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Vui lòng upload một file ảnh hợp lệ.")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi.")

    try:
        client = get_vqa_client()
        return await client.predict(image, question.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Không gọi được Gradio API: {exc}") from exc
