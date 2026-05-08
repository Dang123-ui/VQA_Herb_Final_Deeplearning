import tempfile
import time
from pathlib import Path

from fastapi import UploadFile
from gradio_client import Client, handle_file

from app.core.config import Settings


class GradioVQAClient:
    def __init__(self, settings: Settings):
        if not settings.gradio_api_url:
            raise RuntimeError("GRADIO_API_URL is not configured. Put your Colab Gradio URL in demo/.env.")
        self.settings = settings
        self.client = Client(settings.gradio_api_url)

    async def predict(self, image: UploadFile, question: str) -> dict:
        suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
        started = time.perf_counter()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = self.client.predict(
                handle_file(str(tmp_path)),
                question,
                api_name=self.settings.gradio_api_name,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        inference_ms = int((time.perf_counter() - started) * 1000)
        answer = self._extract_answer(result)
        return {
            "answer": answer,
            "inference_ms": inference_ms,
            "model": self.settings.model_name,
        }

    @staticmethod
    def _extract_answer(result) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            for key in ("answer", "prediction", "output", "text"):
                value = result.get(key)
                if isinstance(value, str):
                    return value.strip()
            return str(result)
        if isinstance(result, (list, tuple)) and result:
            return str(result[0]).strip()
        return str(result).strip()
