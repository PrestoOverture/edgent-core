"""OpenAI-compatible VLM backend (works with MiMO, Qwen-VL, GPT-4V, etc.)."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

from ..types import ImageInput


class OpenAIVLMBackend:
    """VLM backend that calls any OpenAI-compatible vision API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        name: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.name = name or model
        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, image: ImageInput, query: str, **kwargs: object) -> str:
        """Send image + query to the VLM and return the text answer."""

        image_b64 = self._encode_image(image)
        mime = self._guess_mime(image)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{image_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": query,
                        },
                    ],
                }
            ],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _encode_image(image: ImageInput) -> str:
        """Read bytes from ImageInput and return base64 string."""

        if image.data is not None:
            raw = image.data
        elif image.path is not None:
            raw = Path(image.path).read_bytes()
        else:
            raise ValueError("ImageInput has neither data nor path.")
        return base64.b64encode(raw).decode("utf-8")

    @staticmethod
    def _guess_mime(image: ImageInput) -> str:
        """Best-effort MIME type from the file path, default to jpeg."""

        if image.path:
            mime, _ = mimetypes.guess_type(image.path)
            if mime:
                return mime
        return "image/jpeg"