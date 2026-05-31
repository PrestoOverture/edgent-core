"""OpenAI-compatible VLM backend (works with MiMO, Qwen-VL, GPT-4V, etc.)."""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, OpenAI

from ..errors import BackendError
from ..types import ImageInput, Usage

# Max image dimension before resizing (API limit ~20MB base64 ≈ ~5MB raw)
_MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4MB raw, ~5.3MB base64


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
        request_timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.name = name or model
        self.model = model
        self.max_tokens = max_tokens
        self.last_usage: Usage = Usage()
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=request_timeout, max_retries=max_retries,
        )
        self._image_cache: dict[str, tuple[str, str]] = {}  # path -> (base64, mime)

    def generate(self, image: ImageInput, query: str, **kwargs: object) -> str:
        """Send image + query to the VLM and return the text answer."""

        image_b64, mime = self._encode_image_cached(image)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if not isinstance(max_tokens, int):
            max_tokens = self.max_tokens

        try:
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
                max_tokens=max_tokens,
            )
        except APITimeoutError as exc:
            raise BackendError(f"VLM request timed out: {exc}") from exc
        except APIConnectionError as exc:
            raise BackendError(f"VLM connection failed: {exc}") from exc
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
            )
        return response.choices[0].message.content or ""

    def _encode_image_cached(self, image: ImageInput) -> tuple[str, str]:
        """Return (base64, mime), using cache for path-based images."""

        cache_key = image.path
        if cache_key and cache_key in self._image_cache:
            return self._image_cache[cache_key]

        b64 = self._encode_image(image)
        mime = self._guess_mime(image)

        if cache_key:
            self._image_cache[cache_key] = (b64, mime)

        return b64, mime

    @staticmethod
    def _encode_image(image: ImageInput) -> str:
        """Read bytes from ImageInput, resize if too large, and return base64 string."""

        if image.data is not None:
            raw = image.data
        elif image.path is not None:
            raw = Path(image.path).read_bytes()
        else:
            raise ValueError("ImageInput has neither data nor path.")

        if len(raw) > _MAX_IMAGE_BYTES:
            raw = _resize_image(raw)

        return base64.b64encode(raw).decode("utf-8")

    @staticmethod
    def _guess_mime(image: ImageInput) -> str:
        """Best-effort MIME type from the file path, default to jpeg."""

        if image.path:
            mime, _ = mimetypes.guess_type(image.path)
            if mime:
                return mime
        return "image/jpeg"


def _resize_image(raw: bytes, max_bytes: int = _MAX_IMAGE_BYTES) -> bytes:
    """Resize an image to fit within max_bytes using Pillow."""

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        # Calculate scale factor to fit within max_bytes
        # Rough estimate: JPEG quality 85 ≈ 1 byte per 3 pixels
        current_size = len(raw)
        scale = (max_bytes / current_size) ** 0.5
        if scale >= 1.0:
            return raw

        new_width = max(int(img.width * scale), 64)
        new_height = max(int(img.height * scale), 64)
        img = img.resize((new_width, new_height), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        result = buf.getvalue()

        # If still too large, reduce quality
        if len(result) > max_bytes:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=60)
            result = buf.getvalue()

        return result
    except ImportError:
        # Pillow not available, return as-is
        return raw
