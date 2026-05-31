"""Tests for OpenAIVLMBackend image encoding cache."""

from __future__ import annotations

from edgent_core import ImageInput
from edgent_core.backend.openai_vlm import OpenAIVLMBackend


def test_vlm_image_cache_hit_by_path(tmp_path) -> None:
    """Same path → cache hit, only one encode."""
    img_file = tmp_path / "test.jpg"
    img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header

    backend = OpenAIVLMBackend.__new__(OpenAIVLMBackend)
    backend._image_cache = {}

    img1 = ImageInput(path=str(img_file))
    img2 = ImageInput(path=str(img_file))

    b64_1, mime1 = backend._encode_image_cached(img1)
    b64_2, mime2 = backend._encode_image_cached(img2)

    # Same result
    assert b64_1 == b64_2
    assert mime1 == mime2
    # Cache has exactly one entry
    assert len(backend._image_cache) == 1


def test_vlm_image_no_cache_when_path_none() -> None:
    """path=None → no caching, re-encodes every time."""
    backend = OpenAIVLMBackend.__new__(OpenAIVLMBackend)
    backend._image_cache = {}

    img = ImageInput(data=b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    b64_1, _ = backend._encode_image_cached(img)
    b64_2, _ = backend._encode_image_cached(img)

    # Same result (same data)
    assert b64_1 == b64_2
    # No cache entries because path is None
    assert len(backend._image_cache) == 0
