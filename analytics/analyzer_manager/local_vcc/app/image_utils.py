import base64
import io
import os
from typing import Iterable, List

from PIL import Image

from app.logger import logger


def _encode_image_to_data_url(path: str) -> str:
    """Load an image from disk and return a data:image/jpeg;base64,... URL."""
    with Image.open(path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        with io.BytesIO() as bio:
            img.save(bio, format="JPEG")
            payload = bio.getvalue()
    b64 = base64.b64encode(payload).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def load_images(paths: Iterable[str]) -> List[str]:
    """Return a list of base64 data URLs for the given filesystem paths.

    Missing / unreadable images are logged and skipped.
    """
    urls: List[str] = []
    for path in paths or []:
        if not path:
            continue
        if not os.path.isfile(path):
            logger.warning(f"Image path does not exist: {path}")
            continue
        try:
            urls.append(_encode_image_to_data_url(path))
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
    return urls
