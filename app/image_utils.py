import base64
import io
import os
from typing import Any

from PIL import Image, ImageStat, UnidentifiedImageError

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pillow_avif = None


def decode_and_resize_image(
    image_base64: str,
    max_side: int = 1280,
    max_mb: int = 12,
    min_side: int | None = None,
) -> tuple[str, tuple[int, int], dict[str, Any]]:
    if "," in image_base64 and image_base64.split(",", 1)[0].startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    raw = base64.b64decode(image_base64, validate=True)

    if len(raw) > max_mb * 1024 * 1024:
        raise ValueError(f"Image too large. Max allowed is {max_mb}MB.")

    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        magic = raw[:16].hex()
        raise ValueError(
            "Unsupported or invalid image data. "
            f"First 16 bytes: {magic}. "
        ) from exc

    original_size = img.size
    width, height = img.size
    longest_side = max(width, height)
    min_side = min_side if min_side is not None else int(os.getenv("MIN_IMAGE_SIDE", "768"))
    scale_down = min(max_side / longest_side, 1.0)
    scale_up = max(min_side / longest_side, 1.0) if longest_side else 1.0
    scale = scale_down if scale_down < 1.0 else scale_up

    if scale != 1.0:
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=90)

    encoded = base64.b64encode(output.getvalue()).decode("utf-8")
    metadata = _preprocessing_metadata(
        original_size=original_size,
        processed_size=img.size,
        scale=scale,
        image=img,
    )
    return encoded, img.size, metadata


def _preprocessing_metadata(
    *,
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
    scale: float,
    image: Image.Image,
) -> dict[str, Any]:
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    brightness = float(stat.mean[0]) if stat.mean else 0.0
    contrast = float(stat.stddev[0]) if stat.stddev else 0.0
    width, height = processed_size
    original_longest = max(original_size)
    flags: list[str] = []
    if original_longest < int(os.getenv("MIN_IMAGE_SIDE", "768")):
        flags.append("low_resolution")
    if brightness < 35:
        flags.append("dark")
    if brightness > 230:
        flags.append("overexposed")
    if contrast < 18:
        flags.append("low_contrast")

    return {
        "originalSize": {"width": original_size[0], "height": original_size[1]},
        "processedSize": {"width": width, "height": height},
        "scaleFactor": round(scale, 4),
        "upscaled": scale > 1.0,
        "downscaled": scale < 1.0,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "qualityFlags": flags,
    }


def save_base64_image(image_base64: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = base64.b64decode(image_base64)
    with open(path, "wb") as f:
        f.write(raw)
