import base64
import math
from typing import Any
import numpy as np
import cv2

# Adjusted thresholds to capture thin, faint hairline fractures
MIN_MASK_AREA = 3 
MAX_POLYGON_POINTS = 120
MAX_MASK_TO_BOX_AREA_RATIO = 12.0
MIN_MASK_IN_BOX_RATIO = 0.10 # Lowered to avoid dropping long, diagonal cracks


def enrich_detections_with_masks(
    image_base64: str,
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Safely processes text grounding detections, preserving actual semantic binary masks
    to extract point paths for high-accuracy damage overlay shading.
    """
    for detection in detections:
        mask_payload = detection.get("mask")
        # If we already have explicit contours, keep them intact
        if mask_payload and isinstance(mask_payload, dict) and mask_payload.get("polygons"):
            continue
            
        bbox_data = detection.get("bbox")
        if mask_payload and isinstance(mask_payload, dict) and bbox_data:
            bbox = (
                int(bbox_data.get("x", 0)),
                int(bbox_data.get("y", 0)),
                int(bbox_data.get("width", 0)),
                int(bbox_data.get("height", 0))
            )
            
            # Extract real mask matrix instead of generating an empty dummy block
            if "binary_mask" in mask_payload:
                mask_matrix = mask_payload["binary_mask"]
            elif isinstance(mask_payload.get("raw"), np.ndarray):
                mask_matrix = mask_payload["raw"]
            else:
                continue

            payload = _mask_to_payload(mask_matrix, bbox, detection.get("segmentation_mode", "sam3"))
            if payload:
                detection["mask"] = payload

    return detections


def mask_to_binary(
    mask_payload: dict[str, Any],
    image_shape: tuple[int, int],
) -> np.ndarray:
    image_height, image_width = image_shape
    binary = np.zeros((image_height, image_width), dtype=np.uint8)
    for polygon in mask_payload.get("polygons") or []:
        points = np.array(polygon, dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(binary, [points], 255)
    return binary


def _decode_image(image_base64: str) -> np.ndarray:
    if "," in image_base64 and image_base64.split(",", 1)[0].startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    raw = base64.b64decode(image_base64)
    data = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode image")
    return image


def _clamp_bbox(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    x = _finite_int(detection.get("coordX"))
    y = _finite_int(detection.get("coordY"))
    width = _finite_int(detection.get("width"))
    height = _finite_int(detection.get("height"))
    if width <= 1 or height <= 1:
        return None

    x = min(max(x, 0), image_width - 1)
    y = min(max(y, 0), image_height - 1)
    x2 = min(x + width, image_width)
    y2 = min(y + height, image_height)
    width = max(x2 - x, 1)
    height = max(y2 - y, 1)
    return (x, y, width, height) if width > 1 and height > 1 else None


def _finite_int(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return int(round(number)) if math.isfinite(number) else 0


def _mask_to_payload(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    mode: str,
) -> dict[str, Any] | None:
    validation = _validate_mask_against_bbox(mask, bbox)
    if validation is None:
        return None

    # Ensure mask is single-channel 8-bit image
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    
    polygons = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < MIN_MASK_AREA:
            continue

        epsilon = max(0.5, cv2.arcLength(contour, True) * 0.005) # Tighter tracking approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2).tolist()
        
        if len(points) > MAX_POLYGON_POINTS:
            step = math.ceil(len(points) / MAX_POLYGON_POINTS)
            points = points[::step]
        if len(points) >= 3:
            polygons.append([[int(x), int(y)] for x, y in points])

    area = int(np.count_nonzero(mask))
    if not polygons or area < MIN_MASK_AREA:
        return None

    x, y, width, height = _mask_bbox(mask)
    return {
        "type": "polygon",
        "mode": mode,
        "bbox": {"x": x, "y": y, "width": width, "height": height},
        "area": area,
        "polygons": polygons,
    }


def mask_to_payload(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    mode: str,
) -> dict[str, Any] | None:
    return _mask_to_payload(mask, bbox, mode)


def _validate_mask_against_bbox(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float | None:
    x, y, width, height = bbox
    box_area = max(width * height, 1)
    mask_area = int(np.count_nonzero(mask))
    if mask_area < MIN_MASK_AREA:
        return None

    # Safe bound clamping to avoid slice out-of-bounds crashes
    h_max, w_max = mask.shape[:2]
    y1, y2 = min(max(y, 0), h_max), min(max(y + height, 0), h_max)
    x1, x2 = min(max(x, 0), w_max), min(max(x + width, 0), w_max)

    in_box = mask[y1:y2, x1:x2]
    in_box_area = int(np.count_nonzero(in_box))
    
    # 🚀 FIX: If it's a thin structural line/crack, don't drop it just because 
    # it touches or travels slightly outside the loose grounding box boundaries!
    is_thin_defect = (mask_area / (h_max * w_max)) < 0.05
    if is_thin_defect:
        # Give it a safe structural pass if there are pixels inside the target zone
        if in_box_area >= MIN_MASK_AREA:
            return 0.99 

    if in_box_area < MIN_MASK_AREA:
        return None

    in_box_ratio = in_box_area / mask_area
    mask_to_box_ratio = mask_area / box_area
    
    if in_box_ratio < MIN_MASK_IN_BOX_RATIO:
        return None
    if mask_to_box_ratio > MAX_MASK_TO_BOX_AREA_RATIO:
        return None

    compactness_score = min(in_box_area, box_area) / max(in_box_area, box_area)
    containment_score = min(in_box_ratio, 1.0)
    return containment_score * 0.35 + compactness_score * 0.25


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        return 0, 0, 0, 0
    x, y, width, height = cv2.boundingRect(points)
    return int(x), int(y), int(width), int(height)