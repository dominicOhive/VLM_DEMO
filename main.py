import os
import logging
import base64
import uuid
import cv2
import numpy as np
import time
from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

from app.schemas import HealthResponse
from app.grounding import warmup_vision_model, run_grounding_detection
from app.ollama_client import normalize_ollama_base_url
from app.queue_worker import (
    get_queue_worker_status,
    start_queue_worker,
    stop_queue_worker,
)
from app.visualization import render_detection_overlay

# 1. Environment Configurations
OLLAMA_BASE_URL = normalize_ollama_base_url(os.getenv("OLLAMA_BASE_URL"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b")
QWEN_PLANNER_BACKEND = os.getenv("QWEN_PLANNER_BACKEND", "ollama").lower()
QWEN_MODEL_ID = os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
SAM3_MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "app", "storage", "output")
os.environ["VISION_WARMUP_ON_STARTUP"] = "true"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("spatial_vision_worker.main")

# 2. App Initialization & Middleware Configuration
app = FastAPI(title="GPU Vision Worker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    logger.info("Starting GPU Vision Worker...")
    warmup_vision_model()
    logger.info("Starting queue worker...")
    start_queue_worker()


@app.on_event("shutdown")
async def shutdown() -> None:
    await stop_queue_worker()


@app.get("/health", response_model=HealthResponse)
async def health(response: Response):
    queue_status = get_queue_worker_status()
    if not queue_status["healthy"]:
        response.status_code = 503

    qwen_model = QWEN_MODEL_ID if QWEN_PLANNER_BACKEND == "transformers" else OLLAMA_MODEL
    return HealthResponse(
        status="ok" if queue_status["healthy"] else "degraded",
        model=f"qwen-{QWEN_PLANNER_BACKEND}:{qwen_model}+sam3:{SAM3_MODEL_ID}",
        queue=queue_status,
    )


# 3. Core Vision Pipeline Routing Endpoint
@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...), 
    prompt: str = Form("Find any cracks or dents")
):
    start_time = time.perf_counter()
    contents = await image.read()

    # 1. Decode incoming format into an OpenCV pixel matrix
    np_arr = np.frombuffer(contents, dtype=np.uint8)
    img_decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img_decoded is not None:
        height, width = img_decoded.shape[:2]
        
        # NEW: Downscale protection if the image is massive (e.g., 4K camera shots)
        max_side = 1024
        if max(height, width) > max_side:
            scale = max_side / max(height, width)
            new_width = int(round(width * scale))
            new_height = int(round(height * scale))
            img_decoded = cv2.resize(img_decoded, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = new_height, new_width

        image_dimensions = (width, height)
        
        # 2. Convert the matrix into a standard, clean JPEG byte array buffer
        _, encoded_buffer = cv2.imencode(".jpg", img_decoded)
        img_base64 = base64.b64encode(encoded_buffer).decode("utf-8")
    else:
        image_dimensions = (800, 800)
        img_base64 = base64.b64encode(contents).decode("utf-8")

    # 3. Call the grounding engine using the normalized, safe image stream
    parsed, _raw_text = await run_grounding_detection(
        image_base64=img_base64,
        image_size=image_dimensions,
        condition_name=prompt,
    )

    overlay_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.jpg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        if parsed.get("detections"):
            render_detection_overlay(img_base64, parsed.get("detections", []), overlay_path)
        else:
            image_bytes = base64.b64decode(img_base64)
            with open(overlay_path, "wb") as handle:
                handle.write(image_bytes)
        image_url = f"/results/{os.path.basename(overlay_path)}"
    except Exception as exc:
        logger.warning("Failed to render overlay image: %s", exc)
        image_bytes = base64.b64decode(img_base64)
        with open(overlay_path, "wb") as handle:
            handle.write(image_bytes)
        image_url = f"/results/{os.path.basename(overlay_path)}"

    execution_duration = round(time.perf_counter() - start_time, 2)
    logger.info("Image analysis completed locally in %s seconds", execution_duration)

    return {
        "result": parsed,
        "imageUrl": image_url,
        "annotatedImageUrl": image_url,
        "computationTime": execution_duration
    }

# 4. Storage & Frontend Rendering Routes
@app.get("/results/{filename}")
async def get_result_image(filename: str) -> FileResponse:
    image_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(filename)
    return FileResponse(image_path, media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html file not found on server.</h1>"