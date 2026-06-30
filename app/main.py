import os
import logging
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HealthResponse
from app.grounding import warmup_vision_model
from app.ollama_client import normalize_ollama_base_url
from app.queue_worker import (
    get_queue_worker_status,
    start_queue_worker,
    stop_queue_worker,
)


OLLAMA_BASE_URL = normalize_ollama_base_url(os.getenv("OLLAMA_BASE_URL"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b")
QWEN_PLANNER_BACKEND = os.getenv("QWEN_PLANNER_BACKEND", "ollama").lower()
QWEN_MODEL_ID = os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
SAM3_MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger("spatial_vision_worker.main")

app = FastAPI(title="GPU Vision Worker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
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
