import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx
import redis.asyncio as redis

from app.image_utils import decode_and_resize_image, save_base64_image
from app.clustering import annotate_detection_clusters
from app.grounding import run_grounding_detection
from app.ollama_client import normalize_ollama_base_url
from app.schemas import VisionResult
from app.s3_uploader import upload_file_to_s3
from app.segmentation import enrich_detections_with_masks
from app.visualization import render_detection_overlay


OLLAMA_BASE_URL = normalize_ollama_base_url(os.getenv("OLLAMA_BASE_URL"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b")
QWEN_PLANNER_BACKEND = os.getenv("QWEN_PLANNER_BACKEND", "ollama").lower()
REDIS_URL = os.getenv("REDIS_URL") or os.getenv("VISION_REDIS_URL")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or os.getenv("REDIS_AUTH")
QUEUE_NAME = (
    os.getenv("SPATIAL_VISION_GEMINI_QUEUE_NAME")
    or os.getenv("SPATIAL_VISION_QUEUE_NAME")
    or "spatial-vision:jobs:gemini"
)
BACKEND_CALLBACK_URL = os.getenv("BACKEND_CALLBACK_URL")
CALLBACK_TOKEN = os.getenv("SPATIAL_VISION_CALLBACK_TOKEN", "")
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "12"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1024"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))
VISION_DEBUG = os.getenv("VISION_DEBUG", "false").lower() == "true"
REDIS_RECONNECT_MAX_SECONDS = int(os.getenv("REDIS_RECONNECT_MAX_SECONDS", "60"))

JOB_ITEM_ID_KEYS = ("jobItemId", "job_item_id")

def _get_job_item_id(job: dict[str, Any]) -> str | None:
    return job.get("jobItemId") or job.get("job_item_id")

_queue_task: asyncio.Task | None = None
_queue_state = "stopped"
_last_error: str | None = None
_last_heartbeat = 0.0
_redis_connected = False
_jobs_processed = 0
logger = logging.getLogger("spatial_vision_worker")


def start_queue_worker() -> None:
    global _queue_task, _queue_state, _last_error
    if _queue_task:
        return
    if not REDIS_URL:
        _queue_state = "disabled"
        _last_error = "REDIS_URL is not set"
        logger.warning("Spatial Vision queue worker disabled: REDIS_URL is not set")
        return
    if not BACKEND_CALLBACK_URL:
        _queue_state = "disabled"
        _last_error = "BACKEND_CALLBACK_URL is not set"
        logger.warning(
            "Spatial Vision queue worker disabled: BACKEND_CALLBACK_URL is not set"
        )
        return
    logger.info("Starting Spatial Vision queue worker on queue %s", QUEUE_NAME)
    _queue_state = "starting"
    _last_error = None
    _queue_task = asyncio.create_task(_run_queue_loop())


async def stop_queue_worker() -> None:
    global _queue_task, _queue_state, _redis_connected
    if not _queue_task:
        return
    _queue_state = "stopping"
    _queue_task.cancel()
    try:
        await _queue_task
    except asyncio.CancelledError:
        pass
    _queue_task = None
    _redis_connected = False
    _queue_state = "stopped"


def get_queue_worker_status() -> dict[str, Any]:
    task_done = bool(_queue_task and _queue_task.done())
    task_error = None
    if task_done and _queue_task and not _queue_task.cancelled():
        try:
            exception = _queue_task.exception()
        except Exception as exc:
            exception = exc
        if exception:
            task_error = str(exception)

    running = bool(_queue_task and not _queue_task.done())
    configured = bool(REDIS_URL and BACKEND_CALLBACK_URL)
    now = time.monotonic()
    heartbeat_age = round(now - _last_heartbeat, 3) if _last_heartbeat else None

    return {
        "healthy": configured and running and not task_done and _redis_connected,
        "running": running,
        "state": _queue_state,
        "queueName": QUEUE_NAME,
        "redisConfigured": bool(REDIS_URL),
        "callbackConfigured": bool(BACKEND_CALLBACK_URL),
        "redisConnected": _redis_connected,
        "jobsProcessed": _jobs_processed,
        "lastHeartbeatSecondsAgo": heartbeat_age,
        "lastError": task_error or _last_error,
    }


async def _run_queue_loop() -> None:
    global _queue_state, _last_error, _last_heartbeat, _redis_connected
    backoff_seconds = 1
    while True:
        client = None
        try:
            _queue_state = "connecting"
            _last_heartbeat = time.monotonic()
            logger.info("Connecting to Redis at %s", REDIS_URL)
            client = redis.from_url(
                REDIS_URL,
                password=REDIS_PASSWORD,
                decode_responses=True,
            )
            await client.ping()
            _redis_connected = True
            _queue_state = "waiting"
            _last_error = None
            backoff_seconds = 1
            logger.info("Connected to Redis. Waiting for jobs on %s", QUEUE_NAME)
            while True:
                _last_heartbeat = time.monotonic()
                item = await client.blpop(QUEUE_NAME, timeout=5)
                if not item:
                    continue

                _, raw_job = item
                logger.info(
                    "Received Spatial Vision job payload (%s bytes)", len(raw_job)
                )
                await _handle_raw_job(raw_job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _redis_connected = False
            _queue_state = "reconnecting"
            _last_error = str(exc)
            logger.exception(
                "Spatial Vision queue loop failed; reconnecting in %ss",
                backoff_seconds,
            )
            await asyncio.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, REDIS_RECONNECT_MAX_SECONDS)
        finally:
            if client:
                _redis_connected = False
                logger.info("Closing Redis connection")
                await client.aclose()


async def _handle_raw_job(raw_job: str) -> None:
    global _queue_state, _last_error, _jobs_processed
    _queue_state = "processing"
    try:
        await _process_job(json.loads(raw_job))
        _jobs_processed += 1
        _last_error = None
    except Exception as exc:
        _last_error = str(exc)
        logger.exception("Spatial Vision job failed")
        await _send_failure_callback(raw_job, exc)
    finally:
        _queue_state = "waiting"


async def _send_failure_callback(raw_job: str, exc: Exception) -> None:
    try:
        job = json.loads(raw_job)
    except Exception:
        job = {}
    try:
        await _send_callback(
            {
                "jobId": job.get("jobId"),
                "jobItemId": _get_job_item_id(job),
                "sessionId": job.get("sessionId"),
                "status": "failed",
                "imageUrl": job.get("imageUrl"),
                "error": str(exc),
                "detections": [],
            }
        )
    except Exception:
        logger.exception("Failed to send Spatial Vision failure callback")


async def _process_job(job: dict[str, Any]) -> None:
    logger.info(
        "Processing job=%s session=%s steps=%s prompt=%s imageUrl=%s",
        job.get("jobId"),
        job.get("sessionId"),
        len(job.get("steps", [])),
        _preview(job.get("prompt"), 240),
        job.get("imageUrl"),
    )
    _debug("Job payload steps=%s", job.get("steps", []))
    detections = []
    image_base64 = job["imageBase64"]
    image_b64, resized_size, preprocessing = decode_and_resize_image(
        image_base64,
        max_side=MAX_IMAGE_SIDE,
        max_mb=MAX_IMAGE_MB,
    )
    started = int(time.time() * 1000)
    save_base64_image(image_b64, f"app/storage/input/{started}-{job['jobId']}.jpg")

    steps = job.get("steps", [])
    final_answers = []
    vision_plans = []
    counts: dict[str, int] = {}
    for index, step in enumerate(steps):
        logger.info(
            "Running step %s/%s for job=%s step=%s stepName=%s conditions=%s",
            index + 1,
            len(steps),
            job.get("jobId"),
            step.get("id"),
            step.get("name"),
            len(step.get("conditions") or []),
        )
        step_results, step_metadata = await _run_step(
            job,
            step,
            image_b64,
            resized_size,
            preprocessing,
        )
        detections.extend(step_results)
        if step_metadata.get("answer"):
            final_answers.append(step_metadata["answer"])
        if step_metadata.get("visionPlan"):
            vision_plans.append(step_metadata["visionPlan"])
        for label, count in (step_metadata.get("counts") or {}).items():
            counts[label] = counts.get(label, 0) + count
        _enrich_spatial_outputs(image_b64, detections, resized_size)
        output_image_url = _render_and_upload_output(
            job=job,
            image_b64=image_b64,
            detections=detections,
            started=started,
            suffix=f"step-{index + 1}",
        )
        logger.info(
            "Step complete for job=%s step=%s detections=%s total=%s output=%s",
            job.get("jobId"),
            step.get("id"),
            len(step_results),
            len(detections),
            output_image_url,
        )
        _debug("Step detections=%s", step_results)
        if index < len(steps) - 1:
            await _send_callback(
                {
                    "jobId": job.get("jobId"),
                    "jobItemId": _get_job_item_id(job),
                    "sessionId": job.get("sessionId"),
                    "status": "partial",
                    "imageUrl": job.get("imageUrl"),
                    "imageWidth": resized_size[0],
                    "imageHeight": resized_size[1],
                    "outputImageUrl": output_image_url,
                    "finalAnswer": _join_answers(final_answers),
                    "visionPlan": _vision_plan_payload(vision_plans),
                    "preprocessing": preprocessing,
                    "counts": counts,
                    "detections": detections,
                }
            )

    _enrich_spatial_outputs(image_b64, detections, resized_size)
    output_image_url = _render_and_upload_output(
        job=job,
        image_b64=image_b64,
        detections=detections,
        started=started,
        suffix="final",
    )
    await _send_callback(
        {
            "jobId": job.get("jobId"),
            "jobItemId": _get_job_item_id(job),
            "sessionId": job.get("sessionId"),
            "status": "completed",
            "imageUrl": job.get("imageUrl"),
            "imageWidth": resized_size[0],
            "imageHeight": resized_size[1],
            "outputImageUrl": output_image_url,
            "finalAnswer": _join_answers(final_answers),
            "visionPlan": _vision_plan_payload(vision_plans),
            "preprocessing": preprocessing,
            "counts": counts,
            "detections": detections,
        }
    )
    logger.info(
        "Job complete job=%s session=%s detections=%s",
        job.get("jobId"),
        job.get("sessionId"),
        len(detections),
    )
    _debug("Job final detections=%s", detections)


def _enrich_spatial_outputs(
    image_b64: str,
    detections: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> None:
    annotate_detection_clusters(detections, image_size)
    enrich_detections_with_masks(image_b64, detections)


async def _run_step(
    job: dict[str, Any],
    step: dict[str, Any],
    image_b64: str,
    image_size: tuple[int, int],
    preprocessing: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conditions = step.get("conditions") or []
    primary_condition = conditions[0] if conditions else {}
    parsed, _raw_text = await run_grounding_detection(
        image_base64=image_b64,
        image_size=image_size,
        condition_name=primary_condition.get("name") or step.get("name") or "visual",
        condition_type=primary_condition.get("type") or "visual",
        extra_prompt=job.get("prompt"),
        step_name=step.get("name"),
        step_description=step.get("description"),
        conditions=conditions,
        preprocessing=preprocessing,
        ollama_base_url=OLLAMA_BASE_URL,
        ollama_model=OLLAMA_MODEL,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )

    result = VisionResult.model_validate(parsed)
    logger.info(
        "Grounding result step=%s summary=%s checks=%s detections=%s",
        step.get("id"),
        _preview(result.summary, 240),
        len(result.checks),
        len(result.detections),
    )
    results = []
    metadata = {
        "answer": result.answer or result.summary,
        "visionPlan": result.vision_plan,
        "preprocessing": result.preprocessing,
        "counts": result.counts or {},
    }
    step_id = step.get("id")
    if result.summary:
        results.append(
            {
                "stepId": step_id,
                "title": f"{step.get('name') or 'Assessment'}: {result.status}",
                "description": result.summary,
                "abnormalityType": f"assessment:{result.status}",
                "coordX": 0,
                "coordY": 0,
                "width": 0,
                "height": 0,
                "colorHex": _status_color(result.status),
                "inputImageUrl": job.get("imageUrl"),
                "outputImageUrl": None,
                "suppressOverlay": True,
            }
        )

    for check in result.checks:
        bbox = check.bbox
        results.append(
            {
                "stepId": step_id,
                "title": check.name,
                "description": check.evidence or f"Status: {check.status}",
                "abnormalityType": f"check:{check.status}",
                "coordX": round(max(bbox.x, 0)) if bbox else 0,
                "coordY": round(max(bbox.y, 0)) if bbox else 0,
                "width": round(max(bbox.width, 0)) if bbox else 0,
                "height": round(max(bbox.height, 0)) if bbox else 0,
                "colorHex": _status_color(check.status),
                "inputImageUrl": job.get("imageUrl"),
                "outputImageUrl": None,
            }
        )

    for index, detection in enumerate(result.detections):
        bbox = detection.bbox
        label = detection.label or "Detection"
        x = round(max(bbox.x, 0)) if bbox else 0
        y = round(max(bbox.y, 0)) if bbox else 0
        width = round(max(bbox.width, 0)) if bbox else 0
        height = round(max(bbox.height, 0)) if bbox else 0
        item = {
            "stepId": step_id,
            "title": f"{label} #{index + 1}",
            "description": detection.description or detection.evidence,
            "abnormalityType": label,
            "coordX": x,
            "coordY": y,
            "width": width,
            "height": height,
            "colorHex": _instance_color(index),
            "conditionId": detection.condition_id,
            "inputImageUrl": job.get("imageUrl"),
            "outputImageUrl": None,
        }
        if detection.mask:
            item["mask"] = detection.mask
            item["segmentationMode"] = detection.segmentation_mode or "sam3"
            item["segmentationStatus"] = detection.segmentation_status or "ok"
        if detection.sam3_prompt:
            item["sam3Prompt"] = detection.sam3_prompt
        results.append(item)
    return results, metadata


def _join_answers(answers: list[str]) -> str:
    clean = [answer.strip() for answer in answers if answer and answer.strip()]
    return "\n".join(dict.fromkeys(clean))


def _vision_plan_payload(vision_plans: list[dict[str, Any]]) -> dict[str, Any]:
    if len(vision_plans) == 1:
        return vision_plans[0]
    return {"steps": vision_plans}


def _render_and_upload_output(
    job: dict[str, Any],
    image_b64: str,
    detections: list[dict[str, Any]],
    started: int,
    suffix: str,
) -> str:
    job_id = _safe_key_part(job.get("jobId") or "job")
    session_id = _safe_key_part(job.get("sessionId") or "session")
    user_id = _safe_key_part(job.get("userId") or "unknown")
    output_path = f"app/storage/output/{started}-{job_id}-{suffix}.jpg"
    rendered_detections = render_detection_overlay(
        image_b64,
        detections,
        output_path,
    )
    logger.info(
        "Rendered detections job=%s suffix=%s input=%s rendered=%s outputPath=%s",
        job.get("jobId"),
        suffix,
        len(detections),
        len(rendered_detections),
        output_path,
    )
    key = (
        f"data/{user_id}/spatial-vision/output/"
        f"{session_id}/{started}-{job_id}-{suffix}.jpg"
    )
    output_image_url = upload_file_to_s3(output_path, key)

    detections[:] = [
        {
            **detection,
            "outputImageUrl": output_image_url,
        }
        for detection in rendered_detections
    ]
    return output_image_url


def _safe_key_part(value: Any) -> str:
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "-"
        for char in str(value)
    ).strip("-") or "unknown"


def _debug(message: str, *args: Any) -> None:
    if VISION_DEBUG:
        logger.info(message, *args)


def _preview(value: Any, limit: int = 1000) -> str:
    text = str(value or "").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _status_color(status: str) -> str:
    if status == "pass":
        return "#16A34A"
    if status == "fail":
        return "#DC2626"
    if status == "not_applicable":
        return "#6B7280"
    return "#D97706"


def _instance_color(index: int) -> str:
    palette = [
        "#00AEEF",
        "#F97316",
        "#22C55E",
        "#E11D48",
        "#8B5CF6",
        "#FACC15",
        "#14B8A6",
        "#EC4899",
        "#3B82F6",
        "#A3E635",
    ]
    return palette[index % len(palette)]


async def _send_callback(payload: dict[str, Any]) -> None:
    if not payload.get("sessionId") and not payload.get("jobItemId") and not payload.get("job_item_id"):
        logger.warning(
            "Skipping callback because payload has no sessionId or jobItemId"
        )
        return

    headers = {}
    if CALLBACK_TOKEN:
        headers["x-spatial-vision-callback-token"] = CALLBACK_TOKEN

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(BACKEND_CALLBACK_URL, json=payload, headers=headers)
        response.raise_for_status()
    logger.info(
        "Callback sent job=%s session=%s status=%s detections=%s",
        payload.get("jobId"),
        payload.get("sessionId"),
        payload.get("status"),
        len(payload.get("detections", [])),
    )
