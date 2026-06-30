import asyncio
import base64
import io
import json
import logging
import math
import os
import re
from contextlib import nullcontext
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from app.ollama_client import call_ollama_vision, normalize_ollama_base_url
from app.prompts import build_vision_planner_prompt
from app.segmentation import mask_to_payload

logger = logging.getLogger("spatial_vision_worker.grounding")

OLLAMA_BASE_URL = normalize_ollama_base_url(os.getenv("OLLAMA_BASE_URL"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b")
QWEN_PLANNER_BACKEND = os.getenv("QWEN_PLANNER_BACKEND", "ollama").lower()
QWEN_MODEL_ID = os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-Instruct")
QWEN_DEVICE_MAP = os.getenv("QWEN_DEVICE_MAP", "auto")
QWEN_ATTN_IMPLEMENTATION = os.getenv("QWEN_ATTN_IMPLEMENTATION", "")
GROUNDING_MAX_NEW_TOKENS = int(os.getenv("GROUNDING_MAX_NEW_TOKENS", "1024"))
GROUNDING_NMS_IOU = float(os.getenv("GROUNDING_NMS_IOU", "0.65"))
QWEN_PLANNER_TIMEOUT_SECONDS = int(os.getenv("QWEN_PLANNER_TIMEOUT_SECONDS", "180"))
VISION_DEBUG = os.getenv("VISION_DEBUG", "false").lower() == "true"
SAM3_MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
SAM3_CHECKPOINT_PATH = os.getenv("SAM3_CHECKPOINT_PATH", "")
SAM3_BPE_PATH = os.getenv("SAM3_BPE_PATH", "")
SAM3_DEVICE = os.getenv("SAM3_DEVICE", "cuda")
SAM3_MIN_CONFIDENCE = float(os.getenv("SAM3_MIN_CONFIDENCE", "0.0"))
SAM3_AMP_DTYPE = os.getenv("SAM3_AMP_DTYPE", "bfloat16").lower()
MAX_SAM3_TARGETS = int(os.getenv("MAX_SAM3_TARGETS", "8"))
SAM3_TARGET_TIMEOUT_SECONDS = int(os.getenv("SAM3_TARGET_TIMEOUT_SECONDS", "180"))
MAX_ANOMALY_MASK_IMAGE_RATIO = float(os.getenv("SPATIAL_VISION_MAX_ANOMALY_MASK_IMAGE_RATIO", "0.85"))
MAX_ANOMALY_BBOX_IMAGE_RATIO = float(os.getenv("SPATIAL_VISION_MAX_ANOMALY_BBOX_IMAGE_RATIO", "0.98"))

DAMAGE_TARGETS = ("crack", "dent", "scratch", "stain", "chip", "corrosion", "missing part", "broken edge")


async def run_grounding_detection(
    *,
    image_base64: str,
    image_size: tuple[int, int],
    condition_name: str,
    condition_type: str = "visual",
    extra_prompt: str | None = None,
    step_name: str | None = None,
    step_description: str | None = None,
    conditions: list | None = None,
    preprocessing: dict[str, Any] | None = None,
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
    timeout_seconds: int = 180,
) -> tuple[dict[str, Any], str]:
    user_prompt = _user_prompt(
        condition_name=condition_name,
        extra_prompt=extra_prompt,
        step_name=step_name,
        step_description=step_description,
        conditions=conditions,
    )
    plan, raw_text = await _build_vision_plan(
        image_base64=image_base64,
        user_prompt=user_prompt,
        condition_name=condition_name,
        step_name=step_name,
        step_description=step_description,
        conditions=conditions,
        preprocessing=preprocessing or {},
        ollama_base_url=ollama_base_url or OLLAMA_BASE_URL,
        ollama_model=ollama_model or OLLAMA_MODEL,
        timeout_seconds=timeout_seconds,
    )
    plan = normalize_vision_plan(plan, user_prompt)
    _debug("Vision plan=%s", json.dumps(plan, ensure_ascii=False))

    if not plan.get("needs_segmentation"):
        answer = _text_only_answer(plan, user_prompt)
        return _vision_result(plan, [], answer, preprocessing), raw_text

    detections: list[dict[str, Any]] = []
    for target in plan.get("targets", [])[:MAX_SAM3_TARGETS]:
        primary_prompt = _compact_sam3_prompt(target.get("prompt") or target.get("label") or user_prompt)
        
        # Multi-prompt fallback strategy specifically engineered to isolate faint hairline defects
        prompts_to_try = [primary_prompt]
        if target.get("synonyms"):
            prompts_to_try.extend([_compact_sam3_prompt(s) for s in target["synonyms"]])
        is_crack_target = any(w in primary_prompt.lower() for w in ["crack", "fracture", "split", "fissure"])
        if is_crack_target:
            prompts_to_try.extend(["thin line", "dark fracture", "crevice line", "surface split"])

        target_detections = []
        for prompt in prompts_to_try:
            try:
                logger.info("SAM3 execution attempting variant: '%s' (label: %s)", prompt, target.get("label"))
                target_detections = await asyncio.wait_for(
                    asyncio.to_thread(
                        _run_sam3_target,
                        image_base64,
                        image_size,
                        prompt,
                        target,
                    ),
                    timeout=SAM3_TARGET_TIMEOUT_SECONDS,
                )
                if target_detections:
                    logger.info("SAM3 activated successfully on variant '%s' with %s results", prompt, len(target_detections))
                    break
            except asyncio.TimeoutError:
                logger.error("SAM3 timed out on prompt variant '%s'", prompt)
                continue
        detections.extend(target_detections)

    detections = _filter_detections_for_plan(detections, plan, image_size)
    detections = _dedupe_detections(detections)
    answer = _segmentation_answer(plan, detections, user_prompt)
    return _vision_result(plan, detections, answer, preprocessing), raw_text


async def _build_vision_plan(
    *,
    image_base64: str,
    user_prompt: str,
    condition_name: str | None,
    step_name: str | None,
    step_description: str | None,
    conditions: list | None,
    preprocessing: dict[str, Any],
    ollama_base_url: str,
    ollama_model: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], str]:
    prompt = build_vision_planner_prompt(
        user_prompt=user_prompt,
        condition_name=condition_name,
        step_name=step_name,
        step_description=step_description,
        conditions=conditions,
        preprocessing=preprocessing,
    )
    try:
        parsed, raw = await _call_qwen_planner(
            prompt=prompt,
            image_base64=image_base64,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            timeout_seconds=timeout_seconds,
        )
        repair_reason = _plan_repair_reason(parsed)
        if repair_reason:
            repair_prompt = _build_plan_repair_prompt(
                user_prompt=user_prompt,
                original_prompt=prompt,
                raw_text=raw,
                parsed=parsed,
                reason=repair_reason,
            )
            repaired, repair_raw = await _call_qwen_planner(
                prompt=repair_prompt,
                image_base64=image_base64,
                ollama_base_url=ollama_base_url,
                ollama_model=ollama_model,
                timeout_seconds=timeout_seconds,
            )
            return repaired, f"{raw}\n\n--- planner repair ---\n{repair_raw}"
        return parsed, raw
    except Exception as exc:
        logger.warning("Qwen planner failed; returning text-only failure plan: %s", exc)
        plan = _planner_failure_plan(str(exc), preprocessing)
        return plan, json.dumps({"planner_error": str(exc), "fallback_plan": plan})


async def _call_qwen_planner(
    *,
    prompt: str,
    image_base64: str,
    ollama_base_url: str,
    ollama_model: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], str]:
    planner_timeout = min(timeout_seconds, QWEN_PLANNER_TIMEOUT_SECONDS)
    if QWEN_PLANNER_BACKEND == "transformers":
        logger.info("Qwen planner start backend=transformers model=%s timeout=%ss", QWEN_MODEL_ID, planner_timeout)
        raw = await asyncio.wait_for(
            asyncio.to_thread(_run_transformers_qwen, prompt, image_base64),
            timeout=planner_timeout,
        )
        logger.info("Qwen planner complete backend=transformers rawBytes=%s", len(raw.encode("utf-8")))
        return _parse_json_object(raw), raw

    logger.info("Qwen planner start backend=ollama model=%s timeout=%ss", ollama_model, planner_timeout)
    parsed, raw = await call_ollama_vision(
        base_url=ollama_base_url,
        model=ollama_model,
        prompt=prompt,
        image_base64=image_base64,
        timeout_seconds=planner_timeout,
    )
    logger.info("Qwen planner complete backend=ollama rawBytes=%s", len(raw.encode("utf-8")))
    return parsed, raw


def normalize_vision_plan(plan: Any, user_prompt: str) -> dict[str, Any]:
    if not isinstance(plan, dict):
        plan = _planner_failure_plan("Planner returned a non-object response.", {})
    task_type = str(plan.get("task_type") or "mixed").lower()
    if task_type not in {"detect", "find", "identify", "count", "assess", "describe", "mixed"}:
        task_type = "mixed"

    needs_segmentation = bool(plan.get("needs_segmentation"))
    answer_mode = str(plan.get("answer_mode") or "").lower()
    if answer_mode not in {"text-only", "segmentation", "mixed"}:
        answer_mode = "segmentation" if needs_segmentation else "text-only"

    targets = []
    for item in plan.get("targets") or []:
        if not isinstance(item, dict):
            continue
        prompt = _compact_sam3_prompt(item.get("prompt") or item.get("label") or "")
        if not prompt:
            continue
        targets.append(
            {
                "label": str(item.get("label") or prompt).strip()[:80],
                "prompt": prompt,
                "synonyms": [
                    _compact_sam3_prompt(value)
                    for value in (item.get("synonyms") or [])
                    if _compact_sam3_prompt(value)
                ][:3],
                "reason": str(item.get("reason") or "").strip()[:240],
            }
        )

    if targets:
        needs_segmentation = True
        if answer_mode == "text-only":
            answer_mode = "mixed"

    expected_output = str(plan.get("expected_output") or "masks")

    return {
        "task_type": "detect" if needs_segmentation and task_type == "mixed" else task_type,
        "answer_mode": answer_mode,
        "needs_segmentation": needs_segmentation,
        "expected_output": expected_output,
        "targets": targets[:MAX_SAM3_TARGETS],
        "constraints": plan.get("constraints") if isinstance(plan.get("constraints"), dict) else {},
        "quality_notes": [str(note) for note in (plan.get("quality_notes") or [])][:8],
        "warnings": [str(note) for note in (plan.get("warnings") or [])][:8],
        "answer_hint": str(plan.get("answer_hint") or "").strip(),
    }


def _planner_failure_plan(error: str, preprocessing: dict[str, Any] | None = None) -> dict[str, Any]:
    quality_notes = list((preprocessing or {}).get("qualityFlags") or [])
    return {
        "task_type": "assess",
        "answer_mode": "text-only",
        "needs_segmentation": False,
        "expected_output": "assessment",
        "targets": [],
        "constraints": {},
        "quality_notes": quality_notes,
        "warnings": [f"Planner failed: {error}"],
        "answer_hint": "I could not create a reliable segmentation plan from the image and prompt.",
    }


def _plan_repair_reason(plan: Any) -> str | None:
    if not isinstance(plan, dict):
        return "planner response is not a JSON object"

    needs_segmentation = plan.get("needs_segmentation")
    answer_mode = str(plan.get("answer_mode") or "").lower()
    expected_output = str(plan.get("expected_output") or "").lower()
    targets = plan.get("targets") or []
    has_targets = isinstance(targets, list) and any(isinstance(item, dict) for item in targets)

    if needs_segmentation is True and not has_targets:
        return "needs_segmentation is true but targets is empty"
    if needs_segmentation is False and has_targets:
        return "needs_segmentation is false but targets were provided"
    if answer_mode == "text-only" and has_targets:
        return "answer_mode is text-only but targets were provided"
    if answer_mode in {"segmentation", "mixed"} and needs_segmentation is False:
        return "answer_mode asks for segmentation but needs_segmentation is false"
    if expected_output in {"masks", "count", "list"} and needs_segmentation is False:
        return "expected_output requires local results but needs_segmentation is false"
    if expected_output in {"masks", "count", "list"} and not has_targets:
        return "expected_output requires local results but targets is empty"
    return None


def _build_plan_repair_prompt(
    *,
    user_prompt: str,
    original_prompt: str,
    raw_text: str,
    parsed: Any,
    reason: str,
) -> str:
    return f"""
Your previous planner response was structurally invalid: {reason}.
User request:
{user_prompt}

Previously parsed JSON:
{json.dumps(parsed, ensure_ascii=False)[:4000]}

Raw previous response excerpt:
{str(raw_text or "")[:4000]}

Return ONLY corrected JSON using the same schema below.
Decide from the image and user request whether segmentation is needed. Do not infer from any hardcoded keyword list.
If segmentation is needed, provide concrete short English SAM3 targets. If no segmentation is needed, targets must be empty.
{original_prompt}
""".strip()


def _run_sam3_target(
    image_base64: str,
    image_size: tuple[int, int],
    prompt: str,
    target: dict[str, Any],
) -> list[dict[str, Any]]:
    processor, torch = _load_sam3_processor()
    image = _image_from_base64(image_base64)
    with torch.inference_mode(), _sam3_autocast_context(torch):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)

    return _sam3_output_to_detections(
        output,
        image_width=image_size[0],
        image_height=image_size[1],
        default_label=target.get("label") or prompt,
        prompt=prompt,
    )


def warmup_vision_model() -> None:
    if os.getenv("VISION_WARMUP_ON_STARTUP", "false").lower() != "true":
        logger.info("Vision model warmup skipped on startup; first request will load models")
        return

    _load_sam3_processor()
    if QWEN_PLANNER_BACKEND == "transformers" and os.getenv("QWEN_WARMUP_ON_STARTUP", "false").lower() == "true":
        _load_transformers_qwen()
    elif QWEN_PLANNER_BACKEND == "transformers":
        logger.info("Qwen planner warmup skipped; first job will load %s", QWEN_MODEL_ID)
    else:
        logger.info("Qwen planner backend=%s will warm through Ollama on first request", QWEN_PLANNER_BACKEND)


@lru_cache(maxsize=1)
def _load_transformers_qwen():
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2VLForConditionalGeneration,
        )
    except Exception as exc:
        raise RuntimeError(
            "Native Qwen planner requires torch, transformers, accelerate, qwen-vl-utils, and json-repair."
        ) from exc

    kwargs = {"dtype": "auto", "device_map": QWEN_DEVICE_MAP}
    if QWEN_ATTN_IMPLEMENTATION:
        kwargs["attn_implementation"] = QWEN_ATTN_IMPLEMENTATION

    logger.info("Loading native Qwen planner model=%s deviceMap=%s", QWEN_MODEL_ID, QWEN_DEVICE_MAP)
    if QWEN_MODEL_ID.startswith("Qwen/Qwen2-VL") or "/Qwen2-VL" in QWEN_MODEL_ID:
        loader = Qwen2VLForConditionalGeneration
    elif QWEN_MODEL_ID.startswith("Qwen/Qwen2.5-VL") or "/Qwen2.5-VL" in QWEN_MODEL_ID:
        loader = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported or unlisted offline Qwen VL model ID: {QWEN_MODEL_ID}")

    try:
        model = loader.from_pretrained(QWEN_MODEL_ID, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = loader.from_pretrained(QWEN_MODEL_ID, **kwargs)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    return model, processor, torch, process_vision_info


def _run_transformers_qwen(prompt: str, image_base64: str) -> str:
    model, processor, torch, process_vision_info = _load_transformers_qwen()
    image = _image_from_base64(image_base64)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{_image_to_base64(image)}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    model.eval()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=GROUNDING_MAX_NEW_TOKENS)
    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0] if output_text else ""


@lru_cache(maxsize=1)
def _load_sam3_processor():
    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except Exception as exc:
        raise RuntimeError(
            "SAM3 backend requires the facebookresearch/sam3 package, PyTorch, and accessible weights."
        ) from exc

    kwargs: dict[str, Any] = {}
    if SAM3_CHECKPOINT_PATH:
        kwargs["checkpoint_path"] = SAM3_CHECKPOINT_PATH
        kwargs["load_from_HF"] = False
    if SAM3_BPE_PATH:
        kwargs["bpe_path"] = SAM3_BPE_PATH
    if SAM3_DEVICE:
        device = SAM3_DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        kwargs["device"] = device

    model = build_sam3_image_model(**kwargs)
    model.eval()
    return Sam3Processor(model), torch


def _sam3_output_to_detections(
    output: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    default_label: str,
    prompt: str,
) -> list[dict[str, Any]]:
    boxes = _tensor_to_numpy(output.get("boxes"))
    masks = _tensor_to_numpy(output.get("masks"))
    scores = _tensor_to_numpy(output.get("scores"))
    if boxes is None or len(boxes) == 0:
        return []

    detections = []
    for index, raw_box in enumerate(boxes):
        confidence = float(scores[index]) if scores is not None and len(scores) > index else 1.0
        if confidence < SAM3_MIN_CONFIDENCE:
            continue

        bbox = _sam3_box_to_xyxy(raw_box, image_width, image_height)
        if not bbox:
            continue
        x1, y1, x2, y2 = _clamp_bbox(bbox, image_width, image_height)
        width = x2 - x1
        height = y2 - y1
        if width <= 1 or height <= 1:
            continue

        mask_payload = None
        if masks is not None and len(masks) > index:
            mask = _sam3_mask_to_binary(masks[index], image_width, image_height)
            # Create a structure that preserves the mask array payload internally
            initial_payload = {"type": "raw_mask", "binary_mask": mask}
            mask_payload = mask_to_payload(mask, (x1, y1, width, height), "sam3")
            if mask_payload:
                mask_bbox = mask_payload["bbox"]
                x1 = mask_bbox["x"]
                y1 = mask_bbox["y"]
                width = mask_bbox["width"]
                height = mask_bbox["height"]
            else:
                mask_payload = initial_payload

        detections.append(
            {
                "label": default_label,
                "description": f"SAM3 matched '{prompt}'.",
                "bbox": {"x": x1, "y": y1, "width": width, "height": height},
                "confidence": confidence,
                "evidence": "SAM3 text-prompt segmentation",
                "mask": mask_payload,
                "segmentation_mode": "sam3" if mask_payload else None,
                "segmentation_status": "ok" if mask_payload else "failed",
                "sam3_prompt": prompt,
            }
        )
    return detections


def _filter_detections_for_plan(
    detections: list[dict[str, Any]],
    plan: dict[str, Any],
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    constraints = plan.get("constraints") if isinstance(plan.get("constraints"), dict) else {}
    damage_like = bool(constraints.get("damage_only") or constraints.get("anomaly_only"))
    
    if not damage_like or plan.get("task_type") in ["find", "identify", "detect", "count"]:
        return detections

    image_area = max(int(image_size[0]) * int(image_size[1]), 1)
    filtered = []
    for detection in detections:
        bbox = detection.get("bbox") or {}
        bbox_area = max(int(bbox.get("width") or 0) * int(bbox.get("height") or 0), 0)
        mask = detection.get("mask") or {}
        mask_area = int(mask.get("area") or bbox_area)
        mask_ratio = mask_area / image_area
        bbox_ratio = bbox_area / image_area

        if mask_ratio > MAX_ANOMALY_MASK_IMAGE_RATIO:
            continue
        if bbox_ratio > MAX_ANOMALY_BBOX_IMAGE_RATIO:
            continue
        filtered.append(detection)
    return filtered


def _vision_result(
    plan: dict[str, Any],
    detections: list[dict[str, Any]],
    answer: str,
    preprocessing: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "result_type": "mixed" if detections and answer else "detection" if detections else "assessment",
        "status": "pass" if detections else "uncertain",
        "summary": answer,
        "answer": answer,
        "vision_plan": plan,
        "preprocessing": preprocessing or {},
        "counts": _counts_by_label(detections),
        "checks": [],
        "detections": detections,
    }


def _text_only_answer(plan: dict[str, Any], user_prompt: str) -> str:
    notes = plan.get("quality_notes") or []
    suffix = f" Image quality notes: {', '.join(notes)}." if notes else ""
    return (plan.get("answer_hint") or f"This request is best answered as a visual description: {user_prompt}.") + suffix


def _segmentation_answer(plan: dict[str, Any], detections: list[dict[str, Any]], user_prompt: str) -> str:
    count = len(detections)
    counts = _counts_by_label(detections)
    if count == 0:
        targets = ", ".join(target.get("label") or target.get("prompt") for target in plan.get("targets", []))
        return f"I did not find localized matches for {targets or user_prompt}."
    if plan.get("expected_output") == "count" or plan.get("task_type") == "count":
        return f"I found {count} localized result{'s' if count != 1 else ''}."
    label_text = ", ".join(f"{label}: {value}" for label, value in counts.items())
    return f"I found {count} localized result{'s' if count != 1 else ''}: {label_text}."


def _counts_by_label(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detection in detections:
        label = str(detection.get("label") or "Detection")
        counts[label] = counts.get(label, 0) + 1
    return counts


def _user_prompt(
    *,
    condition_name: str,
    extra_prompt: str | None,
    step_name: str | None,
    step_description: str | None,
    conditions: list | None,
) -> str:
    parts = []
    if extra_prompt:
        parts.append(str(extra_prompt))
    for condition in conditions or []:
        value = condition.get("value") if isinstance(condition, dict) else getattr(condition, "value", None)
        name = condition.get("name") if isinstance(condition, dict) else getattr(condition, "name", None)
        if value or name:
            parts.append(str(value or name))
    parts.extend(str(value) for value in (condition_name, step_name, step_description) if value)
    return "; ".join(dict.fromkeys(part.strip() for part in parts if part and part.strip())) or "visible object"


def _compact_sam3_prompt(text: Any) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    quoted = re.findall(r"['\"]([^'\"]{1,80})['\"]", text)
    if quoted:
        text = quoted[0].strip()
    text = re.sub(
        r"^(detect|find|segment|locate|identify|count|highlight|check for|look for|show me)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.split(".", 1)[0].split(",", 1)[0].split(";", 1)[0].strip()
    words = text.split()
    if words:
        words[-1] = _singularize_prompt_word(words[-1])
    return " ".join(words[:5]).strip()


def _singularize_prompt_word(word: str) -> str:
    if len(word) > 3 and word.lower().endswith("ies"):
        return f"{word[:-3]}y"
    if len(word) > 3 and word.lower().endswith("s") and not word.lower().endswith(("ss", "us")):
        return word[:-1]
    return word


def _sam3_autocast_context(torch: Any):
    if SAM3_DEVICE == "cpu" or SAM3_AMP_DTYPE in ("", "none", "false", "off", "fp32", "float32"):
        return nullcontext()

    dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }.get(SAM3_AMP_DTYPE)
    if dtype is None:
        return nullcontext()

    device_type = "cuda" if SAM3_DEVICE.startswith("cuda") else SAM3_DEVICE
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return torch.autocast(device_type=device_type, dtype=dtype)


def _tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _sam3_box_to_xyxy(
    raw_box: Any,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    values = np.asarray(raw_box, dtype=np.float32).reshape(-1)
    if len(values) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in values[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= image_width
        x2 *= image_width
        y1 *= image_height
        y2 *= image_height
    return x1, y1, x2, y2


def _sam3_mask_to_binary(mask: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    binary = (mask > 0).astype(np.uint8) * 255
    if binary.shape[:2] != (image_height, image_width):
        import cv2
        binary = cv2.resize(binary, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return binary.astype("uint8")


def _parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Model returned an empty planner response")
    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        from json_repair import repair_json
        text = repair_json(text)
    except Exception:
        text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if not match:
            raise
        snippet = re.sub(r",\s*([}\]])", r"\1", match.group(1))
        parsed = json.loads(snippet)
    if not isinstance(parsed, dict):
        raise ValueError("Planner JSON response must be an object")
    return parsed


def _clamp_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left = int(round(max(min(x1, x2), 0)))
    top = int(round(max(min(y1, y2), 0)))
    right = int(round(min(max(x1, x2), image_width)))
    bottom = int(round(min(max(y1, y2), image_height)))
    left = min(left, max(image_width - 1, 0))
    top = min(top, max(image_height - 1, 0))
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return left, top, right, bottom


def _dedupe_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for detection in sorted(detections, key=lambda item: item.get("confidence", 0), reverse=True):
        if all(_iou(detection["bbox"], existing["bbox"]) < GROUNDING_NMS_IOU for existing in kept):
            kept.append(detection)
    return kept


def _iou(a: dict[str, int], b: dict[str, int]) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union else 0.0


def _image_from_base64(image_base64: str) -> Image.Image:
    return Image.open(io.BytesIO(_raw_image_bytes(image_base64))).convert("RGB")


def _image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _raw_image_bytes(image_base64: str) -> bytes:
    if "," in image_base64 and image_base64.split(",", 1)[0].startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    return base64.b64decode(image_base64)


def _debug(message: str, *args: Any) -> None:
    if VISION_DEBUG:
        logger.info(message, *args)