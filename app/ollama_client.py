import json
import logging
import os
import re
from urllib.parse import urlparse

try:
    from json_repair import repair_json
except Exception:
    repair_json = None

logger = logging.getLogger("spatial_vision_worker.ollama")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "low")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
VISION_DEBUG = os.getenv("VISION_DEBUG", "false").lower() == "true"


def normalize_ollama_base_url(base_url: str | None = None) -> str:
    configured = (base_url or os.getenv("OLLAMA_BASE_URL") or "").strip()
    if not configured:
        return "http://127.0.0.1:11434"

    parsed = urlparse(configured)
    if parsed.scheme and parsed.netloc:
        if parsed.hostname in {"ollama"}:
            return "http://127.0.0.1:11434"
        return configured

    return configured or "http://127.0.0.1:11434"


def _candidate_ollama_urls(base_url: str | None = None) -> list[str]:
    configured = normalize_ollama_base_url(base_url)
    candidates = [configured]
    parsed = urlparse(configured)
    if parsed.hostname == "ollama":
        candidates.append("http://127.0.0.1:11434")
    elif configured.endswith(":11434") and parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        candidates.append("http://127.0.0.1:11434")
    return list(dict.fromkeys(candidates))


def extract_json(text: str) -> dict:
    """
    Handles cases where the model accidentally returns text around JSON.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Model returned an empty response")

    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"```$", "", text).strip()

    try:
        repaired = repair_json(text) if repair_json else text
        parsed = json.loads(repaired)
    except json.JSONDecodeError as exc:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not match:
            preview = text[:500].replace("\n", "\\n")
            raise ValueError(f"Model returned non-JSON text: {preview}") from exc
        snippet = match.group(0)
        if repair_json:
            snippet = repair_json(snippet)
        parsed = json.loads(snippet)

    if isinstance(parsed, list):
        return {
            "result_type": "detection" if parsed else "assessment",
            "status": "pass" if parsed else "uncertain",
            "summary": f"Found {len(parsed)} localized result{'s' if len(parsed) != 1 else ''}.",
            "checks": [],
            "detections": parsed,
        }
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON response must be an object")
    if "result_type" not in parsed:
        parsed["result_type"] = "mixed" if parsed.get("detections") else "assessment"
    if "status" not in parsed:
        parsed["status"] = "uncertain"
    if "summary" not in parsed:
        parsed["summary"] = ""
    if "checks" not in parsed:
        parsed["checks"] = []
    if "detections" not in parsed:
        parsed["detections"] = []
    return parsed


def _prompt_text(prompt: str) -> str:
    return f"/no_think\n{prompt}"


def _chat_payload(
    model: str,
    prompt: str,
    image_base64: str,
    json_mode: bool = True,
) -> dict:
    payload = {
        "model": model,
        "prompt": _prompt_text(prompt),
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }
    if json_mode:
        payload["format"] = "json"
    return payload


def _generate_payload(
    model: str,
    prompt: str,
    image_base64: str,
    json_mode: bool = True,
) -> dict:
    return {
        "model": model,
        "prompt": _prompt_text(prompt),
        "images": [image_base64],
        "stream": False,
        **({"format": "json"} if json_mode else {}),
        "think": OLLAMA_THINK,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }


async def call_ollama_vision(
    base_url: str,
    model: str,
    prompt: str,
    image_base64: str,
    timeout_seconds: int = 600,
) -> tuple[dict, str]:
    import httpx

    raw_text = ""
    timeout = httpx.Timeout(
        timeout_seconds,
        connect=30,
        read=timeout_seconds,
        write=60,
        pool=30,
    )
    payload = _chat_payload(model, prompt, image_base64, True)
    last_error: Exception | None = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        attempt_name = "generate_json"
        for attempt_index, candidate_url in enumerate(_candidate_ollama_urls(base_url)):
            url = f"{candidate_url}/api/generate"
            logger.info(
                "Ollama attempt=%s url=%s timeoutSeconds=%s numPredict=%s",
                attempt_name,
                url,
                timeout_seconds,
                OLLAMA_NUM_PREDICT,
            )
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                raw_text = data.get("response") or ""
                thinking = data.get("thinking") or ""

                if not raw_text.strip() and thinking.strip():
                    raw_text = thinking

                logger.info(
                    "Ollama attempt=%s done_reason=%s contentBytes=%s thinkingBytes=%s",
                    attempt_name,
                    data.get("done") or data.get("done_reason"),
                    len(raw_text.encode("utf-8")),
                    len(thinking.encode("utf-8")),
                )

                if raw_text.strip():
                    if VISION_DEBUG:
                        logger.info(
                            "Ollama raw_text=%s",
                            raw_text[:5000].replace("\n", "\\n"),
                        )
                    return extract_json(raw_text), raw_text

                break
            except httpx.ReadTimeout as exc:
                last_error = TimeoutError(
                    f"Ollama {attempt_name} timed out after {timeout_seconds}s"
                )
                logger.warning("Ollama attempt=%s failed url=%s: %s", attempt_name, url, exc)
            except httpx.RequestError as exc:
                last_error = exc
                logger.warning("Ollama attempt=%s failed url=%s: %s", attempt_name, url, exc)
            except Exception as exc:
                last_error = exc
                logger.warning("Ollama attempt=%s failed url=%s: %s", attempt_name, url, exc)
                break

            if attempt_index < len(_candidate_ollama_urls(base_url)) - 1:
                continue

    logger.error(
        "Failed to parse Ollama response. raw_preview=%s",
        (raw_text or "")[:1000].replace("\n", "\\n"),
    )
    if isinstance(last_error, TimeoutError):
        raise last_error
    if last_error is not None:
        raise ConnectionError(f"Ollama request failed: {last_error}") from last_error
    raise ValueError("Model returned an empty response")
