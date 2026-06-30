import sys

sys.path.insert(0, "/home/ubuntu")

from app import grounding


def test_warmup_vision_model_skips_loading_by_default(monkeypatch):
    calls = []

    monkeypatch.delenv("VISION_WARMUP_ON_STARTUP", raising=False)
    monkeypatch.setattr(grounding, "QWEN_PLANNER_BACKEND", "ollama")
    monkeypatch.setattr(grounding, "_load_sam3_processor", lambda: calls.append("sam3") or object())

    grounding.warmup_vision_model()

    assert calls == []
