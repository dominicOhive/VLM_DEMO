def build_vision_planner_prompt(
    *,
    user_prompt: str,
    condition_name: str | None = None,
    step_name: str | None = None,
    step_description: str | None = None,
    conditions: list | None = None,
    preprocessing: dict | None = None,
) -> str:
    condition_lines = []
    for condition in conditions or []:
        name = _field(condition, "name")
        value = _field(condition, "value")
        kind = _field(condition, "type") or "visual"
        if name or value:
            condition_lines.append(f"- {name or value} ({kind}): {value or name}")

    quality = preprocessing or {}
    flags = ", ".join(quality.get("qualityFlags") or []) or "none"
    context = []
    if step_name:
        context.append(f"Step name: {step_name}")
    if step_description:
        context.append(f"Step description: {step_description}")
    if condition_name:
        context.append(f"Primary condition: {condition_name}")
    if condition_lines:
        context.append("Conditions:\n" + "\n".join(condition_lines))
    context_text = "\n".join(context) or "No saved sequence context."

    return f"""
You are the intent planner for a spatial vision chat pipeline.
Your job is to understand the user's visual request and decide whether SAM3 segmentation is needed.

User request:
{user_prompt}

Sequence context:
{context_text}

Image preprocessing:
- original_size: {quality.get("originalSize")}
- processed_size: {quality.get("processedSize")}
- scale_factor: {quality.get("scaleFactor")}
- quality_flags: {flags}

Return ONLY valid JSON as one object. Do not include markdown.

Schema:
{{
  "task_type": "detect | find | identify | count | assess | describe | mixed",
  "answer_mode": "text-only | segmentation | mixed",
  "needs_segmentation": true,
  "expected_output": "count | list | masks | explanation | assessment",
  "targets": [
    {{
      "label": "user-facing label",
      "prompt": "short singular SAM3 prompt",
      "synonyms": ["optional short prompt"],
      "reason": "why this target is useful"
    }}
  ],
  "constraints": {{
    "location": null,
    "material": null,
    "severity": null,
    "damage_only": false
  }},
  "quality_notes": ["short notes about blur, darkness, compression, low resolution, occlusion"],
  "warnings": [],
  "answer_hint": "one sentence explaining how to answer after segmentation"
}}

Rules:
- Use SAM3 for any visible localizable objects, items, regions, defects, structural components, or text areas specified by the user.
- CRITICAL: Always set "needs_segmentation": true and "answer_mode": "segmentation" or "mixed" for any prompts structured around identification, verification, or location mapping, specifically including questions starting with or containing "is this a...", "can you find...", "locate the...", or "find the...". Polygons must always be generated to provide a visual answer for these requests.
- If the user asks for generic anomalies, damages, or defects, set constraints.damage_only=true and produce narrow anomaly-level targets (e.g., crack, paint chip, rust spot, dent, scratch).
- If the user asks for real physical objects (e.g., "coffee cup", "fire extinguisher", "solar panel", "window frame"), map them cleanly as parent-level targets and keep constraints.damage_only=false.
- For broad damage requests, expand into concrete short singular prompts such as crack, paint chip, rust spot, dirt buildup.
- For narrow requests such as "find cracks", keep one target: crack.
- For count requests, set expected_output to count and still use segmentation when the target is countable.
- For identify or describe requests that do not ask for masks/locations/counts, set needs_segmentation false and answer_mode text-only.
- Every target prompt must be short, singular, and under 5 words. Do not use full user sentences.
- Prefer at most 8 targets. Use fewer when the user is specific.
""".strip()
def _field(condition, name: str):
    if isinstance(condition, dict):
        return condition.get(name)
    return getattr(condition, name, None)
