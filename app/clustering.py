import math
import os
from typing import Any


CLUSTER_MIN_SIZE = int(os.getenv("SPATIAL_VISION_CLUSTER_MIN_SIZE", "3"))
CLUSTER_DISTANCE_RATIO = float(
    os.getenv("SPATIAL_VISION_CLUSTER_DISTANCE_RATIO", "0.14")
)


def annotate_detection_clusters(
    detections: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    for index, detection in enumerate(detections):
        detections[index] = {
            key: value
            for key, value in detection.items()
            if key
            not in (
                "groupId",
                "groupIndex",
                "groupSize",
                "groupBbox",
                "suppressOverlay",
            )
        }

    if len(detections) < CLUSTER_MIN_SIZE:
        return detections

    image_width, image_height = image_size
    diagonal = math.hypot(image_width, image_height)
    distance_limit = max(48.0, diagonal * CLUSTER_DISTANCE_RATIO)
    indexed = [
        (index, detection)
        for index, detection in enumerate(detections)
        if _has_bbox(detection)
    ]
    groups = _connected_groups(indexed, distance_limit)

    for group_index, group in enumerate(groups, start=1):
        if len(group) < CLUSTER_MIN_SIZE:
            continue

        group_id = f"cluster-{group_index}"
        group_bbox = _group_bbox([detections[index] for index in group])
        for position, detection_index in enumerate(group, start=1):
            detections[detection_index] = {
                **detections[detection_index],
                "groupId": group_id,
                "groupIndex": position,
                "groupSize": len(group),
                "groupBbox": group_bbox,
            }

    return detections


def _connected_groups(
    indexed: list[tuple[int, dict[str, Any]]],
    distance_limit: float,
) -> list[list[int]]:
    remaining = {index for index, _detection in indexed}
    by_index = dict(indexed)
    groups = []

    while remaining:
        seed = remaining.pop()
        group = [seed]
        queue = [seed]
        while queue:
            current = queue.pop(0)
            current_detection = by_index[current]
            neighbors = [
                candidate
                for candidate in list(remaining)
                if _same_cluster_family(current_detection, by_index[candidate])
                and _center_distance(current_detection, by_index[candidate])
                <= distance_limit
            ]
            for neighbor in neighbors:
                remaining.remove(neighbor)
                queue.append(neighbor)
                group.append(neighbor)
        groups.append(sorted(group))

    return groups


def _same_cluster_family(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        str(a.get("stepId") or "") == str(b.get("stepId") or "")
        and str(a.get("abnormalityType") or a.get("title") or "").lower()
        == str(b.get("abnormalityType") or b.get("title") or "").lower()
    )


def _center_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.hypot(ax - bx, ay - by)


def _center(detection: dict[str, Any]) -> tuple[float, float]:
    return (
        float(detection.get("coordX") or 0) + float(detection.get("width") or 0) / 2,
        float(detection.get("coordY") or 0) + float(detection.get("height") or 0) / 2,
    )


def _group_bbox(detections: list[dict[str, Any]]) -> dict[str, int]:
    x1 = min(int(detection.get("coordX") or 0) for detection in detections)
    y1 = min(int(detection.get("coordY") or 0) for detection in detections)
    x2 = max(
        int(detection.get("coordX") or 0) + int(detection.get("width") or 0)
        for detection in detections
    )
    y2 = max(
        int(detection.get("coordY") or 0) + int(detection.get("height") or 0)
        for detection in detections
    )
    return {
        "x": x1,
        "y": y1,
        "width": max(x2 - x1, 0),
        "height": max(y2 - y1, 0),
    }


def _has_bbox(detection: dict[str, Any]) -> bool:
    return (
        int(detection.get("width") or 0) > 1
        and int(detection.get("height") or 0) > 1
    )
