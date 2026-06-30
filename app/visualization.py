import cv2
import numpy as np
import base64

def render_detection_overlay(image_base64: str, detections: list, output_path: str) -> list:
    """
    Decodes an image from base64, applies translucent fills inside mask coordinates, 
    and draws definitive structural border boxes around identified damage anomalies
    """
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    encoded = base64.b64decode(image_base64)
    np_arr = np.frombuffer(encoded, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return detections

    for detection in detections:
        # Check for mask polygon points [cite: 283]
        mask = detection.get("mask")
        if mask and "polygons" in mask:
            for polygon in mask["polygons"]:
                pts = np.array(polygon, dtype=np.int32)
                # Create semi-transparent overlay mask for shading the crack [cite: 284]
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], (239, 174, 0)) # Highlight Blue/Amber shade [cite: 284]
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img) # [cite: 284]
                cv2.polylines(img, [pts], True, (255, 23, 0), 2) # Outer line border [cite: 284]

        # Render standard tracking bounding box fallback 
        bbox = detection.get("bbox")
        if bbox:
            # Handle both dictionary formats dynamically
            x = int(bbox.get("x") if "x" in bbox else bbox.get("coordX", 0))
            y = int(bbox.get("y") if "y" in bbox else bbox.get("coordY", 0))
            w = int(bbox.get("width") if "width" in bbox else bbox.get("width", 0))
            h = int(bbox.get("height") if "height" in bbox else bbox.get("height", 0))
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 110, 239), 2) 
            label = detection.get("label", "Anomaly")
            cv2.putText(img, label, (x, max(y - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 110, 239), 2) 

    # Save output back onto system disk cache layer
    cv2.imwrite(output_path, img)
    return detections