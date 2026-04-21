from typing import Dict

import cv2


def evaluate_frame_quality(frame, low_light_thresh=45.0, blur_thresh=35.0) -> Dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    low_light = brightness < float(low_light_thresh)
    blurry = blur_var < float(blur_thresh)
    quality_score = 1.0
    if low_light:
        quality_score -= 0.45
    if blurry:
        quality_score -= 0.35
    quality_score = max(0.0, min(1.0, quality_score))
    return {
        "brightness": brightness,
        "blur_var": blur_var,
        "low_light": low_light,
        "blurry": blurry,
        "critical_low_light": brightness < (float(low_light_thresh) * 0.6),
        "quality_score": quality_score,
    }
